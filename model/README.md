# Outdated...

# EdgeAudioUSE (Universal Speech Enhancer)

**EdgeAudioUSE** is a causal, stateful 1D Convolutional U-Net fused with a State Space Model (Mamba) bottleneck. 

While the core topology is inspired by CleanUMamba, this implementation has been heavily customized and re-engineered for **real-time edge inference**. It is designed to perform simultaneous noise suppression, dereverberation, and Bandwidth Extension (BWE) directly on raw waveforms, maintaining a sub-20ms algorithmic latency suitable for live B2B VoIP and telehealth communications. 

Crucially, this architecture bridges the gap between Python-based offline training and stateful C++ edge deployment by emulating continuous memory buffers directly within the PyTorch graph.

---

## Core Architecture

The network acts as an autoencoder, compressing raw audio into a dense latent space so the Mamba blocks can track global acoustic context without thrashing CPU caches, before upsampling back to a clean waveform.

### 1. The Encoder (Analysis)
* **Structure:** A deep stack of 1D convolutions that progressively downsample the temporal resolution while doubling the channel capacity (capped at 768).
* **Activation:** Uses a custom Gated Linear Unit (GLU) with `SiLU` (Swish) activations to extract complex, non-linear acoustic textures and provide "analog warmth."
* **Causality:** Utilizes `AdjustableConv1d` to enforce strict left-sided padding. The network cannot look into the future, guaranteeing zero algorithmic lookahead latency.

### 2. The Bottleneck (Memory)
* **Structure:** `UnifiedMambaBlock` handles the State Space Model (SSM) integration. 
* **Why Mamba:** Standard Transformers scale quadratically with sequence length, making them impossible to run on edge CPUs. Mamba scales linearly, allowing the model to track long-term room reverberation (tail lengths) with a fixed memory footprint.

### 3. The Decoder (Synthesis)
* **Structure:** A stack of `ConvTranspose1d` layers (`AdjustableConvTranspose1d`) that project the compressed latent state back into a high-resolution waveform.
* **Activation:** Uses standard `ReLU` to maintain mathematical stability and sparsity during waveform reconstruction, preventing high-frequency hallucinations.
* **Skip Connections:** Standard U-Net skip connections route the high-resolution features from the encoder directly into the decoder to reconstruct transient details (consonants, snare hits).

---

## The Streaming Inference Engine

Standard PyTorch is stateless. Passing a 10-second audio file through a U-Net calculates the receptive fields for the entire file at once. In live VoIP, audio arrives in continuous 10-millisecond buffers. If fed naively into a stateless U-Net, the convolutional layers and Mamba blocks will drop their acoustic memory at every boundary, causing severe audio stuttering.

This model solves this via the `StreamingInferenceMixin` and the `_cached_streaming_forward` method, which emulate the exact memory allocation strategy required for the C++ SDK.

### The Tensor Flow (`_cached_streaming_forward`)

When operating in live inference mode (`model.feed()`), the model bypasses the standard `forward()` pass and executes a custom execution loop to maintain temporal continuity:

1. **Overlap-Save (Encoder):** Strided convolutions require context from the past. The engine retrieves the trailing edge of the previous 10ms frame from `self.encoder_cache` and `self.decoder_cache`, concatenates it to the front of the current tensor, aligns the sliding window, and saves the new trailing edge for the next callback.
2. **State Relay (Mamba):** The compressed latent tensor is passed to the Mamba blocks along with `self.inference_params`. Mamba retrieves the historical hidden state ($h_t$), updates its memory using the new frame, and overwrites the cache so the room context persists across boundaries.
3. **Overlap-Add (Decoder):** Transposed convolutions "splat" data, meaning the output of Frame A mathematically bleeds into the time-domain of Frame B. The engine slices off this excess trailing edge, saves it, and mathematically adds it to the leading edge of the next frame to prevent clicks and pops.

> **Note: The Double-Count Bias Paradox**
> During the decoder's overlap-add phase, the network's bias is manually subtracted from the cached trailing edge before saving it. Because $y = Wx + b$, adding two raw outputs together would result in $(Wx_A + b) + (Wx_B + b) = Wx_A + Wx_B + 2b$. Stripping the bias prevents a compounding DC offset glitch at the buffer boundaries.

---

## Mathematical & Structural Optimizations

### Macro vs. Micro Padding
To prevent PyTorch from breaking the audio stream, the model handles two distinct types of padding:
* **Micro-Padding (`conv_blocks.py`):** Shifts the padding inside the convolutional kernels entirely to the left (the past) to prevent causality leaks.
* **Macro-Padding (`padding.py`):** A U-Net with 6 layers and a stride of 2 compresses audio by a factor of 64 ($2^6$). If an input sequence is not a perfect multiple of 64, floor division will permanently truncate trailing samples. `pad_signal` appends dummy zeros to the end of the sequence to guarantee it survives the bottleneck intact.

### Gradient Stabilization (`init.py`)
Mamba models rely on a continuous mathematical recurrence loop. If the initial weights of the convolutional encoder are too large, they mathematically compound inside the bottleneck, leading to exploding gradients (`NaN` loss) early in the training run.
* **Solution:** A custom `weight_scaling_init` algorithm (based on CPC architecture) forces the variance of all convolutional filters down before training begins.

### Loss Function
The model optimizes against two simultaneous objectives:
* **Time-Domain (`L1 Loss`):** Enforces strict mathematical adherence to the target waveform's physical structure.
* **Frequency-Domain (`MultiResolutionSTFTLoss`):** Calculates Spectral Convergence and Log-Magnitude loss across three distinct physical time windows (15ms, 37.5ms, 75ms). This forces the network to accurately reconstruct spectral harmonics and phase stability, optimizing for human auditory perception rather than just raw math.

---

## Testing the Streaming Logic (The U-Net Isolation Test)

Before deploying to a C++ environment, the Python streaming logic must be validated. Because pure-Python Mamba fallbacks (used on local machines without Triton/CUDA) cannot perfectly persist the `inference_params` state, you must run the **U-Net Isolation Test**.

1. Temporarily replace the Mamba blocks in both `forward()` and `_enhance_frame_step()` with `nn.Identity()`.
2. Pass a `.wav` file through the offline `forward()` pass (The Answer Key).
3. Pass the same `.wav` file in 10ms chunks through `feed()` (The Simulation).
4. Assert `torch.allclose()`. If the U-Net overlap math is perfect, the outputs will be bit-for-bit identical.