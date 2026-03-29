# In Progress

- Data Loading...

# Completed

## CleanUMamba Cherry-Pick Extraction List

The original `CleanUMamba` codebase contains several highly optimized utilities for managing edge-case streaming logic. Once the core PyTorch Lightning architecture is stood up, extract the following elements into the local codebase:

### 1. The Streaming Inference Engine (`feed`, `flush`, and `_denoise_frame`)
* **Why we need it:** The standard PyTorch `forward` pass processes an entire 10-second audio clip at once. For live comms, we must process continuous 10-millisecond buffers. The `feed` function demonstrates exactly how to manage the `self.pending` audio buffer. Crucially, `_denoise_frame` shows how to cache the Mamba hidden states (`self.inference_params`) and the Convolutional overlaps (`self.encoder_decoder_state`) between function callbacks.
* **The Translation:** When porting this inference loop into MittelVec, this Python state-caching logic maps 1:1 to how we allocate and hold state across C++ audio callbacks.

### 2. The Mathematical Padding Calculators (`valid_length` and `pad_signal`)
* **Why we need it:** If an audio file is passed into the U-Net that isn't perfectly divisible by the total stride (e.g., $2^6 = 64$), the downsampling and upsampling math will misalign, truncating the output audio. `valid_length` automatically calculates the nearest valid sequence length so the convolutions process perfectly, and `pad_signal` appends the required zeros.

### 3. The Shape-Agnostic Weight Loader (`load_pruned_state_dict` & `prune_model_from_state_dict_shapes`)
* **Why we need it:** When we eventually apply the 8x structured pruning strategy, the size of the linear layers and convolutional channels will shrink dynamically. Trying to load pruned PyTorch weights into the standard `EdgeAudioUSE` class will cause a `Size Mismatch` crash. This nested function reads incoming weight dimensions and dynamically resizes the `nn.Module` channels on the fly to fit the pruned weights perfectly.

### 4. External Imports to Hunt Down
Find and extract the following from the original repository:
* **`src/network/layers.py`:** Locate the `Activation` class. This contains a custom GLU (Gated Linear Unit) used as the primary activation function in the encoder. Replacing the PoC's `nn.ReLU()` with this GLU improves perceptual warmth and frequency response.
* **`src/util/util.py`:** Locate `weight_scaling_init`. This applies a custom weight initialization scaling to the convolutions, which prevents exploding gradients when the audio passes into the Mamba bottleneck.