import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import auraloss
from modules.unified_mamba_block import UnifiedMambaBlock
from utils.init import weight_scaling_init
from utils.padding import pad_signal, get_valid_length
from utils.streaming import StreamingInferenceMixin
from layers import EncoderBlock, DecoderBlock
from typing import cast

# --- The PyTorch Lightning Wrapper ---

class EdgeAudioUSE(pl.LightningModule, StreamingInferenceMixin):
    """
    Universal Speech Enhancer (USE) optimized for real-time edge inference.

    This model fuses a causal 1D Convolutional U-Net topology with a State Space Model 
    (Mamba) bottleneck. It is designed to perform simultaneous noise suppression, 
    dereverberation, and Bandwidth Extension (BWE) directly on raw waveforms, while 
    maintaining a sub-20ms algorithmic latency suitable for live VoIP communications.

    The architecture compresses raw audio into a dense latent space so the Mamba 
    blocks can track global acoustic context without thrashing the CPU's L1 cache, 
    before upsampling back to a clean waveform.

    Hyperparameters:
        channels_H (int): The base channel capacity of the network. The encoder will 
            progressively double this capacity at each layer (capped at 768) to 
            capture complex acoustic textures. Defaults to 64.
        num_encoder_layers (int): The depth of the U-Net. Dictates the temporal downsampling 
            ratio. 6 layers at a stride of 2 yields a 64x compression rate, ensuring 
            the 12ms target latency. Defaults to 6.
        d_model (int): The internal dimension of the Mamba bottleneck layers. 
            A larger dimension increases expressive power for complex reverb removal 
            but quadratically increases MACs. Defaults to 512.
        kernel_size (int): The size of the convolving kernel used in all encoder and 
            decoder blocks. Dictates the local receptive field (how many raw audio samples 
            are analyzed per step). Defaults to 4.
        stride (int): The downsampling/upsampling rate per layer. Combined with 
            `num_encoder_layers`, this determines the total temporal compression ratio 
            (e.g., a stride of 2 across 6 layers yields 64x compression). This mathematically 
            dictates the minimum valid frame length and controls algorithmic latency. Defaults to 2.
        lookahead (int or str): The global causality toggle for the entire network. 
            - Strictly Causal (lookahead=0): Pads only the past. Guarantees zero algorithmic 
            latency for real-time edge inference.
            - Micro-Lookahead (lookahead>0): Sacrifices a specific number of frames (e.g., 2ms) 
            of latency to give the model future context. Used to test the BWE Causal Paradox 
            (whether the model needs to see the end of a transient to reconstruct high frequencies).
            - Non-Causal (lookahead='centered'): Traditional offline processing for theoretical baselines.
            Defaults to 0.
    """
    def __init__(self, d_model=512, kernel_size=4, stride=2, lookahead=0,):
        super().__init__()
        self.save_hyperparameters()
        self.lookahead = lookahead
        self.kernel_size = kernel_size
        self.stride = stride

        """
        Multi-Resolution STFT Loss (Frequency-Domain Perception)
        Calculates spectrogram loss across three distinct physical time windows (assuming 16kHz audio):
        1. win_length=240 (15ms): Micro-window. Forces accurate reconstruction of sharp transients/consonants.
        2. win_length=600 (37.5ms): Standard window. Captures general speech articulation.
        3. win_length=1200 (75ms): Macro-window. Forces accurate, phase-stable reconstruction of sustained vowels.
        The hop_sizes dictate an ~80% overlap to ensure smooth gradient flow during backpropagation.
        """
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[50, 120, 240],
            win_lengths=[240, 600, 1200]
        )

        # ---------------------------------------------------------
        # Declarative U-Net Architecture
        # ---------------------------------------------------------
        
        # 1. The Encoder (Walking up the channel ladder)
        # Input to the first layer is 1 (Mono Audio)
        enc_channels = [
            (1, 64), 
            (64, 128), 
            (128, 256), 
            (256, 512), 
            (512, 768), 
            (768, 768)
        ]
        self.num_encoder_layers = len(enc_channels)
        
        self.encoder = nn.ModuleList([
            EncoderBlock(in_c, out_c, kernel_size, stride, lookahead) 
            for in_c, out_c in enc_channels
        ])

        # 2. The Mamba Bottleneck (Operating at peak depth: 768)
        self.proj_in = nn.Conv1d(768, d_model, kernel_size=1)
        self.mamba_blocks = nn.ModuleList([
            UnifiedMambaBlock(d_model=d_model, d_state=64) for _ in range(3)
        ])
        self.proj_out = nn.Conv1d(d_model, 768, kernel_size=1)

        # 3. The Decoder (Walking down the ladder, catching skips, ending in Mono Audio)
        dec_channels = [
            (768, 768), 
            (768, 512), 
            (512, 256), 
            (256, 128), 
            (128, 64), 
            (64, 1)      # Collapse back to mono audio
        ]
        
        self.decoder = nn.ModuleList()
        for i, (in_c, out_c) in enumerate(dec_channels):
            # No ReLU on the absolute final layer (Index 5)
            is_last = (i == len(dec_channels) - 1)
            self.decoder.append(
                DecoderBlock(in_c, out_c, kernel_size, stride, is_last, lookahead)
            )

        # ---------------------------------------------------------
        # Streaming Initialization (streaming.py)
        # Initialize the buffers and state tracking variables required 
        # by the StreamingInferenceMixin.
        # ---------------------------------------------------------

        # Calculate the required frame length for the streaming buffer
        self.frame_length = get_valid_length(1, self.num_encoder_layers, self.kernel_size, self.stride)
        self.init_streaming_state(channels_input=1)

        # ---------------------------------------------------------
        # The Gradient Stabilizer (init.py)
        # Apply custom weight scaling to prevent exploding gradients 
        # in the convolutional layers before they hit the Mamba blocks.
        # ---------------------------------------------------------
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(module)

    def forward(self, x):
        # ---------------------------------------------------------
        # Macro-Padding (padding.py)
        # Prevent fractional truncation. If the input length isn't a perfect multiple 
        # of the total stride, the U-Net will delete trailing audio samples. 
        # We pass the U-Net's geometric parameters (layers, kernel, stride) to `pad_signal` 
        # so it can calculate the exact number of dummy zeros needed on the right side 
        # to guarantee the audio survives the bottleneck intact.
        # ---------------------------------------------------------
        original_length = x.shape[-1]
        x = pad_signal(x, self.num_encoder_layers, self.kernel_size, self.stride)

        skips = []
        
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            
        skips = skips[::-1] 

        x = self.proj_in(x)
        # Mamba blocks need (Batch, Time, Channels)
        x = x.permute(0, 2, 1) 
        
        for mamba in self.mamba_blocks:
            x = mamba(x)
        
        # Swapping back to (Batch, Channels, Time)
        x = x.permute(0, 2, 1) 
        x = self.proj_out(x)

        for i, block in enumerate(self.decoder):
            # U-net skip logic
            skip = skips[i][..., :x.shape[-1]]
            x = x + skip
            x = block(x)

        # Slice off the mathematical padding before returning the final audio
        return x[..., :original_length]

    """
    The Streaming Step (streaming.py)
    Processes a single, perfectly sized frame of audio during live edge inference.
    We'll recreate this is C++ in our edge SDKs but need this now for testing.
    The Mixin requires this method to be implemented. This defines what happens
    to a SINGLE 10ms frame during live edge inference.
    
    Unlike the stateless `forward()` pass which processes an entire audio file at once, 
    this method relies on the `StreamingInferenceMixin` to maintain acoustic continuity 
    across continuous 10ms microphone buffers. It utilizes three specific state-caching 
    mechanisms to prevent audio boundary artifacts (clicks, pops, and phase smearing).

    Tensor Flow & Architecture:
    1. The Encoder (Overlap-Save): 
        Strided convolutions require context from the past. For each encoder layer, 
        we retrieve the trailing edge of the previous frame from `self.encoder_decoder_state`, 
        concatenate it to the front of the current tensor, and calculate the exact offset 
        needed to align the sliding window. We then save the new trailing edge for the next frame.
        
    2. The Bottleneck (Mamba State Passing):
        The highly compressed latent tensor is passed to the Mamba blocks along with 
        `self.inference_params`. Mamba uses this object to retrieve the historical hidden 
        state (h_t), updates its memory using the new frame, and overwrites the cache 
        so the memory persists into the next callback.
        
    3. The Decoder (Overlap-Add):
        Transposed convolutions "splat" data, meaning the mathematical output of Frame A 
        physically bleeds into the time-domain of Frame B. For each decoder layer, we slice 
        off this excess trailing edge, save it to the cache, and mathematically add the 
        cached edge from the previous frame to the leading edge of the current frame.
        
    Args:
        frame (torch.Tensor): A raw audio tensor of shape (Batch, 1, Time) whose length 
                                perfectly matches `self.frame_length`.
                                
    Returns:
        torch.Tensor: A fully enhanced audio frame ready to be dispatched to the speaker.
    """
    def _enhance_frame_step(self, frame):
        """
        Processes a single valid frame during streaming inference.
        Unlike `forward()`, this method caches the convolutional overlaps
        and Mamba hidden states in `self.encoder_decoder_state` to maintain
        temporal continuity between frames.
        """
        x = frame.unsqueeze(1) if frame.dim() == 2 else frame
        skips = []
        
        # 1. The Encoder (With Overlap Caching)
        total_stride_tracker = self.stride ** self.num_encoder_layers
        for i, encode in enumerate(self.encoder):
            total_stride_tracker //= self.stride
            length = x.shape[-1]
            
            # Retrieve the cached overlap from the previous frame
            prev_state = self.encoder_decoder_state.get(f"enc{i}", None)
            
            if prev_state is not None:
                # Calculate how much of the current frame we need to chop off 
                # because it was already processed in the previous frame's lookahead
                offset = length - self.kernel_size - self.stride * ((length - self.kernel_size) // self.stride - prev_state.shape[-1])
                x = x[..., offset:]

            # Process the block
            x = encode(x)

            if prev_state is not None:
                # Stitch the cached overlap to the front of the new tensor
                x = torch.cat([prev_state, x], dim=-1)

            # Save the trailing edge of this block's output for the NEXT frame
            self.encoder_decoder_state[f"enc{i}"] = x[..., total_stride_tracker:]
            skips.append(x)

        # 2. The Mamba Bottleneck (With State Passing)
        x = self.proj_in(x)
        x = x.permute(0, 2, 1) 
        
        # Pass `self.inference_params` into the Mamba blocks so they can 
        # read the previous state and write the new state
        for mamba in self.mamba_blocks:
            # Note: We will need to update UnifiedMambaBlock to accept inference_params
            x = mamba(x, inference_params=self.inference_params) 
            
        x = x.permute(0, 2, 1) 
        x = self.proj_out(x)

        # 3. The Decoder (With Overlap-Add)
        skips = skips[::-1]
        for i, _block in enumerate(self.decoder):
            # Forcibly tell Pylance: "This generic module is actually a DecoderBlock"
            block = cast(DecoderBlock, _block)
            skip = skips[i][..., :x.shape[-1]]
            x = x + skip
            
            # The decoder block consists of [1x1 Conv, ReLU, TransposedConv, ReLU]
            # We step through them manually to handle the state caching
            x = block.conv1x1(x)
            x = block.relu1(x)
            # Call the raw conv_t module directly to bypass the Adjustable auto-trimmer
            # or else we'll double trim the overlap.
            x = block.trans_conv.conv_t(x)
            
            # Get the previous overlap
            prev_state = self.encoder_decoder_state.get(f"dec{i}", None)

            overlap_len = self.kernel_size - self.stride

            # Safely handle the bias to satisfy Pylance (and prevent edge-case crashes)
            bias_val = 0
            if block.trans_conv.conv_t.bias is not None:
                bias_val = block.trans_conv.conv_t.bias.view(-1, 1)
            
            # Save the current trailing edge for the next frame
            # And subtract bias so we don't count it twice across two frames
            self.encoder_decoder_state[f"dec{i}"] = x[..., -overlap_len:] - bias_val
            
            # Trim the trailing edge off the current frame
            x = x[..., :-overlap_len]
            
            if prev_state is not None:
                # Add the previous frame's overlap to the current frame's leading edge
                x[..., :overlap_len] += prev_state

            if not block.is_last:
                x = F.relu(x)

        return x[0]

    def training_step(self, batch, batch_idx):
        noisy_audio, clean_audio = batch 
        denoised_audio = self(noisy_audio)

        """ Loss Derivation
        L1 loss alone matches the waveform mathematically but sounds "phasy" and unnatural. 
        Multi-Resolution STFT loss analyzes the audio in the frequency domain across multiple 
        window sizes (e.g., 50ms, 120ms, 240ms). This forces the network to accurately reconstruct 
        the spectral harmonics, which aligns with how the human ear actually perceives audio quality.
        """
        # 1. Time-Domain Loss (Waveform structure)
        l1_loss = F.l1_loss(denoised_audio, clean_audio)
        
        # 2. Frequency-Domain Loss (Human perceptual quality)
        # Spectral Convergence loss: Calculates the Frobenius norm (matrix norm) of the difference between the target and predicted magnitude spectrograms, divided by the norm of the target.
        # Log Magnitude loss: This calculates the L1 distance (absolute error) between the logarithm of the spectrograms.
        # The auraloss library computes both losses above and sums them into a single stft_loss return value.
        stft_loss = self.stft_loss(denoised_audio, clean_audio)
        
        loss = l1_loss + stft_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('l1_loss', l1_loss)
        self.log('stft_loss', stft_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        predicted = self(noisy)
        
        # Calculate both losses for true parity with the training step
        l1_val = torch.nn.functional.l1_loss(predicted, clean)
        stft_val = self.stft_loss(predicted, clean)
        
        # Combine them (using the same weighting you use in training_step)
        loss = l1_val + stft_val
        
        # Log the combined metric
        self.log("val_loss", loss, prog_bar=True, sync_dist=True if torch.cuda.is_available() else False)
        return loss

    def configure_optimizers(self):
        """ Learning Rate
        The CleanUMamba authors specify using the ADAM optimizer with a learning rate of 0.0002.
        This is the community standard for State Space Models.
        Because Mamba relies on a continuous mathematical recurrence loop,
        larger learning rates (like 1e-3) often cause the gradients to explode, resulting in NaN losses.
        """
        """ Weight Decay Sorting
        Separate 1D and 2D parameters for weight decay.
        Why: Weight decay smooths convolutional filters (good), but decaying Mamba's 
        1D state parameters (A, B, C, Delta) artificially shrinks its time-constants 
        and destroys its long-term memory capacity.
        """
        decay_params = []
        no_decay_params = []
        
        for _, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # If the parameter is 1D (biases, norm weights, Mamba state vectors)
            if param.ndim <= 1:
                no_decay_params.append(param)
            # If the parameter is 2D or higher (Convolutional filters, Linear weights)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.02},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(optim_groups, lr=2e-4)
