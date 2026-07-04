import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import auraloss
from modules.unified_mamba_block import UnifiedMambaBlock
from utils.init import weight_scaling_init
from utils.padding import pad_signal
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
            # No SiLU activation on the absolute final layer (Index 5)
            is_last = (i == len(dec_channels) - 1)
            self.decoder.append(
                DecoderBlock(in_c, out_c, kernel_size, stride, is_last, lookahead)
            )

        # ---------------------------------------------------------
        # Learnable Skip Dampening Parameter:
        # Initialize at 0.1 to starve the skip connections early, forcing 
        # the Mamba blocks to learn the primary noise suppression routing.
        #
        # Initially introduced for as overfitting hack but architecture will also benefit from this at scale.
        # ---------------------------------------------------------
        self.skip_gains = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(len(self.decoder))
        ])

        # ---------------------------------------------------------
        # Streaming Initialization (streaming.py)
        # Initialize the buffers and state tracking variables required 
        # by the StreamingInferenceMixin.
        # ---------------------------------------------------------
        self.init_streaming_state(channels_input=1)
        # A causal U-Net must process a sequence that is a perfect multiple 
        # of its total stride (2^6 = 64). 128 samples = 8ms latency.
        self.frame_length = 128

        # ---------------------------------------------------------
        # The Gradient Stabilizer (init.py)
        # Apply custom weight scaling to prevent exploding gradients 
        # in the convolutional layers before they hit the Mamba blocks.
        # ---------------------------------------------------------
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(module)

    def forward(self, x, mamba_streaming_parity=False):
        original_length = x.shape[-1]
        # Calculate total stride (e.g., 2^6 = 64)
        total_stride = self.stride ** self.num_encoder_layers
        x = pad_signal(x, total_stride)

        skips = []
        
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            
        skips = skips[::-1] 

        x = self.proj_in(x)
        # Mamba blocks need (Batch, Time, Channels)
        x = x.permute(0, 2, 1) 
        
        for mamba in self.mamba_blocks:
            x = mamba(x, mamba_streaming_parity=mamba_streaming_parity)
        
        # Swapping back to (Batch, Channels, Time)
        x = x.permute(0, 2, 1) 
        x = self.proj_out(x)

        for i, block in enumerate(self.decoder):
            # U-net skip logic
            skip = skips[i][..., :x.shape[-1]]
            # Apply the learnable gain for this specific layer
            x = x + (skip * self.skip_gains[i])
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
        we retrieve the trailing edge of the previous frame from `self.encoder_cache`, 
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
        off this excess trailing edge, save it to `self.decoder_cache`, and mathematically add the 
        cached edge from the previous frame to the leading edge of the current frame.
        
    Args:
        frame (torch.Tensor): A raw audio tensor of shape (Batch, 1, Time) whose length 
                                perfectly matches `self.frame_length`.
                                
    Returns:
        torch.Tensor: A fully enhanced audio frame ready to be dispatched to the speaker.
    """
    def _cached_streaming_forward(self, frame):
        """
        Processes a single valid frame (e.g., 10ms) during real-time edge inference.
        Relies on `StreamingInferenceMixin` to inject temporal memory into the graph.
        """
        x = frame.unsqueeze(1) if frame.dim() == 2 else frame
        skips = []
        
        # Calculate the length of the overlap
        # This is also the cache size
        overlap_len = self.kernel_size - self.stride
        
        # ---------------------------------------------------------
        # 1. The Encoder (Overlap-Save / Causal Input Caching)
        # ---------------------------------------------------------
        for i, _block in enumerate(self.encoder):
            # Tell Pylance this is an EncoderBlock
            block = cast(EncoderBlock, _block)

            # Retrieve the cached, trailing edge of the PREVIOUS frame's input
            prev_input = self.encoder_cache.get(f"layer_{i}", None)
            
            if prev_input is not None:
                # Prepend the cached past input to the current input
                x_padded = torch.cat([prev_input, x], dim=-1)
            else:
                # First frame: pad with zeros (silence)
                x_padded = F.pad(x, (overlap_len, 0))
                
            # Cache the trailing edge of the CURRENT input for the NEXT callback frame
            self.encoder_cache[f"layer_{i}"] = x_padded[..., -overlap_len:]
            
            # Call the raw nn.Conv1d (conv.conv) primitive directly to bypass F.pad
            # We want auto padding for training, but not here for inference
            x = block.conv.conv(x_padded)
            x = block.activation(x)
            x = block.conv1x1(x)

            skips.append(x)

        # ---------------------------------------------------------
        # 2. The Mamba Bottleneck (Recurrent State Passing)
        # ---------------------------------------------------------
        x = self.proj_in(x).permute(0, 2, 1) 
        
        # Pass `self.inference_params` into the Mamba blocks so they can 
        # read the previous state and write the new state
        for i, mamba in enumerate(self.mamba_blocks):
            # 1. Create a private memory namespace for this specific layer
            layer_key = f"mamba_layer_{i}"
            if layer_key not in self.inference_params:
                self.inference_params[layer_key] = {}
                
            # 2. Pass ONLY that private dictionary to the block
            x = mamba(x, inference_params=self.inference_params[layer_key])

        x = x.permute(0, 2, 1) 
        x = self.proj_out(x)

        # ---------------------------------------------------------
        # 3. The Decoder (Overlap-Add Splatting)
        # ---------------------------------------------------------
        for i, _block in enumerate(self.decoder):
            # Tell Pylance this is an DecoderBlock
            block = cast(DecoderBlock, _block)
            
            # FILO Stack: Pop the deepest skip connection directly off the list
            skip = skips.pop()[..., -x.shape[-1]:]
            
            # Apply the learned skip dampener parameter
            x = x + (skip * self.skip_gains[i])
            
            x = block.conv1x1(x)
            x = block.silu(x)
            # Call the raw inner nn.ConvTranspose1d (trans_conv.conv_t) block here to bypass padding
            # We want auto padding for training, but not here for inference
            x = block.trans_conv.conv_t(x)
            
            # 3A. State Stitching
            prev_state = self.decoder_cache.get(f"layer_{i}", None)
            if prev_state is not None:
                x[..., :overlap_len] += prev_state
                
            # 3B. Bias Adjustment & State Saving
            bias_val = block.trans_conv.conv_t.bias.view(-1, 1) if block.trans_conv.conv_t.bias is not None else 0
            self.decoder_cache[f"layer_{i}"] = x[..., -overlap_len:] - bias_val
            
            # 3C. Truncate the future bleed from the current frame
            x = x[..., :-overlap_len]
            
            if not block.is_last:
                x = F.silu(x)

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
        
        # @Overfit-Config: This ratio seems to work for overfitting but is probably worth revisiting
        loss = l1_loss + (0.2 * stft_loss)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('l1_loss', l1_loss)
        self.log('stft_loss', stft_loss)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """
        PyTorch Lightning Hook: Fires after loss.backward() but before optimizer.step().
        This is the only safe place to inspect gradients.
        """
        # 1. Collect all gradients from the Mamba blocks
        mamba_grads = [p.grad for p in self.mamba_blocks.parameters() if p.grad is not None]
        
        # 2. Calculate the total L2 norm of these gradients
        if len(mamba_grads) > 0:
            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in mamba_grads]), 2)
            # Log it to TensorBoard
            self.log('mamba_grad_norm', total_norm, prog_bar=False, on_step=True, on_epoch=False)
        else:
            # Fallback if gradients are dropped/None
            self.log('mamba_grad_norm', 0.0, prog_bar=False, on_step=True, on_epoch=False)
    
    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        predicted = self(noisy)
        
        # Calculate both losses for true parity with the training step
        l1_val = torch.nn.functional.l1_loss(predicted, clean)
        stft_val = self.stft_loss(predicted, clean)
        
        # @Overfit-Config: This ratio seems to work for overfitting but is probably worth revisiting
        loss = l1_val + (0.2 * stft_val)
        
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
            # {'params': decay_params, 'weight_decay': 0.02},
            {'params': decay_params, 'weight_decay': 0.0}, # @Overfit-Config: 0.0 for overfitting, restore line above for scale training
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(optim_groups, lr=2e-4)
