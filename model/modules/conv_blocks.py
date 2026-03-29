import torch.nn as nn
import torch.nn.functional as F

# --- Causal Wrappers (The aTENNuate Constraint) ---

class AdjustableConv1d(nn.Module):
    """
    A 1D Convolutional layer with dynamic temporal padding for real-time audio processing.

    In the context of the EdgeAudioUSE architecture, this layer serves as the primary 
    downsampling and feature-extraction mechanism in the U-Net encoder. By default, 
    standard PyTorch convolutions use centered padding, which requires future audio 
    samples to compute the current frame (breaking causality).

    This module shifts the padding mathematically to control data leakage:
    - Strictly Causal (lookahead=0): Pads only the past. Guarantees zero algorithmic 
      latency for real-time edge inference.
    - Micro-Lookahead (lookahead>0): Sacrifices a specific number of frames (e.g., 2ms) 
      of latency to give the model future context. Used to test the BWE Causal Paradox 
      (whether the model needs to see the end of a transient to reconstruct high frequencies).
    - Non-Causal (lookahead='centered'): Traditional offline processing for theoretical baselines.

    Args:
        in_channels (int): Number of channels in the input audio/latent tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        lookahead (int or str, optional): Temporal leakage allowance. 'centered' for 
            non-causal, or an integer representing the number of future frames to 
            include in the receptive field. Defaults to 0 (strictly causal).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lookahead=0):
        super().__init__()
        self.stride = stride
        self.lookahead = lookahead
        
        # Total padding required to maintain the sequence length ratio
        total_padding = kernel_size - stride
        
        if lookahead == 'centered':
            # Non-Causal: Distribute padding equally (or near-equally)
            self.pad_right = total_padding // 2
            self.pad_left = total_padding - self.pad_right
        else:
            # Causal / Micro-lookahead: Shift padding to the left, allowing N future frames
            self.pad_right = min(lookahead, total_padding)
            self.pad_left = total_padding - self.pad_right
            
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad the temporal dimension (left, right)
        x = F.pad(x, (self.pad_left, self.pad_right))
        return self.conv(x)


class AdjustableConvTranspose1d(nn.Module):
    """
    A 1D Transposed Convolutional layer with dynamic temporal trimming.

    In the EdgeAudioUSE architecture, this layer performs the critical upsampling 
    in the U-Net decoder, projecting the compressed Mamba latent state back into 
    a high-resolution raw audio waveform. 

    Because transposed convolutions inherently "splat" data across a wider temporal 
    window, they produce excess samples that act as non-causal look-ahead. This 
    module dynamically slices the output tensor to trim those excess samples based 
    on the defined lookahead budget. This ensures the reconstructed waveform stays 
    perfectly aligned with the causal constraints of the encoder, preventing audio 
    stuttering or phase smearing during real-time streaming.

    Args:
        in_channels (int): Number of channels in the compressed latent tensor.
        out_channels (int): Number of channels produced by the transposed convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution (dictates upsampling rate). Defaults to 2.
        lookahead (int or str, optional): Temporal leakage allowance. Must match the 
            lookahead used in the encoder to ensure mathematical alignment. 
            Defaults to 0 (strictly causal).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, lookahead=0):
        super().__init__()
        self.stride = stride
        self.conv_t = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        
        # Total excess samples produced that need to be sliced off
        total_trim = kernel_size - stride
        
        if lookahead == 'centered':
            self.trim_right = total_trim // 2
            self.trim_left = total_trim - self.trim_right
        else:
            # Causal / Micro-lookahead: Trim the "future" samples off the left/right
            self.trim_left = min(lookahead, total_trim)
            self.trim_right = total_trim - self.trim_left

    def forward(self, x):
        out = self.conv_t(x)
        
        # Slice the tensor to remove excess temporal samples
        left_idx = self.trim_left
        right_idx = out.shape[-1] - self.trim_right
        return out[..., left_idx:right_idx]