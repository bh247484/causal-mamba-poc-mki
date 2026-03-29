"""
Macro-Padding Utility for U-Net Audio Architectures

OVERVIEW:
Unlike "Micro-Padding" (which handles left-sided causal padding inside individual 
convolutional layers to prevent future-lookahead), this utility handles "Macro-Padding" 
for the entire audio sequence. 

THE PROBLEM (Fractional Truncation):
U-Net encoders compress audio using strided convolutions. If an architecture has 6 layers 
with a stride of 2, the total temporal compression factor is 2^6 = 64. If you feed an audio 
sequence that is not a perfect multiple of 64, PyTorch's floor division will truncate 
the fractional samples during the downsampling phase. When the decoder upsamples 
the audio back to its original resolution, the output will be missing those trailing 
samples (e.g., put 100 samples in, get 96 out).

THE SOLUTION:
- For Recorded Audio (Offline Inference & Training): `pad_signal` calculates the nearest 
  valid length and appends dummy zeros to the *end* (right side) of the sequence before 
  it enters the network. The zeros are sliced off after the forward pass.
- For Streaming Audio (Live Edge Inference): `get_valid_length(1, ...)` calculates the 
  absolute minimum viable frame size (e.g., 64 samples) that can survive the U-Net bottleneck 
  without fracturing. This dictates exactly how much audio the microphone buffer must collect 
  before triggering a forward pass, guaranteeing zero truncation during live communications.
"""

import torch
import torch.nn.functional as F
import numpy as np

def get_valid_length(length: int, encoder_n_layers: int, kernel_size: int, stride: int) -> int:
    """
    Calculates the nearest valid sequence length to ensure 1D convolutions 
    process perfectly without truncating the output audio.
    """
    for _ in range(encoder_n_layers):
        if length < kernel_size:
            length = 1
        else:
            length = 1 + np.ceil((length - kernel_size) / stride)

    for _ in range(encoder_n_layers):
        length = (length - 1) * stride + kernel_size

    return int(length)

def pad_signal(input_tensor: torch.Tensor, encoder_n_layers: int, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Pads zeros to the right of the signal to match the required valid length.
    Expects input_tensor of shape (Batch, Channels, Time).
    """
    L = get_valid_length(input_tensor.shape[-1], encoder_n_layers, kernel_size, stride)
    padding_needed = L - input_tensor.shape[-1]
    
    if padding_needed > 0:
        return F.pad(input_tensor, (0, padding_needed))
    return input_tensor