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