import torch
import torch.nn.functional as F

def pad_signal(input_tensor: torch.Tensor, total_stride: int) -> torch.Tensor:
    """
    Macro-Padding Utility.
    Pads zeros to the right of the signal to ensure the length is a perfect 
    multiple of the U-Net's total downsampling stride (e.g., 64). 
    Prevents fractional truncation during strided convolutions.
    """
    length = input_tensor.shape[-1]
    remainder = length % total_stride
    
    if remainder > 0:
        padding_needed = total_stride - remainder
        return F.pad(input_tensor, (0, padding_needed))
        
    return input_tensor