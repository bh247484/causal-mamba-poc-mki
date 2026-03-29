import torch
import torch.nn as nn

def weight_scaling_init(layer: nn.Module):
    """
    Weight rescaling initialization from https://arxiv.org/abs/1911.13254.
    Scales down the initial variance of convolutional filters to prevent 
    exploding gradients when the latent representation enters the Mamba bottleneck.
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    
    layer.weight.data /= torch.sqrt(alpha)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data /= torch.sqrt(alpha)