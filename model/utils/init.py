import torch
import torch.nn as nn

def weight_scaling_init(layer: nn.Module):
    """
    Weight rescaling initialization from https://arxiv.org/abs/1911.13254.
    Scales down the initial variance of convolutional filters to prevent 
    exploding gradients when the latent representation enters the Mamba bottleneck.
    """
    weight = getattr(layer, 'weight', None)

    if not isinstance(weight, torch.Tensor):
        raise RuntimeError('Layer has no weight attribute or it is not of type Tensor')
    
    w = weight.detach()
    alpha = 10.0 * w.std()
    
    weight.data /= torch.sqrt(alpha)

    bias = getattr(layer, 'bias', None)
    if isinstance(bias, torch.Tensor):
        bias.data /= torch.sqrt(alpha)