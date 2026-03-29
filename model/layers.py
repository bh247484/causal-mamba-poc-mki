import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.conv_blocks import AdjustableConv1d, AdjustableConvTranspose1d

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, lookahead=0):
        super().__init__()
        self.net = nn.Sequential(
            AdjustableConv1d(in_channels, out_channels, kernel_size, stride, lookahead),
            # Swap ReLU for the custom GLU (SiLU variant) for richer harmonics
            Activation(activation="SiLU", bypass_channels=0),
            # 1x1 Convs don't shift the time axis, so standard Conv1d is fine here
            nn.Conv1d(out_channels, out_channels, kernel_size=1) 
        )
        
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, is_last=False, lookahead=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            AdjustableConvTranspose1d(in_channels, out_channels, kernel_size, stride, lookahead)
        )
        self.is_last = is_last

    def forward(self, x):
        out = self.net(x)
        if not self.is_last:
            out = F.relu(out)
        return out
    
# Lifted from CleanUMamba codebase
class Activation(nn.Module):
    r"""Applies the gated linear unit function, adjusted to use torch primitives
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.
    """

    def __init__(self, activation="Sigmoid", bypass_channels=0) -> None:
        super().__init__()
        assert activation in ['Sigmoid', 'ReLU', "SiLU", "GELU"], f'activation={activation}'

        self.bypass_channels = bypass_channels
        if activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return input[:, :input.shape[1] // 2] * self.sigmoid(input[:, input.shape[1] // 2:])

        nX = self.bypass_channels
        if nX == 0:
            nAB = (input.shape[1]) // 2
            A, B = torch.split(input, [nAB, nAB], 1)  # torch.split is explicitly handled by torch_pruning
            return A * self.activation(B)


        nAB = (input.shape[1] - nX) // 2

        X, A, B = torch.split(input, [nX, nAB, nAB], 1)  # torch.split is explicitly handled by torch_pruning
        assert A.shape == B.shape, f'A.shape={A.shape}, B.shape={B.shape}'

        return torch.cat([X, A * self.activation(B)], 1)