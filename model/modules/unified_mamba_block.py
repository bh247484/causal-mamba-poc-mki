import torch
import torch.nn as nn

# 1. Attempt the import at the module level
try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_OFFICIAL_MAMBA = True
except ImportError:
    HAS_OFFICIAL_MAMBA = False
    from mambapy.mamba import Mamba as MinimalMamba, MambaConfig

class UnifiedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # 2. Check BOTH the package presence and the hardware presence
        if HAS_OFFICIAL_MAMBA and torch.cuda.is_available():
            self.block = OfficialMamba(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
        else:
            config = MambaConfig(
                d_model=d_model, 
                n_layers=1, 
                d_state=d_state, 
                expand_factor=expand
            )
            self.block = MinimalMamba(config)

    def forward(self, x):
        return self.block(x)