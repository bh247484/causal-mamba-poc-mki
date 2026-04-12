import torch
import torch.nn as nn

# 1. Attempt the import at the module level
try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_OFFICIAL_MAMBA = True

    # Explicitly bind the fallbacks so Pylance knows they exist
    MinimalMamba = None 
    MambaConfig = None
except ImportError:
    from mambapy.mamba import Mamba as MinimalMamba, MambaConfig
    HAS_OFFICIAL_MAMBA = False
    # Explicitly bind the official version so Pylance knows it exists
    OfficialMamba = None

class UnifiedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # 2. Check BOTH the package presence and the hardware presence
        if HAS_OFFICIAL_MAMBA and torch.cuda.is_available() and OfficialMamba is not None:
            self.block = OfficialMamba(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
        elif MambaConfig is not None and MinimalMamba is not None:
            config = MambaConfig(
                d_model=d_model, 
                n_layers=1, 
                d_state=d_state, 
                expand_factor=expand
            )
            self.block = MinimalMamba(config)
        else:
            raise RuntimeError("Failed to initialize Mamba block. Check your mamba dependencies.")


    def forward(self, x, inference_params=None):
        if HAS_OFFICIAL_MAMBA and torch.cuda.is_available():
            # The official CUDA implementation natively accepts the cache object
            return self.block(x, inference_params=inference_params)
        else:
            # MinimalMamba processes the whole sequence statically.
            # (Note: For strictly local streaming tests on a Mac/CPU without Triton, 
            # the Mamba state won't persist perfectly, but it will still validate 
            # the U-Net overlap-add math before cloud GPU training).
            return self.block(x)