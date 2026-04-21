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
        

    def forward(self, x, inference_params=None, mamba_streaming_parity=False):
        # ---------------------------------------------------------
        # 1. Production Path (CUDA)
        # ---------------------------------------------------------
        if HAS_OFFICIAL_MAMBA and torch.cuda.is_available():
            return self.block(x, inference_params=inference_params)

        
        # ---------------------------------------------------------
        # 2. Emulator Path (Local MPS/CPU Proof of Concept)
        # ---------------------------------------------------------
        else:

            # If no cache is provided AND we aren't running a parity test, use mambapy fast parallel scan.
            if inference_params is None and not mamba_streaming_parity:
                return self.block(x)
            
            B, L, D = x.size(0), x.size(1), x.size(2)
            is_stateless = (inference_params is None)
            active_cache = None
            
            # Determine if we have an existing cache to resume from
            if not is_stateless and "mambapy_cache" in inference_params:
                active_cache = inference_params["mambapy_cache"]
            else:
                # Build a fresh cache for either a stateless pass or the first streaming chunk
                active_cache = []
                mamba_config = getattr(self.block, "config", None)
                if mamba_config is None:
                    raise RuntimeError("No mamba_config present")
                    
                ED = int(getattr(mamba_config, "d_inner", D * 2)) 
                d_conv = int(getattr(mamba_config, "d_conv", 4))
                
                for _ in getattr(self.block, "layers", []):
                    inputs = torch.zeros(B, ED, d_conv - 1, device=x.device)
                    active_cache.append((None, inputs))

            # Execute strictly sequentially to guarantee mathematical parity
            out_steps = []
            step_fn = getattr(self.block, "step")
            
            for t in range(L):
                x_t, active_cache = step_fn(x[:, t, :], active_cache)
                out_steps.append(x_t)
                
            # Only persist the cache forward if the streaming mixin requested it
            if not is_stateless:
                inference_params["mambapy_cache"] = active_cache
                
            return torch.stack(out_steps, dim=1)