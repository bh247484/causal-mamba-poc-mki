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
            # CPU / MPS Fallback
            if inference_params is not None:
                # Use .size() instead of unpacking .shape to satisfy Pylance
                B = x.size(0)
                L = x.size(1)
                D = x.size(2)
                
                # 1. Initialize the custom mambapy cache if it doesn't exist
                if "mambapy_cache" not in inference_params:
                    caches = []
                    
                    # Fetch the config dynamically and cast ints to satisfy the linter
                    mamba_config = getattr(self.block, "config", None)

                    if mamba_config is None:
                        raise RuntimeError("No mamba_config present")
                    
                    ED = int(getattr(mamba_config, "d_inner", D * 2)) 
                    d_conv = int(getattr(mamba_config, "d_conv", 4))
                    
                    layers = getattr(self.block, "layers", [])
                    for _ in layers:
                        # hidden_state handles 'None' natively, but we must pre-allocate the conv buffer
                        inputs = torch.zeros(B, ED, d_conv - 1, device=x.device)
                        caches.append((None, inputs))
                        
                    inference_params["mambapy_cache"] = caches
                
                caches = inference_params["mambapy_cache"]
                out_steps = []
                
                # Fetch the step function dynamically
                step_fn = getattr(self.block, "step")
                
                # 2. Process strictly timestep-by-timestep to persist memory
                for t in range(L):
                    x_t = x[:, t, :]  # Shape: (B, D)
                    x_t, caches = step_fn(x_t, caches)
                    out_steps.append(x_t)
                
                # 3. Save the updated cache for the next 10ms audio chunk
                inference_params["mambapy_cache"] = caches
                
                # Re-stack into standard (B, L, D) shape
                return torch.stack(out_steps, dim=1)
            else:
                # Stateless pass (Training)
                return self.block(x)