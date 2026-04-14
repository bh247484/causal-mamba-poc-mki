"""
Structured Channel Pruning Utility for Edge Mamba Architectures

OVERVIEW:
To deploy this model on constrained edge devices (like older Android phones or 
browser wasm), the baseline 40-million parameter model must be shrunk. 
This file contains the utilities to dynamically load a pruned state dictionary 
into a standard, unpruned PyTorch model architecture without causing 'Size Mismatch' crashes.

THE PRUNING HEURISTIC (Group Taylor Importance):
This utility is designed to work with weights pruned via "Structured Channel Pruning".
Unlike "unstructured pruning" (which randomly zeroes out individual weights, creating 
sparse matrices that CPUs actually struggle to process efficiently), structured pruning 
removes entire convolutional channels and linear dimensions.

1. Importance Metric: The pruning algorithm evaluates the importance of each channel 
   using "Group Taylor importance" (a metric that combines the weight magnitude and 
   its gradient).
2. The Hardware Constraint (Multiples of 8): To maintain mathematical compatibility 
   with vectorized CPU instructions and causal convolutions, channels are strictly 
   pruned in multiples of 8.
3. The Result: This strategy achieves an 8X reduction in model size (e.g., shrinking 
   from 41M to 3.2M parameters) without compromising perceptual audio quality.

HOW THIS SCRIPT WORKS:
When you load a state dictionary that has had entire channels deleted, PyTorch will 
normally crash because the instantiated `nn.Conv1d` expects, for example, 512 channels, 
but the incoming tensor only has 256. 

The `load_pruned_state_dict` function intercepts the loading process:
1. It reads the dimensions of the incoming (pruned) weight tensors.
2. It surgically mutates the instantiated PyTorch `nn.Module` objects in memory 
   (updating `module.in_channels`, `module.out_features`, etc.) to match the smaller size.
3. It recalculates the Mamba-specific internal dimension ratios (`d_inner`, `expand`).
4. It safely loads the weights into the newly resized architecture.
"""

import copy
import itertools
import torch.nn as nn

def load_pruned_state_dict(model: nn.Module, pruned_state_dict: dict):
    """
    Dynamically resizes a PyTorch model's channels to match a pruned state dict 
    before loading the weights. Prevents 'Size Mismatch' crashes after 8x channel pruning.
    """
    def prune_model_from_state_dict_shapes(module, local_state_dict, prefix=''):
        # Get persistent buffers and parameters
        persistent_buffers = {k: v for k, v in module._buffers.items() if
                              k not in getattr(module, '_non_persistent_buffers_set', set())}
        local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in local_state_dict:
                # Prune the local state to the shape of the state_dict shape
                param.data = copy.deepcopy(local_state_dict[key].data)

                # Update the in_channels and out_channels of the module
                layers = [nn.LayerNorm, nn.Conv1d, nn.ConvTranspose1d, nn.Linear]
                if any([isinstance(module, layer) for layer in layers]):
                    weight = getattr(module, 'weight', None)
                    if weight is None:
                        continue

                    # Update module variables so PyTorch knows the new size
                    if isinstance(module, nn.LayerNorm):
                        module.normalized_shape = weight.shape
                    elif isinstance(module, nn.Conv1d):
                        module.in_channels = weight.shape[1]
                        module.out_channels = weight.shape[0]
                        if module.groups > 1:
                            module.groups = weight.shape[0]
                    elif isinstance(module, nn.ConvTranspose1d):
                        module.in_channels = weight.shape[0]
                        module.out_channels = weight.shape[1]
                    elif isinstance(module, nn.Linear):
                        module.in_features = weight.shape[1]
                        module.out_features = weight.shape[0]
            else:
                print(f"Warning: Cannot find {key} in pruned state dict for {module}")

        # Recurse through children
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                prune_model_from_state_dict_shapes(child, child_state_dict, child_prefix)

        # Mamba-specific internal dimension updates
        if module.__class__.__name__ in ["Mamba", "UnifiedMambaBlock"]:
            # Extract the projection layers
            dt_proj = getattr(module, 'dt_proj', None)
            x_proj = getattr(module, 'x_proj', None)
            in_proj = getattr(module, 'in_proj', None)
            
            # 2. Type Guard: Prove to Pylance these are explicitly nn.Linear layers
            if isinstance(dt_proj, nn.Linear) and isinstance(x_proj, nn.Linear) and isinstance(in_proj, nn.Linear):
                dt_rank = dt_proj.in_features
                d_state = (x_proj.out_features - dt_rank) // 2
                d_inner = x_proj.in_features
                d_model = in_proj.in_features
                expand = d_inner / d_model

                # Using setattr() here instead of direct assignment keeps Pylance linting happy
                setattr(module, 'dt_rank', dt_rank)
                setattr(module, 'd_state', d_state)
                setattr(module, 'd_inner', d_inner)
                setattr(module, 'd_model', d_model)
                setattr(module, 'expand', expand)

    prune_model_from_state_dict_shapes(model, pruned_state_dict)
    model.load_state_dict(pruned_state_dict, strict=True)