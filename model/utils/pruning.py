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
            try:
                module.dt_rank = module.dt_proj.in_features
                module.d_state = (module.x_proj.out_features - module.dt_rank) // 2
                module.d_inner = module.x_proj.in_features
                module.d_model = module.in_proj.in_features
                module.expand = module.d_inner / module.d_model
            except AttributeError:
                pass # Fails silently if using the minimal mamba package fallback

    prune_model_from_state_dict_shapes(model, pruned_state_dict)
    model.load_state_dict(pruned_state_dict, strict=True)