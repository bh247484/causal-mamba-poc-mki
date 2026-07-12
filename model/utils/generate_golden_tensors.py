import torch
import numpy as np
import struct
import os
from typing import cast
import torch.nn as nn
from model import EdgeAudioUSE

CHECKPOINT_FILE_PATH = "tb_logs/StageA_Overfit/version_19/checkpoints/use-overfit-epoch=494-train_loss=0.1772.ckpt"

def export_golden_tensor(tensor: torch.Tensor, filename: str):
    """
    Saves a PyTorch tensor to a raw binary file for C++ consumption.
    Format: [int32: number of elements] followed by [float32: raw bytes]
    """
    flat_data = tensor.detach().cpu().float().flatten().numpy()
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', len(flat_data)))
        f.write(flat_data.tobytes())

def simulate_ggml_fp16_quantization(model):
    """
    Simulates the exact precision loss of the GGML FP16 export by using 
    the identical targeting heuristic from the export_ggml_bin.py script.
    """
    conv_weight_tensors = set()

    # 1. Target the specific architectural zones
    fp16_zones = [
        model.encoder, 
        model.decoder, 
        model.proj_in, 
        model.proj_out
    ]
    
    for zone in fp16_zones:
        for module in zone.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    conv_weight_tensors.add(module.weight)
    
    # 2. Sweep the whole model as a fallback (excluding Mamba blocks)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                if "mamba" not in name:
                    conv_weight_tensors.add(module.weight)

    # 3. Apply the precision loss in-place
    quantized_count = 0
    with torch.no_grad():
        for param in model.parameters():
            # Since PyTorch nn.Parameter hashes by identity, `in` perfectly matches 
            # the exact memory references we collected in the heuristic above.
            if param in conv_weight_tensors:
                param.data = param.data.half().float()
                quantized_count += 1
                
    print(f"[INFO] Simulated FP16 precision loss on {quantized_count} weight tensors.")

def main():
    export_dir = "golden_tensors"
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Initialize the Model
    model = EdgeAudioUSE()

    print(f"Loading weights from {CHECKPOINT_FILE_PATH}...")
    checkpoint = torch.load(CHECKPOINT_FILE_PATH, map_location=torch.device('cpu'))

    # Strip the Lightning 'model.' prefix
    cleaned_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        clean_key = key.replace("model.", "")
        cleaned_state_dict[clean_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    
    # --- THE FP16 FIX ---
    # Force PyTorch to mimic GGML's destroyed precision bits
    simulate_ggml_fp16_quantization(model)
    # --------------------

    # 2. The Deterministic Impulse
    input_audio = torch.zeros(1, 1, 128)
    input_audio[0, 0, 0] = 1.0  
    
    export_golden_tensor(input_audio, os.path.join(export_dir, "golden_input.bin"))

    # 3. The Bulk Oracle (Automated Hook Registry)
    golden_cache = {}
    handles = []

    def get_output_hook(name):
        """Factory: Captures the mathematical result AFTER a layer computes."""
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                golden_cache[name] = output[0].clone()
            else:
                golden_cache[name] = output.clone()
        return hook_fn

    def get_input_hook(name):
        """Factory: Captures the raw tensor right BEFORE a layer computes."""
        def pre_hook_fn(module, inputs):
            # PyTorch passes inputs as a tuple. The primary tensor is inputs[0].
            golden_cache[name] = inputs[0].clone()
        return pre_hook_fn

    # 4. Attach the Probes
    
    # --- NEW INTERNAL PROBE ---
    # We bypass Pylance panics by explicitly grabbing the submodule by its exact string path.
    # This string perfectly matches the "encoder.0.conv.conv.weight" key in C++.
    raw_step1_module = model.get_submodule("encoder.0.conv.conv")

    # Cast it explicitly to nn.Module to permanently squash the Pylance "Tensor" Union warning
    step1_module = cast(nn.Module, raw_step1_module)

    handles.append(
        step1_module.register_forward_pre_hook(get_input_hook("enc_0_step0_padded_input"))
    )
    
    # Bracket Input B: Save the exact static weights loaded into RAM
    if hasattr(step1_module, "weight") and step1_module.weight is not None:
      # Use cast here too if Pylance squiggles on .clone()
      weight_tensor = cast(torch.Tensor, step1_module.weight)
      golden_cache["enc_0_step0_weights"] = weight_tensor.clone()

    handles.append(step1_module.register_forward_hook(get_output_hook("enc_0_step1_conv_out")))

    for i, layer in enumerate(model.encoder):
        handles.append(layer.register_forward_hook(get_output_hook(f"enc_{i}_out")))
        
    handles.append(model.proj_in.register_forward_hook(get_output_hook("proj_in_out")))
    
    for i, block in enumerate(model.mamba_blocks):
        handles.append(block.register_forward_hook(get_output_hook(f"mamba_{i}_out")))
        
    handles.append(model.proj_out.register_forward_hook(get_output_hook("proj_out_out")))
    
    for i, layer in enumerate(model.decoder):
        handles.append(layer.register_forward_hook(get_output_hook(f"dec_{i}_out")))

    # 5. Execute the Graph
    print(f"\n[INFO] Firing deterministic impulse through the network...")
    with torch.no_grad():
        _ = model(input_audio)
        
    for handle in handles:
        handle.remove()

    # 6. Export the Cache
    print(f"[INFO] Exporting {len(golden_cache)} Golden Tensors to ./{export_dir}/\n")
    for name, tensor in golden_cache.items():
        filename = os.path.join(export_dir, f"golden_{name}.bin")
        export_golden_tensor(tensor, filename)
        
    print("\n[SUCCESS] Phase 1 Complete. Golden map generated.")

if __name__ == "__main__":
    main()