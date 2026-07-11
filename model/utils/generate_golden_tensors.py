import torch
import numpy as np
import struct
import os
from model import EdgeAudioUSE

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
    model = EdgeAudioUSE().eval()
    
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

    def get_hook(name):
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                golden_cache[name] = output[0].clone()
            else:
                golden_cache[name] = output.clone()
        return hook_fn

    # 4. Attach the Probes
    for i, layer in enumerate(model.encoder):
        handles.append(layer.register_forward_hook(get_hook(f"enc_{i}_out")))
        
    handles.append(model.proj_in.register_forward_hook(get_hook("proj_in_out")))
    
    for i, block in enumerate(model.mamba_blocks):
        handles.append(block.register_forward_hook(get_hook(f"mamba_{i}_out")))
        
    handles.append(model.proj_out.register_forward_hook(get_hook("proj_out_out")))
    
    for i, layer in enumerate(model.decoder):
        handles.append(layer.register_forward_hook(get_hook(f"dec_{i}_out")))

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