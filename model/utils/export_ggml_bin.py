import torch
import struct
from model import EdgeAudioUSE

# GGML float type Enums
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

CHECKPOINT_FILE_PATH = "tb_logs/StageA_Overfit/version_14/checkpoints/use-overfit-epoch=451-train_loss=0.1980.ckpt"

"""
GGML Binary Exporter for EdgeAudioUSE

This script extracts PyTorch weights and packs them into a custom, flat binary file (.bin) 
designed to be read by C++ GGML without any complex metadata parsing (like GGUF).

ARCHITECTURE (The "Cassette Tape" Model):
This file uses Length-Prefixed Encoding. There is no global table of contents or memory 
offsets. C++ must read it strictly sequentially from byte 0 to the end. Every piece of 
data acts as a header for the *immediately subsequent* piece of data.

FILE STRUCTURE:
1. Header Block:
   - [4 bytes] Magic Number (0x67676d6c / "ggml" in hex) -> Handshake for C++
   - [4 bytes] num_encoder_layers (int32)
   - [4 bytes] d_model (int32)
   - [4 bytes] kernel_size (int32)
   - [4 bytes] stride (int32)

2. Tensor Block (Repeats until EOF):
   For every tensor in the model, we write a "Boxcar":
   - [4 bytes] Name Length (N) -> Tells C++ exactly how many characters to read next.
   - [N bytes] String Name -> e.g., "encoder.0.conv.conv.weight"
   - [4 bytes] Number of Dimensions (D) -> Tells C++ how many integers to read for the shape.
   - [D * 4 bytes] Dimension Sizes -> e.g., [128, 1, 4]
   - [X bytes] Raw FP32 Data -> The actual math floats. X = (Product of Dimensions) * 4 bytes.

Because C++ knows exactly how many bytes the floats take up, the moment it finishes reading 
the floats, its "read head" lands perfectly on the 4-byte Name Length of the next tensor.
"""

def main():
    print("--- Stage B: GGML Binary Exporter ---")

    # 1. Load the exact same checkpoint you used in the streaming simulation
    ckpt_path = CHECKPOINT_FILE_PATH
    bin_out_path = "edge_audio_use.bin"
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    
    # Strip the Lightning 'model.' prefix just like before
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("model.", "")
        cleaned_state_dict[clean_key] = value

    # Load into the model to ensure all hyperparameters and shapes are resolved
    model = EdgeAudioUSE()
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    # Collect all conv layers so we can check against them later
    # Conv layers will have f16 precision, all other layers will have f32
    # GGML's conv layers are fundamentally incompatible with f32, they must be f16
    conv_weight_tensors = set()

    # We need to leave out `model.mamba_blocks` but all other conv blocks should be compressed to f16.
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
    
    for name, module in model.named_modules():
        # Check the actual underlying class type
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                # Entire mamba blocks should be f32, even the conv layers inside it
                if "mamba" not in name:
                    # Add the specific tensor memory reference to our set
                    conv_weight_tensors.add(module.weight)

    # Calculate require memory allocation footprint
    total_float_bytes = 0
    tensor_count = 0

    for name, tensor in model.named_parameters():
        if not tensor.requires_grad:
            continue
            
        tensor_count += 1

        # Mixed Precision Logic: Convolutions are FP16 (2 bytes), everything else is FP32 (4 bytes)
        is_conv_weight = tensor in conv_weight_tensors
        
        if is_conv_weight:
            total_float_bytes += tensor.numel() * 2 # FP16
        else:
            total_float_bytes += tensor.numel() * 4 # FP32

    # 2. Open the binary file in 'wb' (Write Binary) mode
    print(f"\nWriting to {bin_out_path}...")
    with open(bin_out_path, "wb") as f:
        
        # ---------------------------------------------------------
        # HEADER BLOCK
        # ---------------------------------------------------------
        # 1. Magic Number: A signature so C++ knows this is the right file format.
        # We'll use the ASCII hex for "ggml" (0x67676d6c).
        f.write(struct.pack("i", 0x67676d6c))
        
        # 2. Architecture Hyperparameters: 
        # C++ needs to know how much memory to allocate before reading the weights.
        # We pack these as standard 32-bit integers ('i').
        f.write(struct.pack("i", model.num_encoder_layers))
        f.write(struct.pack("i", len(model.mamba_blocks))) # num_mamba_blocks
        f.write(struct.pack("i", 512)) # d_model
        f.write(struct.pack("i", model.kernel_size))
        f.write(struct.pack("i", model.stride))
        f.write(struct.pack("i", total_float_bytes))
        f.write(struct.pack("i", tensor_count))
        
        print(
            "Header written: "
            f"num_encoder_layers={model.num_encoder_layers}, "
            f"num_mamba_blocks={len(model.mamba_blocks)}, "
            "d_model=512, "
            f"kernel={model.kernel_size}, "
            f"stride={model.stride}, "
            f"total_float_bytes={total_float_bytes}, "
            f"tensor_count={tensor_count}\n"
        )

        # ---------------------------------------------------------
        # TENSOR BLOCK
        # ---------------------------------------------------------
        for name, tensor in model.named_parameters():
            if not tensor.requires_grad:
                continue
            
            # 1. Apply Mixed Precision Cast
            is_conv_weight = tensor in conv_weight_tensors
            
            # Convert tensor to a flat, contiguous float (FP32 or FP16) NumPy array
            # .detach() removes gradient tracking, .numpy() converts to raw C array
            if is_conv_weight:
                data = tensor.detach().half().cpu().numpy()
                ggml_type = GGML_TYPE_F16
            else:
                data = tensor.detach().cpu().numpy() # Defaults to FP32
                ggml_type = GGML_TYPE_F32
            
            # 2. Extract Shape
            n_dims = len(data.shape)
            shape = data.shape
            name_bytes = name.encode('utf-8')
            
            # --- WRITE THE TENSOR "BOXCAR" ---
            f.write(struct.pack("i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("i", n_dims))
            
            for dim_size in shape:
                f.write(struct.pack("i", dim_size))
                
            # 3. WRITE THE TENSOR DTYPE (NEW!)
            f.write(struct.pack("i", ggml_type))
                
            # 4. Write the raw float data
            # .tobytes() dumps the entire NumPy array into binary in one instant operation
            f.write(data.tobytes())
            
            print(f"Packed: {name.ljust(40)} | Shape: {str(shape).ljust(15)} | Dims: {n_dims} | Type: {ggml_type}")

    print(f"\n✅ SUCCESS: Exported {tensor_count} tensors to {bin_out_path}")

if __name__ == "__main__":
    main()