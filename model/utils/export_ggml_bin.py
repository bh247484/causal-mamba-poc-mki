import torch
import struct
from model import EdgeAudioUSE

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
    ckpt_path = "tb_logs/StageA_Overfit/version_12/checkpoints/use-overfit-epoch=496-train_loss=0.2653.ckpt"
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

    # Calculate require memory allocation footprint
    total_float_bytes = 0
    tensor_count = 0
    model_dtype_bytes = 0
    for name, tensor in model.named_parameters():
        if tensor.requires_grad:
            # tensor.element_size() returns exactly 4 for FP32, 2 for FP16, etc.
            total_float_bytes += tensor.numel() * tensor.element_size()
            tensor_count += 1

            # Assign float size if not already calculated
            if model_dtype_bytes == 0:
                model_dtype_bytes = tensor.element_size()

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
        f.write(struct.pack("i", 512)) # d_model
        f.write(struct.pack("i", model.kernel_size))
        f.write(struct.pack("i", model.stride))
        f.write(struct.pack("i", total_float_bytes))
        f.write(struct.pack("i", tensor_count))
        f.write(struct.pack("i", model_dtype_bytes))
        
        print(f"Header written: layers={model.num_encoder_layers}, d_model=512, kernel={model.kernel_size}, stride={model.stride}, total_float_bytes={total_float_bytes}, tensor_count={tensor_count}, model_dtype_bytes={model_dtype_bytes}")

        # ---------------------------------------------------------
        # TENSOR BLOCK
        # ---------------------------------------------------------
        
        for name, tensor in model.named_parameters():
            # Skip gradients or anything not strictly necessary for inference
            if not tensor.requires_grad:
                continue
            
            # 1. Convert tensor to a flat, contiguous float (FP32 or FP16) NumPy array
            # .detach() removes gradient tracking, .numpy() converts to raw C array
            data = tensor.detach().cpu().numpy()
            
            # 2. Extract Shape
            n_dims = len(data.shape)
            shape = data.shape
            
            # Encode the string name to utf-8 bytes
            name_bytes = name.encode('utf-8')
            
            # --- WRITE THE TENSOR "BOXCAR" ---
            
            # Write name length
            f.write(struct.pack("i", len(name_bytes)))
            
            # Write the string bytes themselves
            f.write(name_bytes)
            
            # Write number of dimensions
            f.write(struct.pack("i", n_dims))
            
            # Write each dimension size
            for dim_size in shape:
                f.write(struct.pack("i", dim_size))
                
            # Write the raw float data
            # .tobytes() dumps the entire NumPy array into binary in one instant operation
            f.write(data.tobytes())
            
            print(f"Packed: {name.ljust(40)} | Shape: {str(shape).ljust(15)} | Dims: {n_dims}")

    print(f"\n✅ SUCCESS: Exported {tensor_count} tensors to {bin_out_path}")

if __name__ == "__main__":
    main()