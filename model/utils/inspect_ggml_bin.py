import struct
import math

def main():
    print("--- Binary File Inspector ---")
    
    with open("edge_audio_use.bin", "rb") as f:
        # Read Magic Number (4 bytes -> integer)
        magic_bytes = f.read(4)
        if not magic_bytes:
            print("Error: File is empty.")
            return
            
        magic = struct.unpack("i", magic_bytes)[0]
        print(f"Magic Number: {hex(magic)} (Expected: 0x67676d6c)")
        
        # Read Hyperparameters
        num_encoder_layers = struct.unpack("i", f.read(4))[0]
        num_mamba_blocks = struct.unpack("i", f.read(4))[0]
        d_model = struct.unpack("i", f.read(4))[0]
        kernel = struct.unpack("i", f.read(4))[0]
        stride = struct.unpack("i", f.read(4))[0]
        total_float_bytes = struct.unpack("i", f.read(4))[0]
        tensor_count = struct.unpack("i", f.read(4))[0]
        model_dtype_bytes = struct.unpack("i", f.read(4))[0]
        print(
            "Config: "
            f"num_encoder_layers={num_encoder_layers}, "
            f"num_mamba_blocks={num_mamba_blocks}, "
            f"d_model={d_model}, "
            f"kernel={kernel}, "
            f"stride={stride}, "
            f"total_float_bytes={total_float_bytes}, "
            f"tensor_count={tensor_count}, "
            f"model_dtype_bytes={model_dtype_bytes}\n"
          )
        
        print(f"{'Tensor Name'.ljust(40)} | {'Shape'.ljust(15)} | Elements")
        print("-" * 75)

        while True:
            # 1. Try to read the length of the next name
            name_len_bytes = f.read(4)
            
            # If we don't get 4 bytes, we've successfully reached the end of the file
            if not name_len_bytes or len(name_len_bytes) < 4:
                break 
            
            # 2. Read String Name
            name_len = struct.unpack("i", name_len_bytes)[0]
            name = f.read(name_len).decode('utf-8')
            
            # 3. Read Dimensions
            n_dims = struct.unpack("i", f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack("i", f.read(4))[0])
            
            # 4. Calculate total floats in this tensor
            # (Scalars like skip_gains have 0 dimensions, meaning exactly 1 element)
            if n_dims == 0:
                num_elements = 1 
            else:
                num_elements = math.prod(shape)
            
            print(f"{name.ljust(40)} | {str(shape).ljust(15)} | {num_elements}")
            
            # 5. Skip the float bytes to advance the "read head" to the next tensor
            f.read(num_elements * model_dtype_bytes)
            
        print("-" * 75)
        print(f"✅ Successfully traversed to EOF. Counted {tensor_count} tensors.")

if __name__ == "__main__":
    main()