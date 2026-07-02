#include "model.h"
#include "ggml.h"
#include <cstdio>
#include <vector>

int MAGIC_NUM = 0x67676d6c; // "ggml"

void inspect_model(const EdgeAudioModel& model) {
    printf("\n========================================================\n");
    printf("                 GGML MODEL INSPECTION                  \n");
    printf("========================================================\n");
    printf("Architecture (Header Promised):\n");
    printf("  Encoder Layers    : %d\n", model.num_encoder_layers);
    printf("  Mamba Blocks      : %d\n", model.num_mamba_blocks);
    printf("  d_model           : %d\n", model.d_model);
    printf("  kernel_size       : %d\n", model.kernel_size);
    printf("  stride            : %d\n", model.stride);
    printf("  Expected Tensors  : %d\n", model.tensor_count);
    printf("--------------------------------------------------------\n");
    printf("%-45s | %-15s | Elements\n", "Tensor Name", "GGML Shape (ne)");
    printf("--------------------------------------------------------\n");

    size_t actual_loaded_bytes = 0;
    int total_elements = 0;
    int actual_tensor_count = 0;

    // Iterate through our std::map of tensors
    for (const auto& pair : model.tensors) {
        const std::string& name = pair.first;
        struct ggml_tensor* tensor = pair.second;
        actual_tensor_count++;
        int n_dims = ggml_n_dims(tensor);

        // Build a string representing the shape [ne[0], ne[1], ...]
        std::string shape_str = "[";
        for (int i = 0; i < n_dims; i++) {
            shape_str += std::to_string(tensor->ne[i]);
            if (i < n_dims - 1) shape_str += ", ";
        }
        shape_str += "]";

        int elements = ggml_nelements(tensor);
        total_elements += elements;
        actual_loaded_bytes += ggml_nbytes(tensor);

        // 2. Extract the native GGML type
        const char* type_str = "UNKN";
        if (tensor->type == GGML_TYPE_F32) {
            type_str = "FP32";
        } else if (tensor->type == GGML_TYPE_F16) {
            type_str = "FP16";
        }

        printf("%-45s | %-15s | %-6s | %d\n", name.c_str(), shape_str.c_str(), type_str, elements);
    }

printf("========================================================\n");
    
    // DETERMINISTIC ARENA CALCULATION (Expected vs Actual)
    size_t expected_overhead = model.tensor_count * ggml_tensor_overhead();
    size_t expected_padding  = model.tensor_count * GGML_MEM_ALIGN;
    size_t expected_arena    = model.total_float_bytes + expected_overhead + expected_padding;

    size_t actual_overhead   = actual_tensor_count * ggml_tensor_overhead();
    size_t actual_padding    = actual_tensor_count * GGML_MEM_ALIGN;
    size_t actual_arena      = actual_loaded_bytes + actual_overhead + actual_padding;
    
    printf("Total Parameters  : %d\n", total_elements);
    printf("Payload Memory    : %zu bytes (~%.2f MB)\n", 
           actual_loaded_bytes, actual_loaded_bytes / (1024.0f * 1024.0f));
    printf("Struct Overhead   : %zu bytes (Metadata & Padding)\n", 
           actual_overhead + actual_padding);
    printf("Total Arena Size  : %zu bytes (~%.2f MB)\n", 
           actual_arena, actual_arena / (1024.0f * 1024.0f));
           
    // CHECKSUM VERIFICATION
    bool memory_match = (expected_arena == actual_arena);
    bool count_match = (model.tensor_count == actual_tensor_count);

    if (memory_match && count_match) {
        printf("Status            : [PASS] Exact Arena Memory matches header.\n");
    } else {
        printf("Status            : [FAIL] HEADER MISMATCH!\n");
        printf("                    Expected Arena: %zu\n", expected_arena);
        printf("                    Actual Arena  : %zu\n", actual_arena);
    }
    printf("========================================================\n\n");
}

bool load_model(const std::string& bin_path, EdgeAudioModel& model) {
    printf("[INFO] Attempting to load model from: %s\n", bin_path.c_str());
    FILE* bin_file = fopen(bin_path.c_str(), "rb");
    if (!bin_file) {
        printf("[ERROR] Could not open %s\n", bin_path.c_str());
        return false;
    }
    
    // 1. Read and verify Magic Number
    uint32_t magic_num;
    if (fread(&magic_num, sizeof(uint32_t), 1, bin_file) != 1) {
        fclose(bin_file);
        printf("[ERROR] Failed to read magic number from model binary file.\n");
        return false;
    }

    if (magic_num != MAGIC_NUM) {
        printf("[ERROR] Invalid magic number: %x\n", magic_num);
        fclose(bin_file);
        return false;
    }

    // 2. Read Hyperparameters
    fread(&model.num_encoder_layers, sizeof(int), 1, bin_file);
    fread(&model.num_mamba_blocks, sizeof(int), 1, bin_file);
    fread(&model.d_model, sizeof(int), 1, bin_file);
    fread(&model.kernel_size, sizeof(int), 1, bin_file);
    fread(&model.stride, sizeof(int), 1, bin_file);
    fread(&model.total_float_bytes, sizeof(int), 1, bin_file);
    fread(&model.tensor_count, sizeof(int), 1, bin_file);

    printf(
      "[INFO] Config loaded: "
      "num_encoder_layers=%d, "
      "num_mamba_blocks=%d, "
      "d_model=%d, "
      "kernel=%d, "
      "stride=%d, "
      "total_float_bytes=%d, "
      "tensor_count=%d\n",
      model.num_encoder_layers,
      model.num_mamba_blocks,
      model.d_model,
      model.kernel_size,
      model.stride,
      model.total_float_bytes,
      model.tensor_count
    );

           
    // 3. Initialize GGML Memory Arena for Weights
    // - total_float_bytes: Memory required for all floats in the model's tensors.
    // - meta_overhead: How much memory the C-structs take up.
    // - align_padding: The absolute maximum byte-shifting required for CPU alignment.

    size_t meta_overhead = model.tensor_count * ggml_tensor_overhead();
    size_t align_padding = model.tensor_count * GGML_MEM_ALIGN;
    size_t ggml_ctx_size = model.total_float_bytes + meta_overhead + align_padding;
    struct ggml_init_params params = {
      .mem_size   = ggml_ctx_size,
      .mem_buffer = NULL,
      .no_alloc   = false,
    };

    model.ggml_ctx = ggml_init(params);

    if (!model.ggml_ctx) {
      printf("[ERROR] Failed to initialize GGML context.\n");
      fclose(bin_file);
      return false;
    }

    // 4. Read Tensors (The Cassette Tape Loop)
    int tensor_count = 0;
    while (true) {
        int name_len;
        if (fread(&name_len, sizeof(int), 1, bin_file) != 1) {
            break; // EOF reached gracefully
        }

        // Read string name
        std::string name(name_len, '\0');
        fread(&name[0], sizeof(char), name_len, bin_file);

        // Read dimensions
        int n_dims;
        fread(&n_dims, sizeof(int), 1, bin_file);

        int32_t shape[4] = {1, 1, 1, 1}; 
        for (int i = 0; i < n_dims; i++) {
            fread(&shape[i], sizeof(int), 1, bin_file);
        }

        // Read float type (GGML_TYPE_F16 for conv layers, GGML_TYPE_F32 for all others)
        int32_t tensor_dtype;
        fread(&tensor_dtype, sizeof(int), 1, bin_file);
        ggml_type ggml_float_type = (ggml_type)tensor_dtype;

        // 5. Allocate GGML tensors
        // GGML requires at least 1 dimension. PyTorch scalars (0 dims) become size 1.
        // `ne` == "number of elements"
        int ggml_n_dims = (n_dims == 0) ? 1 : n_dims;
        int64_t ne[4] = {1, 1, 1, 1};
        
        if (n_dims == 0) {
            ne[0] = 1; // Scalar
        } else {
            // Reverse PyTorch dimensions for C-contiguous memory
            // e.g., PyTorch [128, 1, 4] -> GGML ne = [4, 1, 128]
            for (int i = 0; i < n_dims; i++) {
                ne[i] = shape[n_dims - 1 - i];
            }
        }

        // Allocate dynamically using the ne array
        struct ggml_tensor* tensor = ggml_new_tensor(model.ggml_ctx, ggml_float_type, ggml_n_dims, ne);

        // Set the tensor name (useful for GGML debugging)
        ggml_set_name(tensor, name.c_str());

        // 6. Read the raw float bytes straight into the GGML tensor memory and calculate byte usage
        size_t elements = ggml_nelements(tensor);
        size_t bytes_per_element = ggml_type_size(ggml_float_type) / ggml_blck_size(ggml_float_type);
        // Use `model_dtype_bytes` here to dynamically account for (potential) changes to the model's float size, i.e. float32 vs float16.
        fread(tensor->data, bytes_per_element, elements, bin_file);

        // Store in our lookup map
        model.tensors[name] = tensor;
        tensor_count++;
    }

    
    fclose(bin_file);
    printf("[INFO] Successfully loaded %d tensors into GGML arena.\n", tensor_count);
    return true;
}

void free_model(EdgeAudioModel& model) {
    if (model.ggml_ctx) {
        ggml_free(model.ggml_ctx);
        model.ggml_ctx = nullptr;
    }
    model.tensors.clear();
}
