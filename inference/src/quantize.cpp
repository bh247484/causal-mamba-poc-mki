#include "ggml.h"
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

int MAGIC_NUM = 0x67676d6c;

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input_fp16.bin> <output_q8_0.bin>\n", argv[0]);
        return 1;
    }

    FILE* f_in = fopen(argv[1], "rb");
    FILE* f_out = fopen(argv[2], "wb");

    if (!f_in || !f_out) {
        printf("[ERROR] Failed to open files.\n");
        return 1;
    }

    // 1. Pass-through the Header
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, f_in);
    fwrite(&magic, sizeof(uint32_t), 1, f_out);

    int hyperparams[7];
    fread(hyperparams, sizeof(int), 7, f_in);
    fwrite(hyperparams, sizeof(int), 7, f_out);

    printf("[INFO] Quantizing model to Q8_0...\n");

    // 2. Process Tensors
    int tensors_processed = 0;
    size_t original_size = 0;
    size_t quantized_size = 0;

    while (true) {
        int name_len;
        if (fread(&name_len, sizeof(int), 1, f_in) != 1) break;

        std::string name(name_len, '\0');
        fread(&name[0], sizeof(char), name_len, f_in);

        int n_dims;
        fread(&n_dims, sizeof(int), 1, f_in);

        int32_t shape[4] = {1, 1, 1, 1};
        fread(shape, sizeof(int), n_dims, f_in);

        int32_t tensor_dtype;
        fread(&tensor_dtype, sizeof(int), 1, f_in);
        ggml_type type = (ggml_type)tensor_dtype;

        // Calculate Elements
        size_t elements = 1;
        for (int i = 0; i < n_dims; i++) elements *= shape[i];

        // In GGML, ne[0] is the last dimension of the PyTorch shape array
        int ne0 = shape[n_dims - 1]; 
        
        // Identify Target Substrings
        // We only want the massive Linear Projections inside the Mamba blocks
        bool is_projection = (name.find("proj.weight") != std::string::npos);
        
        // Verify Hardware Safety
        // Must be a 2D matrix, and the row must be perfectly divisible by 32
        bool is_hardware_safe = (n_dims == 2 && (ne0 % 32 == 0));
        
        // The Decision
        // Determine if we should quantize
        ggml_type target_type = type;
        if (is_projection && is_hardware_safe && (type == GGML_TYPE_F32 || type == GGML_TYPE_F16)) {
            target_type = GGML_TYPE_Q8_0;
        }

        // Write Tensor Metadata to new file
        fwrite(&name_len, sizeof(int), 1, f_out);
        fwrite(name.c_str(), sizeof(char), name_len, f_out);
        fwrite(&n_dims, sizeof(int), 1, f_out);
        fwrite(shape, sizeof(int), n_dims, f_out);
        
        int32_t out_dtype = (int32_t)target_type;
        fwrite(&out_dtype, sizeof(int), 1, f_out);

        // Process Payload
        if (target_type == GGML_TYPE_Q8_0) {
            std::vector<float> f32_data(elements);

            if (type == GGML_TYPE_F16) {
                // Read F16 and upcast to F32
                std::vector<ggml_fp16_t> f16_data(elements);
                fread(f16_data.data(), sizeof(ggml_fp16_t), elements, f_in);
                original_size += elements * sizeof(ggml_fp16_t);

                for (size_t i = 0; i < elements; i++) {
                    f32_data[i] = ggml_fp16_to_fp32(f16_data[i]);
                }
            } else if (type == GGML_TYPE_F32) {
                // Read F32 directly
                fread(f32_data.data(), sizeof(float), elements, f_in);
                original_size += elements * sizeof(float);
            }

            // Compress F32 to Q8_0
            size_t q8_bytes = elements + (elements / 32) * sizeof(ggml_fp16_t); // Q8_0 payload size
            std::vector<char> q8_data(q8_bytes);

            // Modern GGML Public API
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f32_data.data(), q8_data.data(), 0, 1, elements, nullptr);
            
            // Write compressed bytes
            fwrite(q8_data.data(), 1, q8_bytes, f_out);
            quantized_size += q8_bytes;

            printf("  [COMPRESSED] %-40s | %s -> Q8_0\n", name.c_str(), (type == GGML_TYPE_F16 ? "F16" : "F32"));
        } else {
            // Pass-through unquantized data (biases, FP32 layer norms, etc)
            size_t bytes = (ggml_type_size(type) * elements) / ggml_blck_size(type);
            std::vector<char> raw_data(bytes);
            fread(raw_data.data(), 1, bytes, f_in);
            fwrite(raw_data.data(), 1, bytes, f_out);
            
            original_size += bytes;
            quantized_size += bytes;
        }
        tensors_processed++;
    }

    fclose(f_in);
    fclose(f_out);

    printf("\n[SUCCESS] Processed %d tensors.\n", tensors_processed);
    printf("Original Size : %.2f MB\n", original_size / 1024.0 / 1024.0);
    printf("Quantized Size: %.2f MB\n", quantized_size / 1024.0 / 1024.0);

    return 0;
}