#pragma once

#include <string>
#include <map>

// Forward declaration of the GGML context so we don't have to 
// include the entire ggml.h header in every file that touches the model.
struct ggml_context;
struct ggml_tensor;

struct EdgeAudioModel {
    // The memory arena that holds all the weight data
    struct ggml_context* ggml_ctx = nullptr;

    // A dictionary linking the string name (e.g., "encoder.0.conv.conv.weight") 
    // to the actual GGML tensor struct in memory.
    std::map<std::string, struct ggml_tensor*> tensors;

    // Hyperparameters
    int num_encoder_layers = 0;
    int num_mamba_blocks = 0;
    int d_model = 0;
    int kernel_size = 0;
    int stride = 0;

    // Sum memory allocated for tensor floats across the model.
    int total_float_bytes = 0;
    int tensor_count = 0;
    int model_dtype_bytes = 0; // 4 bytes for float32, 2 bytes for float16
};

// Initializes the model and loads weights from the .bin file
bool load_model(const std::string& bin_path, EdgeAudioModel& out_model);

// Frees the memory arena when the program shuts down
void free_model(EdgeAudioModel& model);

// Pretty prints the model architecture as loaded from the bin file and into ggml.
void inspect_model(const EdgeAudioModel& model);