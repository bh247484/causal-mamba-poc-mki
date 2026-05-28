#include "state.h"
#include "ggml.h"
#include <string>
#include <algorithm>
#include <cstdio>

bool AudioStateCache::init(const EdgeAudioModel& model) {
    // 1. Core architecture calculation
    overlap_len = model.kernel_size - model.stride;
    
    encoder_states.resize(model.num_encoder_layers);
    decoder_states.resize(model.num_encoder_layers);

    printf("[INFO] Initializing Audio State Cache (Overlap: %d)...\n", overlap_len);

    // 2. Pre-allocate Encoder Caches
    for (int i = 0; i < model.num_encoder_layers; i++) {
        std::string weight_name = "encoder." + std::to_string(i) + ".conv.conv.weight";
        
        if (model.tensors.find(weight_name) == model.tensors.end()) {
            printf("[ERROR] Missing expected weight: %s\n", weight_name.c_str());
            return false;
        }

        // GGML ne[2] is the out_channels for Conv1d
        int out_channels = model.tensors.at(weight_name)->ne[2];
        encoder_states[i].cache.resize(out_channels * overlap_len, 0.0f);
    }

    // 3. Pre-allocate Decoder Caches
    for (int i = 0; i < model.num_encoder_layers; i++) {
        std::string weight_name = "decoder." + std::to_string(i) + ".trans_conv.conv_t.weight";
        
        if (model.tensors.find(weight_name) == model.tensors.end()) return false;

        // GGML ne[1] is the out_channels for ConvTranspose1d
        int out_channels = model.tensors.at(weight_name)->ne[1];
        decoder_states[i].cache.resize(out_channels * overlap_len, 0.0f);
    }

    // 4. Pre-allocate Mamba Caches
    for (int i = 0; i < model.num_mamba_blocks; i++) {
        std::string conv_name = "mamba_blocks." + std::to_string(i) + ".block.layers.0.mixer.conv1d.weight";
        std::string a_log_name = "mamba_blocks." + std::to_string(i) + ".block.layers.0.mixer.A_log";
        
        if (model.tensors.find(conv_name) == model.tensors.end() || 
            model.tensors.find(a_log_name) == model.tensors.end()) {
            printf("[ERROR] Missing Mamba weights for block %d\n", i);
            return false;
        }

        int d_conv = model.tensors.at(conv_name)->ne[0];
        int d_inner = model.tensors.at(conv_name)->ne[2];
        int d_state = model.tensors.at(a_log_name)->ne[0];

        MambaState state;
        state.conv_state.resize(d_inner * (d_conv - 1), 0.0f);
        state.ssm_state.resize(d_inner * d_state, 0.0f);
        mamba_states.push_back(state);
    }

    printf("[INFO] State caches allocated for %d Encoder/Decoder layers and %d Mamba blocks.\n", 
           model.num_encoder_layers, model.num_mamba_blocks);
           
    return true;
}

void AudioStateCache::reset() {
    // A fast, zero-allocation memory zeroing method using C++ algorithms.
    for (auto& state : encoder_states) {
        std::fill(state.cache.begin(), state.cache.end(), 0.0f);
    }
    for (auto& state : decoder_states) {
        std::fill(state.cache.begin(), state.cache.end(), 0.0f);
    }
    for (auto& state : mamba_states) {
        std::fill(state.conv_state.begin(), state.conv_state.end(), 0.0f);
        std::fill(state.ssm_state.begin(), state.ssm_state.end(), 0.0f);
    }
}

void AudioStateCache::inspect() const {
    printf("\n========================================================\n");
    printf("             AUDIO STATE CACHE INSPECTION               \n");
    printf("========================================================\n");
    printf("Configuration:\n");
    printf("  Overlap Length    : %d samples\n", overlap_len);
    printf("  Encoder Layers    : %zu\n", encoder_states.size());
    printf("  Decoder Layers    : %zu\n", decoder_states.size());
    printf("  Mamba Blocks      : %zu\n", mamba_states.size());
    printf("--------------------------------------------------------\n");
    printf("%-32s | %-10s | %s\n", "Buffer Name", "Elements", "Size (Bytes)");
    printf("--------------------------------------------------------\n");

    size_t total_elements = 0;

    // 1. Encoder Caches
    for (size_t i = 0; i < encoder_states.size(); i++) {
        size_t elements = encoder_states[i].cache.size();
        total_elements += elements;
        
        char name[64];
        snprintf(name, sizeof(name), "Encoder Cache [%zu]", i);
        printf("%-32s | %-10zu | %-10zu\n", name, elements, elements * sizeof(float));
    }

    // 2. Decoder Caches
    for (size_t i = 0; i < decoder_states.size(); i++) {
        size_t elements = decoder_states[i].cache.size();
        total_elements += elements;
        
        char name[64];
        snprintf(name, sizeof(name), "Decoder Cache [%zu]", i);
        printf("%-32s | %-10zu | %-10zu\n", name, elements, elements * sizeof(float));
    }

    // 3. Mamba Caches
    for (size_t i = 0; i < mamba_states.size(); i++) {
        size_t conv_elements = mamba_states[i].conv_state.size();
        size_t ssm_elements = mamba_states[i].ssm_state.size();
        total_elements += (conv_elements + ssm_elements);

        char conv_name[64];
        snprintf(conv_name, sizeof(conv_name), "Mamba [%zu] Conv1D State", i);
        printf("%-32s | %-10zu | %-10zu\n", conv_name, conv_elements, conv_elements * sizeof(float));

        char ssm_name[64];
        snprintf(ssm_name, sizeof(ssm_name), "Mamba [%zu] SSM State (h_t)", i);
        printf("%-32s | %-10zu | %-10zu\n", ssm_name, ssm_elements, ssm_elements * sizeof(float));
    }

    size_t total_bytes = total_elements * sizeof(float);
    printf("========================================================\n");
    printf("Total Cache Elements: %zu\n", total_elements);
    printf("Total Cache Memory  : %zu bytes (~%.2f MB)\n", 
           total_bytes, total_bytes / (1024.0f * 1024.0f));
    printf("========================================================\n\n");
}