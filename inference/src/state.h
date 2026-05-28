#pragma once

#include <vector>
#include "model.h"

// A single U-Net layer's overlap-save or overlap-add cache.
// Flattened shape: [out_channels * overlap_len]
struct LayerState {
    std::vector<float> cache; 
};

// The state required for a single step of the Mamba associative scan.
struct MambaState {
    // Hidden state for the Causal Conv1D: [d_inner * (d_conv - 1)]
    std::vector<float> conv_state;
    
    // Hidden state for the State Space Model (h_t): [d_inner * d_state]
    std::vector<float> ssm_state; 
};

struct AudioStateCache {
    int overlap_len = 0;
    
    std::vector<LayerState> encoder_states;
    std::vector<LayerState> decoder_states;
    std::vector<MambaState> mamba_states;

    // Called once during startup to allocate all arrays.
    bool init(const EdgeAudioModel& model);

    // Called when a stream ends (or a VoIP call drops) to flush the memory 
    // without releasing it back to the OS.
    void reset();

    // Print the memory layout
    void inspect() const;
};