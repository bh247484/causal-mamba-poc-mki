#pragma once

#include "model.h"
#include "state.h"

// ============================================================================
// EDGE AUDIO COMPUTATION GRAPH
// ============================================================================
// This file is strictly isolated from the OS audio threads. It does not know 
// about microphones, speakers, or .wav files. It operates purely on raw float 
// arrays and pre-allocated GGML arenas.
// ============================================================================

/**
 * Executes a single forward pass of the EdgeAudioUSE network.
 * @param model         The read-only struct containing the GGML weight arena.
 * @param state         The mutable state caches (Mamba h_t & Conv overlaps).
 * @param input_audio   Pointer to the raw noisy audio floats.
 * @param output_audio  Pointer to the buffer where clean floats will be written.
 * @param num_samples   The length of the frame (default 128).
 * @param memory_buffer The reference to the preallocated memory area.
 * @param memory_size   The size of the preallocated memory arena.
 * * @return true if the graph executed successfully, false if an error occurred.
 */
bool process_audio_frame(
    const EdgeAudioModel& model,
    AudioStateCache& state,
    const float* input_audio,
    float* output_audio,
    uint8_t* memory_buffer,
    size_t memory_size,
    int num_samples = 128
);