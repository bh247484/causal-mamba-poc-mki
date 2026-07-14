// inference/src/main.cpp
#include <iostream>
#include <string>
#include <vector>
#include "model.h"
#include "state.h"
#include "graph.h"

// A clean struct to hold our runtime configuration.
// As we add features (e.g., chunk size, audio devices), they go here.
struct AppConfig {
    std::string model_path = "edge_audio_use.bin";
    std::string input_wav = "";
    bool streaming_mode = false;
};

// Minimal, dependency-free CLI parser
AppConfig parse_cli_args(int argc, char** argv) {
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) config.model_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) config.input_wav = argv[++i];
        } else if (arg == "--stream") {
            config.streaming_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: edge_audio [options]\n"
                      << "  -m, --model <path>   Path to the GGML .bin file\n"
                      << "  -i, --input <path>   Path to input .wav (for offline testing)\n"
                      << "  --stream             Enable live microphone streaming\n";
            exit(0);
        }
    }
    return config;
}

int main(int argc, char** argv) {
    std::cout << "--- EdgeAudioUSE GGML Inference Engine ---\n";
    
    // 1. Parse Arguments
    AppConfig config = parse_cli_args(argc, argv);

    // 2. Initialize Subsystems (The Weight Arena)
    EdgeAudioModel model;
    if (!load_model(config.model_path, model)) {
        std::cerr << "[ERROR] Failed to load model architecture. Exiting.\n";
        return 1;
    }

    // Debug print of model.
    inspect_model(model);

    // Initialize the temporal memory arena
    AudioStateCache state;
    if (!state.init(model)) {
        std::cerr << "[ERROR] Failed to initialize state caches. Exiting.\n";
        free_model(model);
        return 1;
    }

    // Debug print of state cache.
    state.inspect();

    // Placeholder dummy audio buffers to test compilation.
    float input_audio[128] = {0.0f};
    float output_audio[128] = {0.0f};

    std::vector<uint8_t> compute_arena;
    compute_arena.resize(32 * 1024 * 1024);

    process_audio_frame(
        model,
        state,
        input_audio,
        output_audio,
        compute_arena.data(),
        compute_arena.size(),
        128
    );

    // 3. Threading & Execution Lifecycle
    if (config.streaming_mode) {
        std::cout << "[INFO] Entering Live Streaming Mode...\n";
        // TODO: In the future, this is where we spin up the threads:
        // std::thread audio_thread(audio_io_loop, std::ref(ring_buffer));
        // std::thread inference_thread(ggml_worker_loop, std::ref(model), std::ref(ring_buffer));
        // audio_thread.join();
        
    } else if (!config.input_wav.empty()) {
        std::cout << "[INFO] Processing offline file: " << config.input_wav << "\n";
        // TODO: Load .wav, run through GGML sequentially, save output .wav
        
    } else {
        std::cout << "[WARN] No input specified. Run with -h for options.\n";
    }

    // 4. Graceful Teardown
    free_model(model);
    std::cout << "--- Engine Shutdown Gracefully ---\n";
    
    return 0;
}