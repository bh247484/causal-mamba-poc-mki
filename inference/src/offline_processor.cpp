#include <iostream>
#include <string>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "model.h"
#include "state.h"
#include "graph.h"

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "       EdgeAudioUSE Offline File Processor              \n";
    std::cout << "========================================================\n";

    std::string model_path = "../edge_audio_use.bin";
    std::string input_wav  = "../test_loop.wav";
    std::string output_wav = "../output_processed.wav";

    // 1. Initialize the GGML Engine
    std::cout << "[INFO] Loading model architecture...\n";
    EdgeAudioModel model;
    if (!load_model(model_path, model)) {
        std::cerr << "[ERROR] Failed to load model. Exiting.\n";
        return 1;
    }

    AudioStateCache state;
    if (!state.init(model)) {
        std::cerr << "[ERROR] Failed to initialize state caches.\n";
        free_model(model);
        return 1;
    }

    // 2. Initialize the Miniaudio Decoder (Read)
    std::cout << "[INFO] Opening input file: " << input_wav << "\n";
    ma_decoder decoder;
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 1, 16000); // 16kHz Mono Float
    if (ma_decoder_init_file(input_wav.c_str(), &decoderConfig, &decoder) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to initialize decoder for: " << input_wav << "\n";
        free_model(model);
        return 1;
    }

    // 3. Initialize the Miniaudio Encoder (Write)
    std::cout << "[INFO] Creating output file: " << output_wav << "\n";
    ma_encoder encoder;
    ma_encoder_config encoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, 16000);
    if (ma_encoder_init_file(output_wav.c_str(), &encoderConfig, &encoder) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to initialize encoder for: " << output_wav << "\n";
        ma_decoder_uninit(&decoder);
        free_model(model);
        return 1;
    }

    // 4. Offline Processing Loop
    const int chunk_size = 128; // Strict structural U-Net frame requirement
    std::vector<float> chunk_in(chunk_size, 0.0f);
    std::vector<float> chunk_out(chunk_size, 0.0f);

    uint64_t total_frames_processed = 0;
    bool processing = true;

    std::vector<uint8_t> compute_arena;
    compute_arena.resize(32 * 1024 * 1024);

    std::cout << "[INFO] Core pipeline processing stream active...\n";
    while (processing) {
        ma_uint64 frames_read = 0;
        ma_result result = ma_decoder_read_pcm_frames(&decoder, chunk_in.data(), chunk_size, &frames_read);

        // If we reach EOF or can't pull a complete block, pad with silence and finalize
        if (result != MA_SUCCESS || frames_read == 0) {
            break;
        }

        if (frames_read < chunk_size) {
            std::fill(chunk_in.begin() + frames_read, chunk_in.end(), 0.0f);
            processing = false; // Terminate after this final block
        }

        // --- MATH EXECUTION ---
        // Process the exact physical arrays offline without hardware timing constraints
        process_audio_frame(
          model,
          state,
          chunk_in.data(),
          chunk_out.data(),
          compute_arena.data(), 
          compute_arena.size(),
          chunk_size
        );

        // Check for catastrophic math issues before saving
        for (int k = 0; k < chunk_size; k++) {
            if (std::isnan(chunk_out[k]) || std::isinf(chunk_out[k])) {
                std::cerr << "[WARNING] Floating point anomaly detected at absolute frame " 
                          << total_frames_processed + k << "!\n";
            }
        }

        // Write processed float block to disk
        ma_encoder_write_pcm_frames(&encoder, chunk_out.data(), chunk_size, NULL);
        total_frames_processed += chunk_size;
    }

    // 5. Clean Teardown
    std::cout << "[SUCCESS] Processed " << total_frames_processed << " acoustic samples.\n";
    ma_encoder_uninit(&encoder);
    ma_decoder_uninit(&decoder);
    free_model(model);

    std::cout << "[INFO] Safe disk sync complete. File closed.\n";
    return 0;
}