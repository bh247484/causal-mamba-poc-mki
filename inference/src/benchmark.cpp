// inference/src/benchmark.cpp
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

// Tell miniaudio to compile its implementation inside this specific binary
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "model.h"
#include "state.h"
#include "graph.h"

// ----------------------------------------------------------------------------
// 1. STATE MANAGEMENT (The Loop Seam Fix)
// ----------------------------------------------------------------------------

void reset_audio_state(AudioStateCache& state) {
    // Wipe the U-Net overlap-add/save caches
    for (auto& enc : state.encoder_states) {
        std::fill(enc.cache.begin(), enc.cache.end(), 0.0f);
    }
    for (auto& dec : state.decoder_states) {
        std::fill(dec.cache.begin(), dec.cache.end(), 0.0f);
    }
    // Wipe the Mamba temporal history
    for (auto& mamba : state.mamba_states) {
        std::fill(mamba.conv_state.begin(), mamba.conv_state.end(), 0.0f);
        std::fill(mamba.ssm_state.begin(), mamba.ssm_state.end(), 0.0f);
    }
}

// ----------------------------------------------------------------------------
// 2. THE AUDIO ENGINE PAYLOAD
// ----------------------------------------------------------------------------

// This struct acts as the bridge between our main thread and the background audio thread
struct AudioEngineData {
    EdgeAudioModel* model;
    AudioStateCache* state;
    
    float* wav_data;         // Pointer to the loaded audio floats
    uint64_t wav_frames;     // Total length of the audio file
    uint64_t playhead;       // Current index in the file
    
    double total_inference_ms;
    uint64_t total_inference_calls;
};

// ----------------------------------------------------------------------------
// 3. THE HARDWARE CALLBACK
// ----------------------------------------------------------------------------

// The sound card fires this function automatically whenever it needs more audio.
// We must fill `pOutput` with strictly bounded (-1.0 to 1.0) floats.
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    AudioEngineData* data = (AudioEngineData*)pDevice->pUserData;
    float* fOut = (float*)pOutput;

    // Our U-Net requires perfectly sized 128-sample frames (8ms)
    int chunk_size = 128; 

    for (ma_uint32 i = 0; i < frameCount; i += chunk_size) {
        
        // --- LOOP SEAM HANDLING ---
        // If the next chunk pushes us past the end of the file...
        if (data->playhead + chunk_size > data->wav_frames) {
            data->playhead = 0;                 // Rewind the file
            reset_audio_state(*(data->state));  // Wipe Mamba's memory to prevent pops!
        }

        const float* chunk_in = &(data->wav_data[data->playhead]);
        float* chunk_out = &fOut[i];

        // --- GOLDEN TENSOR IMPULSE OVERRIDE ---
        // We artificially inject the exact same Impulse Python used.
        std::vector<float> impulse(chunk_size, 0.0f);
        impulse[0] = 1.0f;
        chunk_in = impulse.data(); 
        // --------------------------------------

        // --- BENCHMARK AND INFERENCE ---
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute the GGML graph
        process_audio_frame(*(data->model), *(data->state), chunk_in, chunk_out, chunk_size);

        for (int k = 0; k < chunk_size; k++) {
        if (std::isnan(chunk_out[k]) || std::isinf(chunk_out[k])) {
            printf("[FATAL] NaN detected at sample %d!\n", k);
            break;
        }
}
        
        // Pass through override
        // std::memcpy(chunk_out, chunk_in, chunk_size * sizeof(float));
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate the exact millisecond duration of this specific frame
        std::chrono::duration<double, std::milli> diff = end - start;

        // --- STATE UPDATE ---
        data->total_inference_ms += diff.count();
        data->total_inference_calls++;
        data->playhead += chunk_size;
    }
}

// ----------------------------------------------------------------------------
// 4. MAIN ENTRY POINT
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "      EdgeAudioUSE Real-Time Hardware Testbed           \n";
    std::cout << "========================================================\n";
    
    std::string model_path = "../edge_audio_use.bin";
    std::string input_wav = "../test_loop.wav"; 

    // 1. Initialize the GGML Engine
    EdgeAudioModel model;
    if (!load_model(model_path, model)) {
        std::cerr << "[ERROR] Failed to load model architecture. Exiting.\n";
        return 1;
    }

    AudioStateCache state;
    if (!state.init(model)) {
        std::cerr << "[ERROR] Failed to initialize state caches. Exiting.\n";
        free_model(model);
        return 1;
    }

    // 2. Load the Audio File into Memory
    std::cout << "[INFO] Loading target WAV into memory...\n";
    ma_decoder decoder;
    // We explicitly force the decoder to output 16kHz, mono, floats
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 1, 16000);
    
    if (ma_decoder_init_file(input_wav.c_str(), &decoderConfig, &decoder) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to load input WAV file: " << input_wav << "\n";
        free_model(model);
        return 1;
    }

    ma_uint64 total_frames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to retrieve PCM frame length from WAV file.\n";
        ma_decoder_uninit(&decoder);
        free_model(model);
        return 1;
    }
    std::vector<float> wav_buffer(total_frames);
    
    ma_decoder_read_pcm_frames(&decoder, wav_buffer.data(), total_frames, NULL);
    ma_decoder_uninit(&decoder);
    std::cout << "[INFO] Successfully loaded " << total_frames << " audio frames.\n";

    // 3. Configure the Callback Payload
    AudioEngineData engineData;
    engineData.model = &model;
    engineData.state = &state;
    engineData.wav_data = wav_buffer.data();
    engineData.wav_frames = total_frames;
    engineData.playhead = 0;
    engineData.total_inference_ms = 0.0;
    engineData.total_inference_calls = 0;

    // 4. Configure the Sound Card
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = ma_format_f32; 
    deviceConfig.playback.channels = 1;             
    deviceConfig.sampleRate        = 16000;         
    deviceConfig.dataCallback      = data_callback;
    deviceConfig.pUserData         = &engineData;
    
    // SAFETY LOCK: Force the OS to request audio in blocks that divide cleanly by 128.
    deviceConfig.periodSizeInFrames = 512; 

    ma_device device;
    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to initialize hardware playback device.\n";
        free_model(model);
        return 1;
    }

    // 5. Fire the Engine
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "[ERROR] Failed to start audio device.\n";
        ma_device_uninit(&device);
        free_model(model);
        return 1;
    }

    std::cout << "\n>>> REAL-TIME ENGINE RUNNING. AUDIO ROUTED TO SPEAKERS. <<<\n";
    std::cout << ">>> Press ENTER to terminate loop and print benchmarks... <<<\n";
    getchar(); // Main thread blocks here. The callback thread does all the work.

    // 6. Graceful Teardown & Metrics
    ma_device_uninit(&device);
    
    if (engineData.total_inference_calls > 0) {
        double avg_latency = engineData.total_inference_ms / engineData.total_inference_calls;
        
        printf("\n========================================================\n");
        printf("                  EDGE LATENCY BENCHMARK                  \n");
        printf("========================================================\n");
        printf("Target Frame Budget : 8.000 ms (128 samples @ 16kHz)\n");
        printf("Avg Inference Time  : %.3f ms\n", avg_latency);
        
        if (avg_latency < 8.0) {
            printf("Status              : [PASS] Real-time Viable\n");
        } else {
            printf("Status              : [FAIL] Audio Buffer Underrun\n");
        }
        printf("========================================================\n");
    }
    
    free_model(model);
    return 0;
}