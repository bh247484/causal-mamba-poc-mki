#include "graph.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <vector>

bool process_audio_frame(
    const EdgeAudioModel& model,
    AudioStateCache& state,
    const float* input_audio,
    float* output_audio,
    int num_samples
) {
    // ========================================================================
    // 1. THE COMPUTE ARENA
    // ========================================================================
    // Allocate 16MB for the intermediate math. In a fully optimized v2, 
    // we would allocate this once at startup and reuse it. For v1, allocating
    // a single block per frame is perfectly fast enough for edge real-time.
    // We'll also want to run a test eventually to determine the _actual_ amount of 
    // memory used by the graph and the update our hard coded 16MB amount.
    // 16MB provides reasonable headroom for testing for now though.
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create the empty blueprint graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // We must track specific operations that GGML's backward-walker might miss
    std::vector<struct ggml_tensor*> enforced_deps;
    
    // We save the skip connections for the Decoder (Phase 3)
    std::vector<struct ggml_tensor*> skips;
    
    // We save the padded input tensors so we can crop their tails at the very end
    std::vector<struct ggml_tensor*> encoder_padded_tensors;

    // ========================================================================
    // 2. THE INPUT TENSOR
    // ========================================================================
    // Shape: [Time: 128, Channels: 1]
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_samples, 1);
    
    // Because `no_alloc=false`, the tensor memory is ready immediately.
    // We can directly memcpy the microphone floats into the struct.
    memcpy(x->data, input_audio, num_samples * sizeof(float));

    int current_time_len = num_samples;
    int current_channels = 1;

    // ========================================================================
    // 3. PHASE 1: THE ENCODER (Acoustic to Semantic)
    // ========================================================================
    for (int i = 0; i < model.num_encoder_layers; i++) {
        std::string prefix = "encoder." + std::to_string(i);
        struct ggml_tensor* conv_w = model.tensors.at(prefix + ".conv.conv.weight");
        struct ggml_tensor* conv_b = model.tensors.at(prefix + ".conv.conv.bias");
        struct ggml_tensor* conv1x1_w = model.tensors.at(prefix + ".conv1x1.weight");
        struct ggml_tensor* conv1x1_b = model.tensors.at(prefix + ".conv1x1.bias");

        int out_channels = conv_w->ne[2]; // GGML shape is [kernel, in, out]

        // -------------------------------------------------------------
        // A. The Overlap-Save (Concatenation via Stride Magic)
        // -------------------------------------------------------------
        int padded_time = state.overlap_len + current_time_len;
        
        // Create an empty shell tensor to hold the concatenated result
        struct ggml_tensor* x_padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded_time, current_channels);
        encoder_padded_tensors.push_back(x_padded);

        // Map the C++ std::vector state cache into a GGML tensor
        struct ggml_tensor* cache_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state.overlap_len, current_channels);
        memcpy(cache_tensor->data, state.encoder_states[i].cache.data(), state.encoder_states[i].cache.size() * sizeof(float));

        // Copy the state cache into the FRONT of x_padded
        struct ggml_tensor* view_front = ggml_view_2d(ctx, x_padded, state.overlap_len, current_channels, x_padded->nb[1], 0);
        struct ggml_tensor* cpy_front = ggml_cpy(ctx, cache_tensor, view_front);
        enforced_deps.push_back(cpy_front); // Force graph to execute this

        // Copy the current audio 'x' into the BACK of x_padded
        size_t offset_bytes = state.overlap_len * sizeof(float);
        struct ggml_tensor* view_back = ggml_view_2d(ctx, x_padded, current_time_len, current_channels, x_padded->nb[1], offset_bytes);
        struct ggml_tensor* cpy_back = ggml_cpy(ctx, x, view_back);
        enforced_deps.push_back(cpy_back); // Force graph to execute this

        // -------------------------------------------------------------
        // B. The 1D Convolution (Temporal Compression)
        // -------------------------------------------------------------
        // ggml_conv_1d args: (ctx, weight, input, stride, pad, dilation)
        x = ggml_conv_1d(ctx, conv_w, x_padded, model.stride, 0, 1);

        // -------------------------------------------------------------
        // C. Bias Addition & SwiGLU Activation
        // -------------------------------------------------------------
        // Apply the convolution bias
        // GGML Broadcasting trick: Reshape bias from [C] to [1, C].
        // This forces GGML to broadcast the bias across every Time step.
        struct ggml_tensor* conv_b_reshaped = ggml_reshape_2d(ctx, conv_b, 1, out_channels);
        x = ggml_add(ctx, x, conv_b_reshaped);
        
        // SwiGLU splits the channel axis cleanly in half. 
        // We must also grab the newly compressed timeline!
        int split_channels = out_channels / 2;
        int compressed_time = x->ne[0]; // <--- THE FIX (e.g., 64)

        // Slice the first half (A)
        struct ggml_tensor* A = ggml_view_2d(
          ctx,
          x, 
          compressed_time,
          split_channels, 
          x->nb[1],
          0
        );

        // Slice the second half (B)
        size_t offset_B = split_channels * x->nb[1];
        struct ggml_tensor* B = ggml_view_2d(
          ctx,
          x, 
          compressed_time,
          split_channels, 
          x->nb[1],
          offset_B
        );

        // 5. Compute the gate: SiLU(B)
        struct ggml_tensor* gated_B = ggml_silu(ctx, B);

        // 6. Element-wise multiply: A * SiLU(B)
        x = ggml_mul(ctx, A, gated_B);

        // -------------------------------------------------------------
        // D. The 1x1 Convolution (Channel Expansion/Mixing)
        // -------------------------------------------------------------
        x = ggml_conv_1d(ctx, conv1x1_w, x, 1, 0, 1);
        
        // Use split_channels here because conv1x1 outputs the non-doubled channel count!
        struct ggml_tensor* conv1x1_b_reshaped = ggml_reshape_2d(ctx, conv1x1_b, 1, split_channels);
        x = ggml_add(ctx, x, conv1x1_b_reshaped);

        // -------------------------------------------------------------
        // E. Save the final tensor for Decoder skip connections
        // -------------------------------------------------------------
        skips.push_back(x);

        // Update timeline trackers. The Time shrinks, the Channels grow.
        current_time_len = x->ne[0];
        current_channels = x->ne[1];
    }

    printf("[DEBUG] Phase 1 Complete. Bottleneck shape: [Time: %d, Channels: %d]\n", current_time_len, current_channels);

    // ========================================================================
    // TEMPORARY RETURN SO YOU CAN COMPILE
    // ========================================================================
    ggml_free(ctx);
    return true; 
}