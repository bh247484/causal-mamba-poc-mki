// inference/src/graph.cpp
#include "graph.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

// Define the plumbing context to keep helper signatures perfectly clean
struct GraphContext {
    struct ggml_context* ctx;
    const EdgeAudioModel& model;
    AudioStateCache& state;
    std::vector<struct ggml_tensor*>& enforced_deps;
    std::vector<struct ggml_tensor*>& skips;
    std::vector<struct ggml_tensor*>& encoder_padded_tensors;
};

// Abstract the SwiGLU logic out of the main loop
struct ggml_tensor* apply_swiglu(struct ggml_context* ctx, struct ggml_tensor* x, int out_channels) {
    int split_channels = out_channels / 2;
    int compressed_time = x->ne[0];

    // Slice the first half (A)
    struct ggml_tensor* A = ggml_view_2d(ctx, x, compressed_time, split_channels, x->nb[1], 0);
    
    // Slice the second half (B)
    size_t offset_B = split_channels * x->nb[1];
    struct ggml_tensor* B = ggml_view_2d(ctx, x, compressed_time, split_channels, x->nb[1], offset_B);

    // Element-wise multiply: A * SiLU(B)
    return ggml_mul(
      ctx,
      A,
      ggml_silu(ctx, B)  // Compute the gate: SiLU(B)
    );
}

// Abstract a single Encoder Layer pass
struct ggml_tensor* execute_encoder_layer(GraphContext& gctx, struct ggml_tensor* x, int layer_idx, int& current_time_len, int& current_channels) {
    // Layers named like "encoder.[i].conv.conv.weight"
    std::string prefix = "encoder." + std::to_string(layer_idx);
    
    struct ggml_tensor* conv_w = gctx.model.tensors.at(prefix + ".conv.conv.weight");
    struct ggml_tensor* conv_b = gctx.model.tensors.at(prefix + ".conv.conv.bias");
    struct ggml_tensor* conv1x1_w = gctx.model.tensors.at(prefix + ".conv1x1.weight");
    struct ggml_tensor* conv1x1_b = gctx.model.tensors.at(prefix + ".conv1x1.bias");

    int out_channels = conv_w->ne[2]; // GGML shape is [kernel, in, out]

    // Overlap-Save State Concatenation
    int padded_time = gctx.state.overlap_len + current_time_len;
    struct ggml_tensor* x_padded = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, padded_time, current_channels);

    // Store 
    gctx.encoder_padded_tensors.push_back(x_padded);

    // Map the C++ std::vector state cache into a GGML tensor
    struct ggml_tensor* cache_tensor = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, gctx.state.overlap_len, current_channels);
    memcpy(cache_tensor->data, gctx.state.encoder_states[layer_idx].cache.data(), gctx.state.encoder_states[layer_idx].cache.size() * sizeof(float));

     // Copy the state cache into the FRONT of x_padded
    struct ggml_tensor* view_front = ggml_view_2d(gctx.ctx, x_padded, gctx.state.overlap_len, current_channels, x_padded->nb[1], 0);
    gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, cache_tensor, view_front)); // Force graph to execute this

    // Copy the current audio 'x' into the BACK of x_padded
    size_t offset_bytes = gctx.state.overlap_len * sizeof(float);
    struct ggml_tensor* view_back = ggml_view_2d(gctx.ctx, x_padded, current_time_len, current_channels, x_padded->nb[1], offset_bytes);
    gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, x, view_back)); // Force graph to execute this

    // Downsampling 1D Convolution
    // ggml_conv_1d args: (ctx, weight, input, stride, pad, dilation)
    x = ggml_conv_1d(gctx.ctx, conv_w, x_padded, gctx.model.stride, 0, 1);
    // Apply the convolution bias
    // GGML Broadcasting trick: Reshape bias from [C] to [1, C].
    // This forces GGML to broadcast the bias across every Time step.
    x = ggml_add(gctx.ctx, x, ggml_reshape_2d(gctx.ctx, conv_b, 1, out_channels));

    // SwiGLU Activation
    x = apply_swiglu(gctx.ctx, x, out_channels);

    // 1x1 Channel Mixing Convolution
    x = ggml_conv_1d(gctx.ctx, conv1x1_w, x, 1, 0, 1);
    x = ggml_add(gctx.ctx, x, ggml_reshape_2d(gctx.ctx, conv1x1_b, 1, out_channels / 2));

    // E. Save skip connection for Phase 3
    gctx.skips.push_back(x);

    // Update tracking dimensions
    current_time_len = x->ne[0];
    current_channels = x->ne[1];
    
    return x;
}

// The clean main entry point
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
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // We must track specific operations that GGML's backward-walker might miss
    std::vector<struct ggml_tensor*> enforced_deps;
    // We save the skip connections for the Decoder (Phase 3)
    std::vector<struct ggml_tensor*> skips;
      // We save the padded input tensors so we can crop their tails at the very end
    std::vector<struct ggml_tensor*> encoder_padded_tensors;

    // Package the references cleanly
    GraphContext gctx = { ctx, model, state, enforced_deps, skips, encoder_padded_tensors };

    // ========================================================================
    // 2. THE INPUT TENSOR
    // ========================================================================
    // Shape: [Time: 128, Channels: 1]
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_samples, 1);
    memcpy(x->data, input_audio, num_samples * sizeof(float));

    int current_time_len = num_samples;
    int current_channels = 1;

    // ========================================================================
    // 3. ENCODER LOOP
    // ========================================================================
    for (int i = 0; i < model.num_encoder_layers; i++) {
        x = execute_encoder_layer(gctx, x, i, current_time_len, current_channels);
    }

    printf("[DEBUG] Phase 1 Complete. Bottleneck shape: [Time: %d, Channels: %d]\n", current_time_len, current_channels);

    // ========================================================================
    // (Future Phase 2 & 3 will drop right here as clean, isolated blocks)
    // ========================================================================

    ggml_free(ctx);
    return true; 
}
