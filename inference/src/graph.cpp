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
    std::vector<struct ggml_tensor*>& mamba_padded_tensors;
};

// Safely reshapes a 1D bias tensor to [1, Channels] to broadcast across Time
struct ggml_tensor* apply_bias(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* b) {
    if (b == NULL) return x;
    // b->ne[0] inherently knows its own channel count, making this immune to channel-math bugs
    struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
    return ggml_add(ctx, x, b_reshaped);
}

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
    x = apply_bias(gctx.ctx, x, conv_b);

    // SwiGLU Activation
    x = apply_swiglu(gctx.ctx, x, out_channels);

    // 1x1 Channel Mixing Convolution
    x = ggml_conv_1d(gctx.ctx, conv1x1_w, x, 1, 0, 1);
    x = apply_bias(gctx.ctx, x, conv1x1_b);

    // E. Save skip connection for Phase 3
    gctx.skips.push_back(x);

    // Update tracking dimensions
    current_time_len = x->ne[0];
    current_channels = x->ne[1];
    
    return x;
}

// Applies the depthwise 1D Convolution for Branch 1 of the Mamba Block
struct ggml_tensor* apply_mamba_conv1d(GraphContext& gctx, struct ggml_tensor* branch_1, int layer_idx) {
    std::string prefix = "mamba_blocks." + std::to_string(layer_idx) + ".block.layers.0.mixer.";
    struct ggml_tensor* conv_w = gctx.model.tensors.at(prefix + "conv1d.weight");
    struct ggml_tensor* conv_b = gctx.model.tensors.at(prefix + "conv1d.bias");

    // branch_1 is naturally [Channels, Time]
    int channels = branch_1->ne[0];      // 1024
    int current_time_len = branch_1->ne[1];

    // ========================================================================
    // 1. PREPARE THE WEIGHTS
    // PyTorch exports depthwise weights as [4, 1, 1024].
    // We reshape to [4, 1024], transpose, and force it to be physically contiguous
    // in RAM so it becomes a perfect [1024, 4] matrix.
    // ========================================================================
    struct ggml_tensor* w_2d = ggml_reshape_2d(gctx.ctx, conv_w, 4, channels);
    struct ggml_tensor* w_cont = ggml_cont(gctx.ctx, ggml_transpose(gctx.ctx, w_2d));

    // ========================================================================
    // 2. OVERLAP-SAVE CACHE
    // ========================================================================
    int overlap_len = 3; // kernel=4 causal means 3 ticks of history
    int padded_time = overlap_len + current_time_len;
    
    // Allocate [1024, padded_time]
    struct ggml_tensor* x_padded = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, channels, padded_time);
    gctx.mamba_padded_tensors.push_back(x_padded);

    // Load Cache [1024, 3]
    struct ggml_tensor* cache_tensor = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, channels, overlap_len);
    memcpy(cache_tensor->data, gctx.state.mamba_states[layer_idx].conv_state.data(), channels * overlap_len * sizeof(float));

    // Copy cache to the front (Offset 0 on the Time/Y-axis)
    struct ggml_tensor* view_front = ggml_view_2d(gctx.ctx, x_padded, channels, overlap_len, x_padded->nb[1], 0);
    gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, cache_tensor, view_front));

    // Copy audio to the back (Offset by overlap_len on the Time/Y-axis)
    size_t offset_bytes = overlap_len * x_padded->nb[1];
    struct ggml_tensor* view_back = ggml_view_2d(gctx.ctx, x_padded, channels, current_time_len, x_padded->nb[1], offset_bytes);
    gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, branch_1, view_back));

    // ========================================================================
    // 3. THE DEPTHWISE CONVOLUTION (Manual SIMD Math)
    // y[t] = x[t-3]*w0 + x[t-2]*w1 + x[t-1]*w2 + x[t]*w3
    // ========================================================================
    struct ggml_tensor* y = NULL;

    for (int k = 0; k < 4; k++) {
        // Slice the weight for this kernel index -> [1024, 1]
        size_t w_offset = k * w_cont->nb[1];
        struct ggml_tensor* w_k = ggml_view_2d(gctx.ctx, w_cont, channels, 1, w_cont->nb[1], w_offset);

        // Slice the audio for this time offset -> [1024, current_time_len]
        size_t t_offset = k * x_padded->nb[1];
        struct ggml_tensor* x_k = ggml_view_2d(gctx.ctx, x_padded, channels, current_time_len, x_padded->nb[1], t_offset);

        // Multiply element-wise (GGML broadcasts the [1024, 1] weight across the Time columns automatically)
        struct ggml_tensor* mul = ggml_mul(gctx.ctx, x_k, w_k);

        // Accumulate
        if (y == NULL) {
            y = mul;
        } else {
            y = ggml_add(gctx.ctx, y, mul);
        }
    }

    // 4. Add Bias [1024, 1] (Broadcasts across Time)
    struct ggml_tensor* b_reshaped = ggml_reshape_2d(gctx.ctx, conv_b, channels, 1);
    y = ggml_add(gctx.ctx, y, b_reshaped);

    // 5. SiLU Activation
    y = ggml_silu(gctx.ctx, y);

    return y; 
}

// Abstract a single Mamba Block pass
struct ggml_tensor* execute_mamba_block(GraphContext& gctx, struct ggml_tensor* x, int layer_idx, int current_time_len) {
    std::string prefix = "mamba_blocks." + std::to_string(layer_idx) + ".block.layers.0.";
    
    struct ggml_tensor* norm_w = gctx.model.tensors.at(prefix + "norm.weight");
    struct ggml_tensor* in_proj_w = gctx.model.tensors.at(prefix + "mixer.in_proj.weight");
    // (In a full implementation, you would load all Mamba weights: conv1d, x_proj, dt_proj, A_log, D, out_proj here)

    // 1. RMSNorm (Standard Mamba pre-norm)
    struct ggml_tensor* x_norm = ggml_rms_norm(gctx.ctx, x, 1e-5f);
    struct ggml_tensor* norm_w_reshaped = ggml_reshape_2d(gctx.ctx, norm_w, norm_w->ne[0], 1);
    x_norm = ggml_mul(gctx.ctx, x_norm, norm_w_reshaped);

    // ========================================================================
    // GROUND TRUTH DISCOVERY
    // ========================================================================
    // Let's print exactly what shapes PyTorch exported vs what our audio is.
    // GGML dimensions are [ne[0], ne[1], ne[2], ne[3]] (Innermost to Outermost)
    
    printf("\n[DEBUG] --- MAMBA BLOCK %d ARCHITECTURE ---\n", layer_idx);
    printf("[DEBUG] x_norm (Audio) : [%lld, %lld, %lld, %lld]\n", 
           x_norm->ne[0], x_norm->ne[1], x_norm->ne[2], x_norm->ne[3]);
    printf("[DEBUG] in_proj_w (Wgt): [%lld, %lld, %lld, %lld]\n", 
           in_proj_w->ne[0], in_proj_w->ne[1], in_proj_w->ne[2], in_proj_w->ne[3]);
    printf("[DEBUG] -----------------------------------\n");

    // 2. In Projection
    struct ggml_tensor* x_proj = ggml_mul_mat(gctx.ctx, in_proj_w, x_norm);

    // ========================================================================
    // THE SPLIT (Branch 1 and Branch 2)
    // ========================================================================
    // x_proj is [2048, Time]. We slice it into two [1024, Time] branches.
    int d_inner = x_proj->ne[0] / 2;

    // Branch 1 (The Data) - Bottom 1024 channels
    struct ggml_tensor* branch_1 = ggml_view_2d(gctx.ctx, x_proj, d_inner, current_time_len, x_proj->nb[1], 0);
    
    // Branch 2 (The Gate) - Top 1024 channels (Offset down the X-axis of channels)
    // We step forward by d_inner * nb[0] (which is just d_inner * sizeof(float))
    size_t branch_2_offset = d_inner * x_proj->nb[0];
    struct ggml_tensor* branch_2 = ggml_view_2d(gctx.ctx, x_proj, d_inner, current_time_len, x_proj->nb[1], branch_2_offset);

    // Pass Branch 1 through the Temporal Convolution
    struct ggml_tensor* x_in = apply_mamba_conv1d(gctx, branch_1, layer_idx);

    // ========================================================================
    // THE TEMPORAL RNN LOOP (With Pre-allocated Reassembly)
    // ========================================================================
    // 1. Pre-allocate the massive contiguous destination tensor
    struct ggml_tensor* out_assembled = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, x_in->ne[0], current_time_len);
    
    // We must manually step through Time (t=0, then t=1) because Mamba is causal.
    for (int t = 0; t < current_time_len; t++) {
        // A. Slice out a single tick of input time
        size_t in_offset = t * x_in->nb[1];
        struct ggml_tensor* x_t = ggml_view_2d(gctx.ctx, x_in, x_in->ne[0], 1, x_in->nb[1], in_offset);

        // ---------------------------------------------------------
        // B. (Placeholder for the hardcore Mamba Math)
        // GGML will spawn a new, disconnected memory node here for the output.
        // ---------------------------------------------------------
        struct ggml_tensor* y_t = x_t; 

        // C. Create a stencil hovering over the correct slot in our pre-allocated tensor
        size_t out_offset = t * out_assembled->nb[1];
        struct ggml_tensor* slot_view = ggml_view_2d(gctx.ctx, out_assembled, out_assembled->ne[0], 1, out_assembled->nb[1], out_offset);

        // D. Physically copy the math result from its island into the contiguous slot
        gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, y_t, slot_view));
    }

    // ========================================================================
    // GATING & OUT PROJECTION & RESIDUAL SKIP CONNECTION
    // ========================================================================
    // A. Activate the Gate (Branch 2)
    struct ggml_tensor* gate = ggml_silu(gctx.ctx, branch_2);

    // B. Element-wise multiply the Memory (Branch 1) by the Gate (Branch 2)
    // Both are perfectly aligned as [1024, Time]
    struct ggml_tensor* gated_output = ggml_mul(gctx.ctx, out_assembled, gate);

    // C. Output Projection: Project [1024, Time] back down to [512, Time]
    struct ggml_tensor* out_proj_w = gctx.model.tensors.at(prefix + "mixer.out_proj.weight");
    struct ggml_tensor* x_out = ggml_mul_mat(gctx.ctx, out_proj_w, gated_output);

    // D. Standard Deep Learning Residual Skip Connection 
    // Add the original 512-channel input of this block (x) to the 512-channel output (x_out)
    x_out = ggml_add(gctx.ctx, x_out, x);
    
    return x_out;
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
    std::vector<struct ggml_tensor*> mamba_padded_tensors;

    // Package the references cleanly
    GraphContext gctx = {
        ctx,
        model,
        state,
        enforced_deps,
        skips,
        encoder_padded_tensors,
        mamba_padded_tensors
    };

    // ========================================================================
    // 2. THE INPUT TENSOR
    // ========================================================================
    // Shape: [Time: 128, Channels: 1]
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_samples, 1);
    memcpy(x->data, input_audio, num_samples * sizeof(float));

    int current_time_len = num_samples;
    int current_channels = 1;

    // ========================================================================
    // 3. ENCODER LOOP (Acoustic to Semantic Representation)
    // ========================================================================
    for (int i = 0; i < model.num_encoder_layers; i++) {
        x = execute_encoder_layer(gctx, x, i, current_time_len, current_channels);
    }

    printf("[DEBUG] Phase 1 Complete. Bottleneck shape: [Time: %d, Channels: %d]\n", current_time_len, current_channels);

    // ========================================================================
    // 4. THE MAMBA BOTTLENECK (Semantic Reasoning)
    // ========================================================================
    // A. Global Input Projection: [Time: 2, Channels: 768] -> [Time: 2, Channels: 512]
    struct ggml_tensor* proj_in_w = model.tensors.at("proj_in.weight");
    struct ggml_tensor* proj_in_b = model.tensors.at("proj_in.bias");
    
    x = ggml_conv_1d(ctx, proj_in_w, x, 1, 0, 1);
    x = apply_bias(ctx, x, proj_in_b); // apply_bias works perfectly here because x is [Time, Channels]
    current_channels = proj_in_w->ne[2]; // Now 512!

    // B. Transpose for Mamba (The PyTorch .permute(0, 2, 1) equivalent)
    // We physically flip the axis so Channels sit on ne[0].
    x = ggml_transpose(ctx, x);
    x = ggml_cont(ctx, x);

    // C. The Sequential Mamba Blocks
    for (int i = 0; i < model.num_mamba_blocks; i++) {
        x = execute_mamba_block(gctx, x, i, current_time_len);
    }

    // D. Transpose back to [Time, Channels] for the Decoder
    x = ggml_transpose(ctx, x);
    x = ggml_cont(ctx, x); // Force contiguous memory after transposition

    // E. Global Output Projection: [Time: 2, Channels: 512] -> [Time: 2, Channels: 768]
    struct ggml_tensor* proj_out_w = model.tensors.at("proj_out.weight");
    struct ggml_tensor* proj_out_b = model.tensors.at("proj_out.bias");
    
    x = ggml_conv_1d(ctx, proj_out_w, x, 1, 0, 1);
    x = apply_bias(ctx, x, proj_out_b);
    current_channels = proj_out_w->ne[2]; // Back to 768!

    printf("[DEBUG] Phase 2 Complete. Shape: [Time: %d, Channels: %d]\n", current_time_len, current_channels);

    // ========================================================================
    // (Future Phase 3 - DECODER)
    // ========================================================================

    ggml_free(ctx);
    return true; 
}
