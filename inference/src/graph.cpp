// inference/src/graph.cpp
#include "graph.h"
#include "ggml.h"
#include "ggml-cpu.h"
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
    std::vector<struct ggml_tensor*>& mamba_final_ssm_states;
    std::vector<struct ggml_tensor*>& decoder_tail_tensors;
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

// Abstract a single Decoder Layer pass
struct ggml_tensor* execute_decoder_layer(GraphContext& gctx, struct ggml_tensor* x, int layer_idx, int& current_time_len, int& current_channels) {
    std::string prefix = "decoder." + std::to_string(layer_idx);

    // Load Weights
    struct ggml_tensor* conv1x1_w = gctx.model.tensors.at(prefix + ".conv1x1.weight");
    struct ggml_tensor* conv1x1_b = gctx.model.tensors.at(prefix + ".conv1x1.bias");
    struct ggml_tensor* trans_w   = gctx.model.tensors.at(prefix + ".trans_conv.conv_t.weight");
    struct ggml_tensor* trans_b   = gctx.model.tensors.at(prefix + ".trans_conv.conv_t.bias");

    // ========================================================================
    // 1. U-NET SKIP CONNECTION (Phase Restoration)
    // ========================================================================
    // Decoder 0 maps to Encoder 5. Decoder 1 maps to Encoder 4, etc.
    int skip_idx = gctx.model.num_encoder_layers - 1 - layer_idx;
    
    // Retrieve the raw skip tensor we saved during the Encoder pass
    struct ggml_tensor* skip_t = gctx.skips[skip_idx];
    
    // Retrieve the learned scalar gain [1] and broadcast it across the skip tensor
    struct ggml_tensor* skip_gain = gctx.model.tensors.at("skip_gains." + std::to_string(skip_idx));
    struct ggml_tensor* gain_reshaped = ggml_reshape_2d(gctx.ctx, skip_gain, 1, 1);
    struct ggml_tensor* gain_repeated = ggml_repeat(gctx.ctx, gain_reshaped, skip_t);
    
    struct ggml_tensor* scaled_skip = ggml_mul(gctx.ctx, skip_t, gain_repeated);

    // Add the high-resolution phase data back into our semantic audio path
    x = ggml_add(gctx.ctx, x, scaled_skip);

    // ========================================================================
    // 2. CHANNEL MIXING 
    // ========================================================================
    x = ggml_conv_1d(gctx.ctx, conv1x1_w, x, 1, 0, 1);
    x = apply_bias(gctx.ctx, x, conv1x1_b);
    x = ggml_silu(gctx.ctx, x);

    // ========================================================================
    // 3. UPSAMPLING (Transposed Convolution)
    // ========================================================================
    int stride = gctx.model.stride; // 2
    
    // ggml_conv_transpose_1d creates an output of length: (Time - 1) * stride + kernel
    x = ggml_conv_transpose_1d(gctx.ctx, trans_w, x, stride, 0, 1);
    x = apply_bias(gctx.ctx, x, trans_b);

    // ========================================================================
    // 4. THE "IS_LAST" CHECK (Preserve the negative audio waveform)
    // ========================================================================
    // If this is NOT the final layer of the U-Net, apply activation.
    bool is_last = (layer_idx == gctx.model.num_encoder_layers - 1);
    if (!is_last) {
        x = ggml_silu(gctx.ctx, x);
    }

    // ========================================================================
    // 5. OVERLAP-ADD (OLA) PHYSICS
    // ========================================================================
    int target_time = current_time_len * stride;
    int channels = x->ne[1];
    int kernel_size = trans_w->ne[0];
    int overlap_len = kernel_size - stride; // 4 - 2 = 2 samples of overlap

    // Pre-allocate the final continuous warehouse for this layer: [target_time, channels]
    struct ggml_tensor* out_assembled = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, target_time, channels);

    // A. THE HEAD (Overlaps with the previous frame's tail)
    // Slice the first 2 ticks of Time (Time is on the X-axis / ne[0])
    struct ggml_tensor* x_head = ggml_view_2d(gctx.ctx, x, overlap_len, channels, x->nb[1], 0);

    // Load the tail from the previous audio frame from our AudioStateCache
    struct ggml_tensor* cache_tensor = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, overlap_len, channels);
    memcpy(cache_tensor->data, gctx.state.decoder_states[layer_idx].cache.data(), overlap_len * channels * sizeof(float));

    // Mathematically ADD them together and drop into the front of the warehouse
    struct ggml_tensor* head_added = ggml_add(gctx.ctx, x_head, cache_tensor);
    struct ggml_tensor* view_out_head = ggml_view_2d(gctx.ctx, out_assembled, overlap_len, channels, out_assembled->nb[1], 0);
    gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, head_added, view_out_head));

    // B. THE BODY (Passes straight through)
    int body_len = target_time - overlap_len;
    size_t body_offset = overlap_len * x->nb[0]; // Byte offset down the X-axis
    struct ggml_tensor* x_body = ggml_view_2d(gctx.ctx, x, body_len, channels, x->nb[1], body_offset);

    size_t out_body_offset = overlap_len * out_assembled->nb[0];
    struct ggml_tensor* view_out_body = ggml_view_2d(gctx.ctx, out_assembled, body_len, channels, out_assembled->nb[1], out_body_offset);
    gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, x_body, view_out_body));

    // C. THE TAIL (Save for the NEXT audio frame)
    size_t tail_offset = target_time * x->nb[0];
    struct ggml_tensor* x_tail = ggml_view_2d(gctx.ctx, x, overlap_len, channels, x->nb[1], tail_offset);

    // Push to our tracking vector so Phase 5 can extract it to the C++ cache
    gctx.decoder_tail_tensors.push_back(x_tail);

    // Update tracking dimensions for the next loop
    current_time_len = target_time;
    current_channels = channels;

    return out_assembled;
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

// Executes a single millisecond (tick) of the State Space Model recurrence.
// Returns the output y_t, and passes back the updated hidden state h_t.
struct ggml_tensor* step_ssm_recurrence(
    GraphContext& gctx, 
    struct ggml_tensor* x_t, 
    struct ggml_tensor* prev_h, 
    std::string prefix,
    struct ggml_tensor** next_h // Pointer so we can return the updated state
) {
    struct ggml_context* ctx = gctx.ctx;
    
    // Load static weights
    struct ggml_tensor* x_proj_w  = gctx.model.tensors.at(prefix + "mixer.x_proj.weight");
    struct ggml_tensor* dt_proj_w = gctx.model.tensors.at(prefix + "mixer.dt_proj.weight");
    struct ggml_tensor* dt_proj_b = gctx.model.tensors.at(prefix + "mixer.dt_proj.bias");
    struct ggml_tensor* A_log     = gctx.model.tensors.at(prefix + "mixer.A_log");
    struct ggml_tensor* D         = gctx.model.tensors.at(prefix + "mixer.D");

    int d_inner = x_t->ne[0];    // 1024
    int d_state = A_log->ne[0];  // 16
    int dt_rank = dt_proj_w->ne[0]; // 32

    // ========================================================================
    // 1. THE DYNAMIC PROJECTIONS (Generate Delta, B, and C)
    // ========================================================================
    // x_proj_w is [64, 1024]. x_t is [1024, 1]. Output is [64, 1].
    struct ggml_tensor* x_proj_out = ggml_mul_mat(ctx, x_proj_w, x_t);

    // Slice the [64, 1] output into our three dynamic parameters using 1D byte offsets
    struct ggml_tensor* dt_raw = ggml_view_1d(ctx, x_proj_out, dt_rank, 0);
    struct ggml_tensor* B      = ggml_view_1d(ctx, x_proj_out, d_state, dt_rank * sizeof(float));
    struct ggml_tensor* C      = ggml_view_1d(ctx, x_proj_out, d_state, (dt_rank + d_state) * sizeof(float));

    // Project dt_raw [32, 1] -> dt [1024, 1], add bias, and apply Softplus
    struct ggml_tensor* dt = ggml_mul_mat(ctx, dt_proj_w, dt_raw);
    dt = ggml_add(ctx, dt, dt_proj_b);
    struct ggml_tensor* dt_soft = ggml_softplus(ctx, dt);

    // ========================================================================
    // 2. DISCRETIZE THE PHYSICS (Calculate A_bar and B_bar)
    // ========================================================================
    // A_log is parameterized as log(A). The true A is -exp(A_log).
    struct ggml_tensor* A_true = ggml_neg(ctx, ggml_exp(ctx, A_log)); // Shape: [16, 1024]

    // To multiply dt [1024, 1] against A [16, 1024], we must reshape dt to [1, 1024] 
    // and physically repeat (broadcast) it 16 times across the X-axis.
    struct ggml_tensor* dt_reshaped = ggml_reshape_2d(ctx, dt_soft, 1, d_inner);
    struct ggml_tensor* dt_repeated = ggml_repeat(ctx, dt_reshaped, A_true);
    
    // A_bar = exp(dt * A)
    struct ggml_tensor* dt_A = ggml_mul(ctx, dt_repeated, A_true);
    struct ggml_tensor* A_bar = ggml_exp(ctx, dt_A);

    // Calculate B_bar_x = (dt * x_t) * B. (We combine x_t into B_bar for efficiency).
    // dx = dt * x_t -> Shape [1024, 1]
    struct ggml_tensor* dx = ggml_mul(ctx, dt_soft, x_t);
    struct ggml_tensor* dx_reshaped = ggml_reshape_2d(ctx, dx, 1, d_inner);
    struct ggml_tensor* dx_repeated = ggml_repeat(ctx, dx_reshaped, A_true); // Broadcast to [16, 1024]

    // B is [16, 1]. We reshape to [16, 1] (just to be safe) and repeat to [16, 1024]
    struct ggml_tensor* B_reshaped = ggml_reshape_2d(ctx, B, d_state, 1);
    struct ggml_tensor* B_repeated = ggml_repeat(ctx, B_reshaped, A_true);

    // B_bar_x = B * (dt * x_t)
    struct ggml_tensor* B_bar_x = ggml_mul(ctx, B_repeated, dx_repeated);

    // ========================================================================
    // 3. THE RECURRENT MEMORY UPDATE
    // ========================================================================
    // h_t = (A_bar * prev_h) + B_bar_x
    struct ggml_tensor* h_t = ggml_add(ctx, 
                                       ggml_mul(ctx, A_bar, prev_h), 
                                       B_bar_x);
    
    // Save a pointer to the new memory so the main loop can step forward
    *next_h = h_t;

    // ========================================================================
    // 4. THE OUTPUT COMPRESSION
    // ========================================================================
    // y_t = h_t * C
    struct ggml_tensor* C_reshaped = ggml_reshape_2d(ctx, C, d_state, 1);
    struct ggml_tensor* C_repeated = ggml_repeat(ctx, C_reshaped, A_true); // [16, 1024]

    struct ggml_tensor* h_C = ggml_mul(ctx, h_t, C_repeated);

    // h_C is [16, 1024]. We must sum across the 16 state dimensions to compress 
    // it back down to our 1024 channels. 
    struct ggml_tensor* y_t = ggml_sum_rows(ctx, h_C); // Output is natively [1, 1024]

    // Reshape back to [1024, 1] to match the input x_t geometry
    y_t = ggml_reshape_2d(ctx, y_t, d_inner, 1);

    // Add the static Skip Connection: y_t = y_t + (x_t * D)
    struct ggml_tensor* D_reshaped = ggml_reshape_2d(ctx, D, d_inner, 1);
    y_t = ggml_add(ctx, y_t, ggml_mul(ctx, x_t, D_reshaped));

    return y_t;
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
    
    // 2. Load the initial hidden state (h_{t-1}) from our AudioStateCache
    struct ggml_tensor* A_log = gctx.model.tensors.at(prefix + "mixer.A_log");
    // Shape: [d_state, channels] -> [16, 1024]
    int channels = x_in->ne[0];
    int d_state = A_log->ne[0];; 
    struct ggml_tensor* current_h = ggml_new_tensor_2d(gctx.ctx, GGML_TYPE_F32, d_state, channels);
    memcpy(current_h->data, gctx.state.mamba_states[layer_idx].ssm_state.data(), d_state * channels * sizeof(float));

    // 3. Step through Time sequentially
    for (int t = 0; t < current_time_len; t++) {
        // A. Slice out a single tick of input time -> [1024, 1]
        size_t in_offset = t * x_in->nb[1];
        struct ggml_tensor* x_t = ggml_view_2d(gctx.ctx, x_in, channels, 1, x_in->nb[1], in_offset);

        // B. Step the State Space Model
        struct ggml_tensor* next_h = NULL;
        struct ggml_tensor* y_t = step_ssm_recurrence(gctx, x_t, current_h, prefix, &next_h);

        // C. Update our rolling state pointer! 
        // The next tick of time will now use the memory we just generated.
        current_h = next_h;

        // D. Create a stencil hovering over the correct slot in our pre-allocated tensor
        size_t out_offset = t * out_assembled->nb[1];
        struct ggml_tensor* slot_view = ggml_view_2d(gctx.ctx, out_assembled, channels, 1, out_assembled->nb[1], out_offset);

        // E. Physically copy the math result from its island into the contiguous slot
        gctx.enforced_deps.push_back(ggml_cpy(gctx.ctx, y_t, slot_view));
    }

    // 4. Save the final hidden state of this audio chunk
    // current_h now points to the memory state at t = current_time_len.
    // We force GGML to compute it, and we save the pointer so we can extract 
    // the floats back into our AudioStateCache during Phase 3.
    gctx.enforced_deps.push_back(current_h);
    gctx.mamba_final_ssm_states.push_back(current_h);

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

// Extracts the overlap tails and SSM states from the computed GGML arena
// and safely stores them in the C++ AudioStateCache for the next frame.
void extract_state_cache(GraphContext& gctx) {
    AudioStateCache& state = gctx.state;
    const EdgeAudioModel& model = gctx.model;

    // A. Extract Encoder Overlap Tails (Strided Memory)
    // Shape is [Time, Channels]. We need the last `overlap_len` floats from each channel row.
    for (size_t i = 0; i < gctx.encoder_padded_tensors.size(); i++) {
        struct ggml_tensor* padded = gctx.encoder_padded_tensors[i];
        int padded_time = padded->ne[0];
        int channels = padded->ne[1];
        
        for (int c = 0; c < channels; c++) {
            float* src = ((float*)padded->data) + (c * padded_time) + (padded_time - state.overlap_len);
            float* dst = state.encoder_states[i].cache.data() + (c * state.overlap_len);
            memcpy(dst, src, state.overlap_len * sizeof(float));
        }
    }

    // B. Extract Mamba Conv1D Tails (Contiguous Memory)
    // Unrolled shape is [Channels, Time]. Time is on the Y-axis. 
    // The last 3 rows contain all channels perfectly side-by-side.
    for (size_t i = 0; i < gctx.mamba_padded_tensors.size(); i++) {
        struct ggml_tensor* padded = gctx.mamba_padded_tensors[i];
        int channels = padded->ne[0];
        int padded_time = padded->ne[1];
        int overlap = 3; // kernel=4 means overlap=3
        
        // Offset to the start of the final 3 rows
        size_t offset_bytes = (padded_time - overlap) * padded->nb[1];
        float* src = (float*)((char*)padded->data + offset_bytes);
        memcpy(state.mamba_states[i].conv_state.data(), src, channels * overlap * sizeof(float));
    }

    // C. Extract Final Mamba SSM Memory (Contiguous Memory)
    // The hidden state is natively [16, 1024]. We can copy it in one massive block.
    for (size_t i = 0; i < gctx.mamba_final_ssm_states.size(); i++) {
        struct ggml_tensor* final_h = gctx.mamba_final_ssm_states[i];
        memcpy(state.mamba_states[i].ssm_state.data(), final_h->data, ggml_nbytes(final_h));
    }

    // D. Extract Decoder Overlap Tails (Strided Memory)
    // x_tail was sliced in the helper, but the underlying RAM still belongs to the parent matrix.
    // We must respect x_tail->nb[1] to jump to the correct row.
    for (size_t i = 0; i < gctx.decoder_tail_tensors.size(); i++) {
        struct ggml_tensor* x_tail = gctx.decoder_tail_tensors[i];
        int overlap = x_tail->ne[0];
        int channels = x_tail->ne[1];
        
        for (int c = 0; c < channels; c++) {
            float* src = (float*)((char*)x_tail->data + c * (x_tail->nb[1] / sizeof(float)));
            float* dst = state.decoder_states[i].cache.data() + (c * overlap);
            memcpy(dst, src, overlap * sizeof(float));
        }
    }
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
    // Allocate 32mb for the intermediate math. In a fully optimized v2, 
    // we would allocate this once at startup and reuse it. For v1, allocating
    // a single block per frame is perfectly fast enough for edge real-time.
    // We'll also want to run a test eventually to determine the _actual_ amount of 
    // memory used by the graph and the update our hard coded 32MB amount.
    // 32MB provides reasonable headroom for testing for now though.
    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024, // 32 MB - Revisit this
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    struct ggml_context* ctx = ggml_init(params);
    struct ggml_cgraph* ggml_graph = ggml_new_graph(ctx);

    // We must track specific operations that GGML's backward-walker might miss
    std::vector<struct ggml_tensor*> enforced_deps;
    // We save the skip connections for the Decoder (Phase 3)
    std::vector<struct ggml_tensor*> skips;
    // We save the padded input tensors so we can crop their tails at the very end
    std::vector<struct ggml_tensor*> encoder_padded_tensors;
    std::vector<struct ggml_tensor*> mamba_padded_tensors;
    std::vector<struct ggml_tensor*> mamba_final_ssm_states;
    std::vector<struct ggml_tensor*> decoder_tail_tensors;

    // Package the references cleanly
    GraphContext graph_context = {
        ctx,
        model,
        state,
        enforced_deps,
        skips,
        encoder_padded_tensors,
        mamba_padded_tensors,
        mamba_final_ssm_states,
        decoder_tail_tensors
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
        x = execute_encoder_layer(graph_context, x, i, current_time_len, current_channels);
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
        x = execute_mamba_block(graph_context, x, i, current_time_len);
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
    // 5. DECODER LOOP (Semantic to Acoustic Representation)
    // ========================================================================
    for (int i = 0; i < model.num_encoder_layers; i++) { // we can use num_encoder_layers here because the encoder/decoder are symmetrical.
        x = execute_decoder_layer(graph_context, x, i, current_time_len, current_channels);
    }

    printf("[DEBUG] Phase 3 Complete. Final shape: [Time: %d, Channels: %d]\n", current_time_len, current_channels);

    // ========================================================================
    // 6. GRAPH EXECUTION (Phase 4)
    // ========================================================================
    // A. Build the forward graph starting from our final audio output
    ggml_build_forward_expand(ggml_graph, x);

    // B. Force the graph to evaluate our state-tracking copies and SSM updates
    for (struct ggml_tensor* dep : graph_context.enforced_deps) {
        ggml_build_forward_expand(ggml_graph, dep);
    }

    // C. Execute the math on the CPU
    // For real-time edge processing on tiny chunks, 1 thread minimizes context-switching overhead.
    int n_threads = 1; 
    ggml_graph_compute_with_ctx(ctx, ggml_graph, n_threads);

    // D. The graph is finished. Physically copy the final acoustic output to the OS buffer.
    memcpy(output_audio, x->data, num_samples * sizeof(float));

    // ========================================================================
    // 7. STATE EXTRACTION (Phase 5 - Cleanup for the next frame)
    // ========================================================================
    
    extract_state_cache(graph_context);

    // Destroy the compute arena. The extracted state cache survives safely in C++.
    ggml_free(ctx);
    return true;

    ggml_free(ctx);
    return true; 
}
