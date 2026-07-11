# EdgeAudioUSE: C++ Performance Optimization Roadmap

This document outlines the architectural and hardware-level optimizations required to bring the EdgeAudioUSE C++ inference engine below the strict 8ms real-time latency budget. 

**CRITICAL PREREQUISITE:** Never apply these optimizations until the model achieves a "Golden Execution" (MSE < 1e-5 against the PyTorch baseline). Optimizing a mathematically broken graph makes debugging impossible.

---

### 1. Hoist the `malloc` Penalty (Zero-Allocation Audio Thread)
**Current State:** Inside `process_audio_frame`, the engine asks the OS to allocate and destroy a 32MB memory arena (`ggml_init` / `ggml_free`) exactly 125 times a second. 
**The Fix:**
Audio callbacks must be strictly lock-free and allocation-free. 
* Move the `ggml_init` call into the `main` setup phase.
* Pass the persistent `ggml_context` pointer into `process_audio_frame`.
* Use `ggml_reset(ctx)` at the start of each frame to wipe the state without releasing the physical RAM back to the operating system.

### 2. Cache the DAG Topology
**Current State:**
The engine manually reconstructs the Directed Acyclic Graph (DAG) using `ggml_build_forward_expand` on every single 8ms chunk, walking the tree and calculating tensor geometries from scratch.
**The Fix:**
The neural network architecture is completely static. 
* Build the `ggml_cgraph` exactly once during application startup.
* For each new audio frame, write the new PCM floats directly into the `input_audio` tensor's memory pointer.
* Execute the pre-compiled graph directly.

### 3. Threading and Hardware Acceleration
**Current State:**
The graph executor is hardcoded to `n_threads = 1`. While this prevents context-switching overhead on tiny batches, it bottlenecks the massive 512-channel matrix multiplications in the U-Net and Mamba projections.
**The Fix:**
* **Multi-threading:** Expose `n_threads` as a CLI argument and benchmark 2, 4, and 8 threads to find the optimal balance between compute parallelization and thread-spinning overhead.
* **Apple Accelerate:** Ensure the CMake build explicitly links the `Accelerate` framework on macOS, allowing GGML to automatically route dense `ggml_mul_mat` operations to the M-series AMX (Apple Matrix) coprocessors.

### 4. Compiler Flags and Binary Stripping
**Current State:**
Basic `-O3` and `-ffast-math` are enabled in CMake.
**The Fix:**
* **Disable Exceptions:** Add `-fno-exceptions` and `-fno-rtti` to the CMake targets. DSP engines do not use C++ `throw`/`catch` blocks. Disabling them removes hidden stack-unwinding code, reducing the binary size and providing a 5-10% speed boost.
* **Link-Time Optimization (LTO):** Add `-flto` to allow the compiler to aggressively inline functions across different `.cpp` files, removing function call overhead inside the tightly nested Mamba loops.