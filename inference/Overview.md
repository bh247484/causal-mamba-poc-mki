### 1. The Core Philosophy: Data-Oriented, not Object-Oriented

* **Do not use heavy OOP (Classes, Inheritance, Virtual Methods).** `ggml` is written in pure C. If you wrap every GGML tensor in a C++ Class with getters and setters, you will fight the library constantly.
* **Use Data-Oriented Design.** You should use C++ features (like `std::vector`, `std::string`, and `std::thread`), but the core architecture should be plain `structs` passed to functional-style `functions`.

### 2. The File Structure

Keep it flat, modular, and completely separated by concern.

* `main.cpp` - Program lifecycle, CLI arguments, and tying the subsystems together.
* `model.h / .cpp` - Reads the `.bin` file and holds the **Weight Arena** (Tasks 1 & 2).
* `state.h / .cpp` - Holds your streaming history arrays (Encoder cache, Decoder cache, Mamba $h_t$).
* `graph.h / .cpp` - The translation of `_cached_streaming_forward`. Builds the GGML math operations (Task 3).
* `audio.h / .cpp` - `miniaudio` wrappers and the Ring Buffer implementation (Task 4).

### 3. Integrating the Libraries (Cleanly)

* **miniaudio:** This is a "Single Header Library." You do not compile it. You drop `miniaudio.h` into your folder. In exactly *one* file (e.g., `audio.cpp`), you write `#define MINIAUDIO_IMPLEMENTATION` before including it. Everywhere else, you just include the header.
* **ggml:** Add the `ggml` repository as a Git submodule. Your `CMakeLists.txt` will simply `add_subdirectory(ggml)` and link the resulting `ggml` library to your executable.

---

### 4. Clarifying the Architecture (The 3 Subsystems)

To clear up your fuzziness on how audio hits the GGML math, you must separate your mental model into three distinct physical spaces in RAM:

#### A. The Weight Engine (`model.cpp`)

This is read-only memory. It loads your `.bin` file into a `ggml_context` (the Weight Arena). Once the app boots, these bytes never change.

#### B. The Audio Thread (`audio.cpp`)

The OS (macOS CoreAudio, Windows WASAPI) owns this thread. It wakes up unpredictably and says, *"Give me 512 samples of audio NOW."* **CRITICAL RULE:** You cannot do machine learning, allocate memory, or lock a mutex on this thread. If you do, the audio will glitch.

* This thread strictly reads from the microphone into a **Ring Buffer**, and writes from the **Ring Buffer** to the speaker. That is it.

#### C. The Inference Worker Thread (`graph.cpp` - The "Fuzzy" Part)

This is a background C++ `std::thread` that runs an infinite `while(true)` loop. It watches the Ring Buffer.

Here is exactly how the audio multiplies against the GGML structures:

1. **The Trigger:** The worker sees 128 new samples in the Ring Buffer. It pulls them out.
2. **The Injection:** It copies those 128 floats into a designated "input" `ggml_tensor`. It copies the overlapping cache into the "state" `ggml_tensor`s.
3. **The Graph Build:** *You actually rebuild the mathematical graph in C++ every single frame.* You write a function that calls `ggml_conv_1d`, `ggml_add`, etc., linking the input tensor to the weight tensors. (This takes microseconds because it just builds a struct of pointers; it does no math yet).
4. **The Compute:** You call `ggml_graph_compute()`. GGML blasts the graph across your CPU cores.
5. **The Extraction:** You read the floats directly out of the final output `ggml_tensor`, push them into the speaker Ring Buffer, and copy the new Mamba states back to your `state.cpp` arrays.

---

### 5. Where You Might Over-Engineer

1. **Dynamic Graphing:** You might be tempted to build a dynamic graph that handles variable chunk sizes (e.g., sometimes 128, sometimes 256). **Don't.** Hardcode everything to exactly 128 samples. The Ring Buffer handles the variable OS buffer sizes; the ML model should be stubbornly static.
2. **Memory Management:** You might try to `new` and `delete` arrays for your audio chunks. **Don't.** Pre-allocate your Ring Buffer to a fixed size (e.g., 4096 floats) at startup and never allocate memory again.
3. **Complex Queues:** You might look into `std::mutex` or condition variables to pass data between the Audio Thread and the Inference Thread. **Don't.** Use a lock-free Single-Producer Single-Consumer (SPSC) ring buffer. (There are simple 50-line implementations we can write).
4. **The GGUF Rabbit Hole:** We already avoided this, but sticking to the `.bin` format keeps `model.cpp` under 100 lines of code.
