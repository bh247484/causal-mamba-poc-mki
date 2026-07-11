cmake_minimum_required(VERSION 3.14)
project(EdgeAudioUSE VERSION 1.0.0 LANGUAGES C CXX)

# Require C++17 for modern threading (std::thread) and filesystem utilities later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- DIRECTORY STRUCTURE CONVENTIONS ---
# We keep our root directory clean. 
# All custom application code lives in src/
# External submodules (ggml) live in their respective root folders.

# 1. Bring in the GGML submodule
# This automatically builds GGML as a static library we can link against.
add_subdirectory(ggml)

# 2. Define our main executable
add_executable(edge_audio)

# 3. Append source files to our target
# This pattern is highly extensible. When you create graph.cpp or audio.cpp,
# just add them here. We explicitly list them rather than using file(GLOB...) 
# to prevent CMake caching bugs when adding/removing files.
target_sources(edge_audio PRIVATE
    src/main.cpp
    src/model.cpp
    src/state.cpp
    src/graph.cpp
)

# 4. Link GGML and include directories
target_link_libraries(edge_audio PRIVATE ggml)

# Tell the compiler where to find our headers and GGML's headers
target_include_directories(edge_audio PRIVATE 
    src
    ggml/include/ggml
    ggml/src
)

# Optional: Add aggressive optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(edge_audio PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(edge_audio PRIVATE -O3 -ffast-math)
    endif()
endif()

# Other optimization flags to consider later:
# -fno-exceptions and -fno-rtti.
#C++ exceptions add hidden memory overhead to every function.
# Many edge C++ projects disable them entirely for a 5-10% speed boost.


# ---------------------------------------------------------
# Target 2: The Real-Time Hardware Benchmark
# ---------------------------------------------------------
add_executable(benchmark_looper 
    src/benchmark.cpp 
    src/model.cpp 
    src/state.cpp 
    src/graph.cpp
)

# 1. Link the exact same GGML base library
target_link_libraries(benchmark_looper PRIVATE ggml)

# 2. Port the include directories so it can find the headers
target_include_directories(benchmark_looper PRIVATE 
    src
    ggml/include/ggml
    ggml/src
)

# 3. Port the aggressive optimization flags for latency testing
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(benchmark_looper PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(benchmark_looper PRIVATE -O3 -ffast-math)
    endif()
endif()

# (Optional macOS requirement for miniaudio hardware access)
# if (APPLE)
#     find_library(COREAUDIO_LIB CoreAudio)
#     find_library(AUDIOTOOLBOX_LIB AudioToolbox)
#     target_link_libraries(benchmark_looper PRIVATE ${COREAUDIO_LIB} ${AUDIOTOOLBOX_LIB})
# endif()