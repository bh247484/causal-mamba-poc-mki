cmake_minimum_required(VERSION 3.14)
project(EdgeAudioUSE VERSION 1.0.0 LANGUAGES C CXX)

# Require C++17 for modern threading (std::thread) and filesystem utilities later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable global Link-Time Optimization (LTO / Interprocedural Optimization)
# This allows the compiler to inline code aggressively across separate compilation units (.cpp files).
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT error)
if(lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "[OPTIMIZATION] Link-Time Optimization (LTO) enabled for Release builds.")
else()
    message(STATUS "[WARNING] LTO not supported by compiler: ${error}")
endif()

# ----------------------------------------------------------------------------
# 1. HARDWARE & SYSTEM-DEPENDENT ACCELERATION (Isolated Profiles)
# ----------------------------------------------------------------------------
if(APPLE)
    # --- macOS / Apple Silicon Development Profile ---
    message(STATUS "[OPTIMIZATION] Apple platform detected. Binding Accelerate Framework for AMX/BLAS.")
    
    # Force GGML's internal configuration to build with Accelerate support
    set(GGML_ACCELERATE ON CACHE BOOL "Build GGML with Accelerate framework support" FORCE)
    
    # Find and cache the native Apple libraries
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(COREAUDIO_LIB CoreAudio REQUIRED)
    find_library(AUDIOTOOLBOX_LIB AudioToolbox REQUIRED)
    
    set(PLATFORM_LIBS ${ACCELERATE_FRAMEWORK} ${COREAUDIO_LIB} ${AUDIOTOOLBOX_LIB})
else()
    # --- Enterprise Windows/Linux x86 Target Profile ---
    message(STATUS "[OPTIMIZATION] Targeting x86/Enterprise architecture.")
    
    # Explicitly turn off Apple accelerators
    set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
    
    # NOTE FOR ENTERPRISE DEPLOYMENT:
    # Older enterprise laptops may lack AVX2 or AVX-512 instructions. 
    # GGML internally auto-detects SIMD capabilities at compile time based on compiler flags.
    # For a safe generic enterprise binary, avoid hardcoding '-march=native'.
    # Instead, let the build system target standard baseline architectures (like SSE3/AVX).
    set(PLATFORM_LIBS "")
endif()

# ----------------------------------------------------------------------------
# 2. SUBDIRECTORIES
# ----------------------------------------------------------------------------
# Bring in the GGML submodule (will consume the GGML_ACCELERATE flag set above)
add_subdirectory(ggml) [cite: 4]

# ----------------------------------------------------------------------------
# 3. COMPILER OPTIONS & STRIPPING (Target Agnostic)
# ----------------------------------------------------------------------------
# Define common include paths for clean code tracking
set(COMMON_INCLUDES
    src
    ggml/include/ggml
    ggml/src
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # Universal Windows Enterprise Optimizations
        set(RELEASE_FLAGS 
            /O2          # Maximize speed
            /fp:fast     # Aggressive floating-point math
            /GR-         # Disable Run-Time Type Information (RTTI) to shred binary size
            /EHs-c-      # Completely disable C++ Exception Handlers
        )
    else()
        # Universal Unix/Linux/macOS Optimizations
        set(RELEASE_FLAGS 
            -O3 
            -ffast-math 
            -fno-exceptions # Shaves 5-10% CPU overhead by stripping stack-unwinding code 
            -fno-rtti       # Disables runtime class tracking, lowering binary size 
        )
    endif()
endif()

# ----------------------------------------------------------------------------
# 4. TARGET COMPILATION & LINKING
# ----------------------------------------------------------------------------

# Target 1: Production Engine Binary
add_executable(edge_audio)
target_sources(edge_audio PRIVATE [cite: 7]
    src/main.cpp [cite: 7]
    src/model.cpp [cite: 7]
    src/state.cpp [cite: 7]
    src/graph.cpp [cite: 7]
    src/probe.cpp [cite: 7]
)
target_include_directories(edge_audio PRIVATE ${COMMON_INCLUDES})
target_link_libraries(edge_audio PRIVATE ggml ${PLATFORM_LIBS})
if(RELEASE_FLAGS)
    target_compile_options(edge_audio PRIVATE ${RELEASE_FLAGS})
endif()


# Target 2: Real-Time Hardware Benchmark [cite: 10]
add_executable(benchmark_looper [cite: 10]
    src/benchmark.cpp [cite: 10]
    src/model.cpp [cite: 10]
    src/state.cpp [cite: 10]
    src/graph.cpp [cite: 10]
    src/probe.cpp [cite: 10]
)
target_include_directories(benchmark_looper PRIVATE ${COMMON_INCLUDES})
target_link_libraries(benchmark_looper PRIVATE ggml ${PLATFORM_LIBS})
if(RELEASE_FLAGS)
    target_compile_options(benchmark_looper PRIVATE ${RELEASE_FLAGS})
endif()


# Target 3: Offline File Processor
add_executable(offline_processor 
    src/offline_processor.cpp 
    src/graph.cpp 
    src/model.cpp 
    src/state.cpp
    src/probe.cpp
)
target_include_directories(offline_processor PRIVATE ${COMMON_INCLUDES})
target_link_libraries(offline_processor PRIVATE ggml ${PLATFORM_LIBS})
if(RELEASE_FLAGS)
    target_compile_options(offline_processor PRIVATE ${RELEASE_FLAGS})
endif()
