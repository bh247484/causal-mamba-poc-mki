cmake_minimum_required(VERSION 3.14)
project(EdgeAudioUSE VERSION 1.0.0 LANGUAGES C CXX)

# Require C++17 for modern threading (std::thread) and filesystem utilities later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable global Link-Time Optimization (LTO / Interprocedural Optimization)
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
    message(STATUS "[OPTIMIZATION] Apple platform detected. Binding Accelerate Framework for AMX/BLAS.")
    set(GGML_ACCELERATE ON CACHE BOOL "Build GGML with Accelerate framework support" FORCE)
    
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(COREAUDIO_LIB CoreAudio REQUIRED)
    find_library(AUDIOTOOLBOX_LIB AudioToolbox REQUIRED)
    
    set(PLATFORM_LIBS ${ACCELERATE_FRAMEWORK} ${COREAUDIO_LIB} ${AUDIOTOOLBOX_LIB})
else()
    message(STATUS "[OPTIMIZATION] Targeting x86/Enterprise architecture.")
    set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
    set(PLATFORM_LIBS "")
endif()

# ----------------------------------------------------------------------------
# 2. SUBDIRECTORIES
# ----------------------------------------------------------------------------
add_subdirectory(ggml)

# ----------------------------------------------------------------------------
# 3. COMPILER OPTIONS & STRIPPING (Target Agnostic)
# ----------------------------------------------------------------------------
set(COMMON_INCLUDES
    src
    ggml/include/ggml
    ggml/src
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        set(RELEASE_FLAGS 
            /O2          
            /fp:fast     
            /GR-         
            /EHs-c-      
        )
    else()
        set(RELEASE_FLAGS 
            -O3 
            -ffast-math 
            -fno-exceptions 
            -fno-rtti       
        )
    endif()
endif()

# ----------------------------------------------------------------------------
# 4. TARGET COMPILATION & LINKING
# ----------------------------------------------------------------------------

# Target 1: Production Engine Binary
add_executable(edge_audio
    src/main.cpp
    src/model.cpp
    src/state.cpp
    src/graph.cpp
    src/probe.cpp
)
target_include_directories(edge_audio PRIVATE ${COMMON_INCLUDES})
target_link_libraries(edge_audio PRIVATE ggml ${PLATFORM_LIBS})
if(RELEASE_FLAGS)
    target_compile_options(edge_audio PRIVATE ${RELEASE_FLAGS})
endif()


# Target 2: Real-Time Hardware Benchmark
add_executable(benchmark_looper
    src/benchmark.cpp
    src/model.cpp
    src/state.cpp
    src/graph.cpp
    src/probe.cpp
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