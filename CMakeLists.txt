cmake_minimum_required(VERSION 3.16)
project(eif-runtime VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
elseif(APPLE)
    add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)
endif()

# OpenMP support for parallel inference
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found: Enabling multi-threaded inference")
    add_compile_options(${OpenMP_C_FLAGS})
endif()

# Optional Vulkan GPU Compute support
option(EIF_ENABLE_VULKAN "Enable optional Vulkan GPU acceleration" OFF)
if(EIF_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found: Enabling optional EIF GPU compute acceleration")
        add_compile_definitions(EIF_USE_VULKAN=1)
        include_directories(${Vulkan_INCLUDE_DIRS})
    else()
        message(STATUS "Vulkan not found: Gracefully defaulting to CPU-only execution (zero dependency)")
    endif()
endif()

include_directories(include)

# Core sources
set(EIF_SOURCES
    src/eif_llm.c
    src/eif_qwen35.c
    src/eif_tinyllm.c
    src/eif_bpe_tokenizer.c
    src/eif_quantize_bitnet.c
    src/eif_bert.c
    src/eif_gpu.c
)

# 1. Static Library (libeif_runtime.a)
add_library(eif_runtime_static STATIC ${EIF_SOURCES})
set_target_properties(eif_runtime_static PROPERTIES OUTPUT_NAME "eif_runtime")
target_include_directories(eif_runtime_static PUBLIC include)
if(UNIX)
    target_link_libraries(eif_runtime_static PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(eif_runtime_static PRIVATE OpenMP::OpenMP_C)
endif()
if(EIF_ENABLE_VULKAN AND Vulkan_FOUND)
    target_link_libraries(eif_runtime_static PRIVATE Vulkan::Vulkan)
endif()

# 2. Shared Library (libeif_runtime.so / .dll)
add_library(eif_runtime_shared SHARED ${EIF_SOURCES})
set_target_properties(eif_runtime_shared PROPERTIES OUTPUT_NAME "eif_runtime")
target_include_directories(eif_runtime_shared PUBLIC include)
if(UNIX)
    target_link_libraries(eif_runtime_shared PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(eif_runtime_shared PRIVATE OpenMP::OpenMP_C)
endif()
if(EIF_ENABLE_VULKAN AND Vulkan_FOUND)
    target_link_libraries(eif_runtime_shared PRIVATE Vulkan::Vulkan)
endif()

# 3. CLI Runner Executable (eif-run)
add_executable(eif-run cli/main.c)
target_link_libraries(eif-run PRIVATE eif_runtime_static)
if(UNIX)
    target_link_libraries(eif-run PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(eif-run PRIVATE OpenMP::OpenMP_C)
endif()

# 4. Feature Test Suite (test_all_features)
add_executable(test_all_features tests/test_all_features.c)
target_link_libraries(test_all_features PRIVATE eif_runtime_static)
if(UNIX)
    target_link_libraries(test_all_features PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_all_features PRIVATE OpenMP::OpenMP_C)
endif()

# 5. Live Model Test Suite (test_live_models)
add_executable(test_live_models tests/test_live_models.c)
target_link_libraries(test_live_models PRIVATE eif_runtime_static)
if(UNIX)
    target_link_libraries(test_live_models PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_live_models PRIVATE OpenMP::OpenMP_C)
endif()

# 6. ALiBi & Benchmark Test Suite (test_alibi_and_benchmark)
add_executable(test_alibi_and_benchmark tests/test_alibi_and_benchmark.c)
target_link_libraries(test_alibi_and_benchmark PRIVATE eif_runtime_static)
if(UNIX)
    target_link_libraries(test_alibi_and_benchmark PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_alibi_and_benchmark PRIVATE OpenMP::OpenMP_C)
endif()

enable_testing()
add_test(NAME test_all_features COMMAND test_all_features)
add_test(NAME test_live_models COMMAND test_live_models)
add_test(NAME test_alibi_and_benchmark COMMAND test_alibi_and_benchmark)

# Install rules
install(TARGETS eif_runtime_static eif_runtime_shared eif-run
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
