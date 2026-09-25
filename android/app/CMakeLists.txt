cmake_minimum_required(VERSION 3.22.1)

project(vaultlm_native)

# Set C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android 15 16KB page size compliance
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Ensure target compilation flags include -O3 -fPIC
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# ARM64 optimizations
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+crc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crc")
endif()

# 1. Set GGML_VULKAN ON and GGML_VULKAN_CHECK_RESULTS OFF before calling add_subdirectory for llama.cpp
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan GPU backend" FORCE)
set(GGML_VULKAN ON)
set(GGML_VULKAN_CHECK_RESULTS OFF CACHE BOOL "Disable Vulkan check results" FORCE)
set(GGML_VULKAN_CHECK_RESULTS OFF)

set(GGML_CPU_AARCH64 ON CACHE BOOL "Enable ARM64 support")
set(GGML_DOTPROD ON CACHE BOOL "Enable ARM NEON dot product")
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP" FORCE)
set(GGML_STATIC OFF CACHE BOOL "Build as shared library")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server")

# 2. Locate vulkan using find_library(VULKAN_LIB vulkan REQUIRED)
find_library(VULKAN_LIB vulkan REQUIRED)

# Call add_subdirectory for llama.cpp
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../local_plugins/llama_flutter_android/android/src/main/cpp/llama.cpp")
if(EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)
endif()

# Target shared library containing the native bridge
add_library(vaultlm_native SHARED
    src/vaultlm_bridge.cpp
)

target_compile_options(vaultlm_native PRIVATE -O3 -fPIC)

target_include_directories(vaultlm_native PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

# 3. Link ${VULKAN_LIB} into the target shared library containing the native bridge (vaultlm_native)
target_link_libraries(vaultlm_native
    ${VULKAN_LIB}
    android
    log
)

if(TARGET llama)
    target_link_libraries(vaultlm_native llama)
endif()
