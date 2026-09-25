cmake_minimum_required(VERSION 3.22.1)

project(llama_jni)

# Set C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android 15 16KB page size compliance
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Compilation flags: -O3 -fPIC
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# ARM64 optimizations
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+crc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crc")
endif()

# llama.cpp options
set(GGML_CPU_AARCH64 ON CACHE BOOL "Enable ARM64 support")
set(GGML_DOTPROD ON CACHE BOOL "Enable ARM NEON dot product")

# Enable Vulkan GPU backend for Adreno 830 GPU
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan GPU backend" FORCE)
set(GGML_VULKAN ON)
set(GGML_VULKAN_CHECK_RESULTS OFF CACHE BOOL "Disable Vulkan check results" FORCE)
set(GGML_VULKAN_CHECK_RESULTS OFF)

set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP to avoid packaging libomp.so" FORCE)
set(GGML_STATIC OFF CACHE BOOL "Build as shared library")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

# Disable unused features
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server")

# Locate Vulkan library
find_library(VULKAN_LIB vulkan REQUIRED)

# Android NDK sysroot only provides C Vulkan headers (<vulkan/vulkan.h>) and lacks
# C++ Vulkan-Hpp (<vulkan/vulkan.hpp>) and SPIR-V headers (<spirv/unified1/spirv.hpp>).
# Fetch official Khronos Vulkan-Headers and SPIRV-Headers to provide them for the NDK build.
if(ANDROID)
    include(FetchContent)
    FetchContent_Declare(
        vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG        vulkan-sdk-1.3.296.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_Declare(
        spirv_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
        GIT_TAG        vulkan-sdk-1.3.296.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(vulkan_headers)
    if(NOT vulkan_headers_POPULATED)
        FetchContent_Populate(vulkan_headers)
    endif()
    FetchContent_GetProperties(spirv_headers)
    if(NOT spirv_headers_POPULATED)
        FetchContent_Populate(spirv_headers)
    endif()

    set(Vulkan_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include" CACHE PATH "Vulkan Include Dir" FORCE)
    set(Vulkan_INCLUDE_DIRS "${vulkan_headers_SOURCE_DIR}/include" CACHE PATH "Vulkan Include Dirs" FORCE)

    include_directories(BEFORE
        ${vulkan_headers_SOURCE_DIR}/include
        ${spirv_headers_SOURCE_DIR}/include
    )
endif()

# Add llama.cpp
add_subdirectory(src/main/cpp/llama.cpp)

# ggml-base is linked as a standalone shared object and contains C++ sources.
# On Android/NDK it must explicitly link libc++ or symbols such as operator new
# and std::runtime_error stay unresolved during the native build.
if(ANDROID)
    set(GGML_TARGETS ggml-base ggml ggml-cpu llama)
    if(TARGET ggml-vulkan)
        list(APPEND GGML_TARGETS ggml-vulkan)
    endif()
    foreach(target ${GGML_TARGETS})
        if(TARGET ${target})
            target_link_libraries(${target} PRIVATE c++_shared)
        endif()
    endforeach()
    if(TARGET ggml-vulkan)
        target_include_directories(ggml-vulkan BEFORE PRIVATE
            ${vulkan_headers_SOURCE_DIR}/include
            ${spirv_headers_SOURCE_DIR}/include
        )
        target_link_libraries(ggml-vulkan PRIVATE ${VULKAN_LIB} android log)
    endif()
endif()

# JNI wrapper library - simplified without common library
add_library(llama_jni SHARED
    src/main/cpp/jni_wrapper.cpp
)

target_compile_options(llama_jni PRIVATE -O3 -fPIC)

target_link_libraries(llama_jni
    llama
    c++_shared
    android
    log
    ${VULKAN_LIB}
)

# Link vulkan library (-lvulkan) for Android API level 28+ if Vulkan backend is built
if(ANDROID AND TARGET ggml-vulkan)
    target_link_libraries(llama_jni -lvulkan)
endif()

target_include_directories(llama_jni PRIVATE
    src/main/cpp/llama.cpp/include
    src/main/cpp/llama.cpp/ggml/include
)

# Native bridge alias
add_library(vaultlm_native ALIAS llama_jni)
