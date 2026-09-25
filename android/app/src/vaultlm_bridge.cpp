#include <jni.h>
#include <string>
#include <algorithm>
#include "llama.h"
#include "ggml-backend.h"

#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#define LOG_TAG "VaultLMNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOG_TAG "VaultLMNative"
#define LOGI(...) do { printf("[INFO] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGW(...) do { printf("[WARN] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { printf("[ERROR] " __VA_ARGS__); printf("\n"); } while(0)
#endif

extern "C" {

/**
 * 1. Call ggml_backend_load_all() to ensure the Vulkan driver registers the Adreno 830.
 */
void vaultlm_init_backends() {
    ggml_backend_load_all();
    LOGI("VaultLM native: backends initialized (Vulkan/Adreno 830 registered)");
}

/**
 * 2. In llama_model_params, bind n_gpu_layers:
 *    - Default to 999 when the user selects GPU Fast or Auto Fast mode.
 *    - Set to 0 only when CPU Safe is explicitly selected.
 * 4. Add clear Android logcat tags (LOGI("Offloading %d layers to GPU", n_gpu_layers))
 */
llama_model_params vaultlm_create_model_params(int n_gpu_layers) {
    // Call ggml_backend_load_all() to ensure Vulkan driver registers Adreno 830
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();

    // Default to 999 for GPU Fast / Auto Fast; set to 0 only when CPU Safe is selected
    if (n_gpu_layers == 0) {
        model_params.n_gpu_layers = 0;
    } else {
        model_params.n_gpu_layers = (n_gpu_layers > 0) ? n_gpu_layers : 999;
    }

    LOGI("Offloading %d layers to GPU", (int)model_params.n_gpu_layers);
    return model_params;
}

/**
 * 3. Configure KV cache quantization defaults:
 *    - cparams.type_k = GGML_TYPE_Q8_0
 *    - cparams.type_v = GGML_TYPE_Q8_0
 */
llama_context_params vaultlm_create_context_params(int n_ctx, int n_threads, int n_batch) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? (uint32_t)n_ctx : 4096;
    cparams.n_threads = n_threads > 0 ? (int32_t)n_threads : 4;
    cparams.n_batch = n_batch > 0 ? (uint32_t)n_batch : 512;
    cparams.n_ubatch = cparams.n_batch;

    // KV cache quantization defaults
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;

    return cparams;
}

}
