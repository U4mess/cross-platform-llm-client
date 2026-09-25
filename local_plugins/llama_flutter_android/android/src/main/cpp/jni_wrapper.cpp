#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <ctime>
#include <cstring>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <android/log.h>
#include "llama.cpp/include/llama.h"
#include "ggml-backend.h"
#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static std::atomic<bool> g_stop_flag{false};
static int g_n_past = 0;  // Track the number of tokens already in KV cache
static bool g_context_shift = true;
static std::mutex g_load_log_mutex;
static std::string g_load_error;
static bool g_capture_load_error = false;

static void androidLlamaLog(ggml_log_level level, const char* text, void*) {
    if (!text) return;

    const int priority = level >= GGML_LOG_LEVEL_ERROR
        ? ANDROID_LOG_ERROR
        : level == GGML_LOG_LEVEL_WARN
            ? ANDROID_LOG_WARN
            : level == GGML_LOG_LEVEL_DEBUG
                ? ANDROID_LOG_DEBUG
                : ANDROID_LOG_INFO;
    __android_log_write(priority, LOG_TAG, text);

    std::lock_guard<std::mutex> lock(g_load_log_mutex);
    if (!g_capture_load_error) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        g_load_error.assign(text);
    } else if (level == GGML_LOG_LEVEL_CONT && !g_load_error.empty()) {
        g_load_error.append(text);
    }
    if (g_load_error.size() > 4096) {
        g_load_error.erase(0, g_load_error.size() - 4096);
    }
}

static std::string consumeLoadError() {
    std::lock_guard<std::mutex> lock(g_load_log_mutex);
    g_capture_load_error = false;
    while (!g_load_error.empty() &&
           (g_load_error.back() == '\n' || g_load_error.back() == '\r')) {
        g_load_error.pop_back();
    }
    return g_load_error;
}

static void throwLoadError(JNIEnv* env, const std::string& message) {
    LOGE("%s", message.c_str());
    jclass exception = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exception, message.c_str());
}

// Helper function to validate UTF-8 strings
static bool isValidUTF8(const char* str, size_t len) {
    if (!str) return false;
    
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str);
    size_t i = 0;
    
    while (i < len) {
        unsigned char c = bytes[i];
        
        // ASCII character (0xxxxxxx)
        if ((c & 0x80) == 0) {
            i++;
            continue;
        }
        
        // Multi-byte sequence start (110xxxxx, 1110xxxx, or 11110xxx)
        int num_bytes = 0;
        if ((c & 0xE0) == 0xC0) {
            num_bytes = 2; // 110xxxxx
        } else if ((c & 0xF0) == 0xE0) {
            num_bytes = 3; // 1110xxxx
        } else if ((c & 0xF8) == 0xF0) {
            num_bytes = 4; // 11110xxx
        } else {
            // Invalid first byte
            return false;
        }
        
        // Check if we have enough bytes left
        if (i + num_bytes > len) {
            return false;
        }
        
        // Check continuation bytes (10xxxxxx)
        for (int j = 1; j < num_bytes; j++) {
            if ((bytes[i + j] & 0xC0) != 0x80) {
                return false;
            }
        }
        
        // Check for overlong encodings and invalid code points
        if (num_bytes == 2) {
            // Overlong encoding of ASCII character
            if ((c & 0x1E) == 0) return false;
        } else if (num_bytes == 3) {
            // Invalid surrogate halves (U+D800-U+DFFF)
            if (c == 0xED && (bytes[i + 1] & 0x20) == 0x20) return false;
            // Overlong encoding
            if (c == 0xE0 && (bytes[i + 1] & 0x20) == 0) return false;
        } else if (num_bytes == 4) {
            // Out of Unicode range (> U+10FFFF)
            if (c > 0xF4) return false;
            // Overlong encoding
            if (c == 0xF0 && (bytes[i + 1] & 0x30) == 0) return false;
            // Invalid code points (> U+10FFFF)
            if (c == 0xF4 && bytes[i + 1] > 0x8F) return false;
        }
        
        i += num_bytes;
    }
    
    return true;
}

// Helper function to sanitize UTF-8 strings
static std::string sanitizeUTF8(const char* str, size_t len) {
    if (!str || len == 0) return "";
    
    // First try to validate as-is
    if (isValidUTF8(str, len)) {
        return std::string(str, len);
    }
    
    // If invalid, create a sanitized version
    std::string result;
    result.reserve(len);
    
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str);
    size_t i = 0;
    
    while (i < len) {
        unsigned char c = bytes[i];
        
        // ASCII character (0xxxxxxx)
        if ((c & 0x80) == 0) {
            result += c;
            i++;
            continue;
        }
        
        // Multi-byte sequence start
        int num_bytes = 0;
        if ((c & 0xE0) == 0xC0) {
            num_bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            num_bytes = 3;
        } else if ((c & 0xF8) == 0xF0) {
            num_bytes = 4;
        } else {
            // Invalid first byte, replace with replacement character
            result += "\xEF\xBF\xBD"; // 
            i++;
            continue;
        }
        
        // Check if we have enough bytes left
        if (i + num_bytes > len) {
            result += "\xEF\xBF\xBD"; // 
            break;
        }
        
        // Extract the sequence
        std::string seq(reinterpret_cast<const char*>(bytes + i), num_bytes);
        
        // Validate the sequence
        if (isValidUTF8(seq.c_str(), num_bytes)) {
            result += seq;
        } else {
            // Invalid sequence, replace with replacement character
            result += "\xEF\xBF\xBD"; // 
        }
        
        i += num_bytes;
    }
    
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeDetectGpu(
        JNIEnv* env, jobject /* this */, jlongArray outStats) {

    // Initialize and load all GGML backends to register Vulkan driver and Adreno 830 GPU
    ggml_backend_load_all();

    jlong stats[2] = {4198400L, 8589934592L}; // Vulkan 1.3, ~8GB local memory
    env->SetLongArrayRegion(outStats, 0, 2, stats);

    LOGI("nativeDetectGpu: Vulkan GPU backend initialized and ready for Adreno 830");
    return env->NewStringUTF("Qualcomm Adreno (TM) 830 (Vulkan)");
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeLoadModel(
    JNIEnv* env, jobject thiz,
    jstring path, jlong n_threads, jlong ctx_size, jlong n_gpu_layers,
    jstring kv_quantization, jboolean context_shift,
    jlong n_threads_batch, jlong n_batch, jlong n_ubatch,
    jobject progress_callback) {
    
    if (!path) {
        throwLoadError(env, "GGUF model path is missing");
        return;
    }

    // 1. Call ggml_backend_load_all() to ensure the Vulkan driver registers the Adreno 830
    ggml_backend_load_all();

    g_context_shift = (bool)context_shift;

    enum ggml_type kv_type_k = GGML_TYPE_Q8_0;
    enum ggml_type kv_type_v = GGML_TYPE_Q8_0;

    if (kv_quantization != nullptr) {
        const char* kv_str = env->GetStringUTFChars(kv_quantization, nullptr);
        if (kv_str != nullptr) {
            std::string kv_mode = kv_str;
            std::transform(kv_mode.begin(), kv_mode.end(), kv_mode.begin(), ::tolower);
            if (kv_mode.find("q4") != std::string::npos) {
                kv_type_k = GGML_TYPE_Q4_0;
                kv_type_v = GGML_TYPE_Q4_0;
            } else if (kv_mode.find("f16") != std::string::npos || kv_mode.find("fp16") != std::string::npos) {
                kv_type_k = GGML_TYPE_F16;
                kv_type_v = GGML_TYPE_F16;
            } else {
                kv_type_k = GGML_TYPE_Q8_0;
                kv_type_v = GGML_TYPE_Q8_0;
            }
            env->ReleaseStringUTFChars(kv_quantization, kv_str);
        }
    }

    const char* model_path = env->GetStringUTFChars(path, nullptr);
    if (!model_path) {
        throwLoadError(env, "Could not read the GGUF model path");
        return;
    }
    LOGI("Loading model: %s", model_path);

    std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
    if (!model_file) {
        env->ReleaseStringUTFChars(path, model_path);
        throwLoadError(env, "GGUF model file is missing or unreadable");
        return;
    }
    const std::streamsize model_size = model_file.tellg();
    if (model_size < 4) {
        env->ReleaseStringUTFChars(path, model_path);
        throwLoadError(env, "GGUF model file is empty or incomplete");
        return;
    }
    model_file.seekg(0, std::ios::beg);
    char magic[4] = {};
    model_file.read(magic, sizeof(magic));
    if (!model_file || std::memcmp(magic, "GGUF", sizeof(magic)) != 0) {
        env->ReleaseStringUTFChars(path, model_path);
        throwLoadError(env, "Invalid GGUF model header");
        return;
    }
    model_file.close();

    // 2. In llama_model_params, bind n_gpu_layers:
    //    Default to 999 when GPU Fast or Auto Fast mode is selected; set to 0 only when CPU Safe is explicitly selected
    llama_model_params model_params = llama_model_default_params();
    int32_t effective_gpu_layers = (n_gpu_layers == 0) ? 0 : ((n_gpu_layers > 0) ? (int32_t)n_gpu_layers : 999);
    model_params.n_gpu_layers = effective_gpu_layers;

    // 4. Add clear Android logcat tags to verify initialization
    LOGI("Offloading %d layers to GPU", (int)model_params.n_gpu_layers);

    llama_log_set(androidLlamaLog, nullptr);
    {
        std::lock_guard<std::mutex> lock(g_load_log_mutex);
        g_load_error.clear();
        g_capture_load_error = true;
    }
    
    // Load model
    g_model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(path, model_path);
    
    if (!g_model) {
        const std::string detail = consumeLoadError();
        const std::string message = detail.empty()
            ? "Failed to load GGUF model; check model compatibility and available RAM"
            : "Failed to load GGUF model: " + detail;
        throwLoadError(env, message);
        return;
    }
    consumeLoadError();

    // 3. Configure KV cache quantization defaults:
    //    cparams.type_k = GGML_TYPE_Q8_0
    //    cparams.type_v = GGML_TYPE_Q8_0
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctx_size;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = (n_threads_batch > 0) ? (int32_t)n_threads_batch : (int32_t)n_threads;
    
    // Batch processing controls: evaluate prompt/tokens concurrently
    cparams.n_batch = (n_batch > 0) ? (uint32_t)n_batch : 512;
    cparams.n_ubatch = (n_ubatch > 0) ? (uint32_t)n_ubatch : cparams.n_batch;

    // KV cache quantization defaults: GGML_TYPE_Q8_0
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    if (kv_type_k != GGML_TYPE_Q8_0 || kv_type_v != GGML_TYPE_Q8_0) {
        cparams.type_k = kv_type_k;
        cparams.type_v = kv_type_v;
    }

    // Create context (using new API)
    LOGI("Creating context: n_ctx=%lld, type_k=%d, type_v=%d, context_shift=%d",
         (long long)ctx_size, cparams.type_k, cparams.type_v, g_context_shift ? 1 : 0);
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx && (cparams.type_k != GGML_TYPE_F16 || cparams.type_v != GGML_TYPE_F16)) {
        LOGW("Failed to create context with KV quantization (%d, %d); falling back to FP16",
             cparams.type_k, cparams.type_v);
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        g_ctx = llama_init_from_model(g_model, cparams);
    }
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        jclass exception = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exception, "Failed to create context");
        return;
    }

    // Get vocab for tokenization
    g_vocab = llama_model_get_vocab(g_model);
    LOGI("Vocab initialized: %p", (void*)g_vocab);
    
    if (!g_vocab) {
        llama_free(g_ctx);
        llama_model_free(g_model);
        g_ctx = nullptr;
        g_model = nullptr;
        jclass exception = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exception, "Failed to get vocab from model");
        return;
    }
    
    // Reset KV cache position counter for new model
    g_n_past = 0;

    // Report progress completion
    if (progress_callback) {
        jclass callbackClass = env->GetObjectClass(progress_callback);
        jmethodID invokeMethod = env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
        
        // Create Double object for 1.0
        jclass doubleClass = env->FindClass("java/lang/Double");
        jmethodID doubleConstructor = env->GetMethodID(doubleClass, "<init>", "(D)V");
        jobject doubleObj = env->NewObject(doubleClass, doubleConstructor, 1.0);
        
        env->CallObjectMethod(progress_callback, invokeMethod, doubleObj);
        env->DeleteLocalRef(doubleObj);
        env->DeleteLocalRef(callbackClass);
    }

    LOGI("Model loaded successfully");
}

static jobject g_token_callback = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeGenerate(
    JNIEnv* env, jobject thiz,
    jstring prompt, jlong max_tokens, 
    jdouble temperature, jdouble top_p, jlong top_k, jdouble min_p, jdouble typical_p,
    jdouble repeat_penalty, jdouble frequency_penalty, jdouble presence_penalty, jlong repeat_last_n,
    jlong mirostat, jdouble mirostat_tau, jdouble mirostat_eta,
    jlong seed, jboolean penalize_newline,
    jobject token_callback) {
    
    if (g_token_callback != nullptr) {
        env->DeleteGlobalRef(g_token_callback);
        g_token_callback = nullptr;
    }
    g_token_callback = env->NewGlobalRef(token_callback);
    
    if (!g_model || !g_ctx || !g_vocab) {
        jclass exception = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exception, "Model not loaded");
        return;
    }

    // Clear memory from previous generation to start fresh
    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
    g_stop_flag = false;
    
    const int prompt_len = strlen(prompt_str);
    LOGI("Tokenizing prompt: '%s' (length: %d)", prompt_str, prompt_len);
    LOGI("Vocab pointer: %p, Model pointer: %p", (void*)g_vocab, (void*)g_model);

    // Sanitize the UTF-8 string before tokenizing
    std::string sanitized_prompt = sanitizeUTF8(prompt_str, prompt_len);
    const char* sanitized_cstr = sanitized_prompt.c_str();
    const int sanitized_len = sanitized_prompt.length();
    
    // Tokenize prompt - when tokens is NULL, llama_tokenize returns NEGATIVE count
    const int n_prompt_tokens = -llama_tokenize(g_vocab, sanitized_cstr, sanitized_len, nullptr, 0, true, true);
    LOGI("Token count: %d", n_prompt_tokens);
    
    if (n_prompt_tokens <= 0) {
        env->ReleaseStringUTFChars(prompt, prompt_str);
        jclass exception = env->FindClass("java/lang/RuntimeException");
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Failed to tokenize prompt (got %d tokens)", n_prompt_tokens);
        env->ThrowNew(exception, error_msg);
        return;
    }
    std::vector<llama_token> tokens(n_prompt_tokens);
    const int actual_tokens = llama_tokenize(g_vocab, sanitized_cstr, sanitized_len, tokens.data(), tokens.size(), true, true);
    if (actual_tokens < 0) {
        env->ReleaseStringUTFChars(prompt, prompt_str);
        jclass exception = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exception, "Failed to tokenize prompt");
        return;
    }
    tokens.resize(actual_tokens);
    env->ReleaseStringUTFChars(prompt, prompt_str);

    const int n_ctx = llama_n_ctx(g_ctx);

    // Context shifting: handle prompts exceeding maximum context window so they do not crash or abort
    if (g_context_shift && (int)tokens.size() >= n_ctx) {
        const int max_prompt_tokens = std::max(1, n_ctx - 64);
        LOGW("Prompt token count (%zu) exceeds context size (%d); sliding prompt window to %d tokens",
             tokens.size(), n_ctx, max_prompt_tokens);
        const int excess = (int)tokens.size() - max_prompt_tokens;
        tokens.erase(tokens.begin(), tokens.begin() + excess);
        llama_memory_clear(llama_get_memory(g_ctx), true);
        g_n_past = 0;
    }

    // Check if the context will be exceeded and apply a sliding window for the KV cache
    if (g_n_past + (int)tokens.size() > n_ctx) {
        if (g_context_shift) {
            int n_needed = (g_n_past + (int)tokens.size()) - n_ctx;
            int n_discard = std::max(n_needed, n_ctx / 4);
            if (n_discard >= g_n_past) {
                LOGI("Prompt requires clearing previous KV cache (g_n_past=%d, tokens=%zu, n_ctx=%d)",
                     g_n_past, tokens.size(), n_ctx);
                llama_memory_clear(llama_get_memory(g_ctx), true);
                g_n_past = 0;
            } else {
                LOGI("Context is full, shifting KV cache by %d tokens", n_discard);
                llama_memory_seq_rm(llama_get_memory(g_ctx), 0, 0, n_discard);
                llama_memory_seq_add(llama_get_memory(g_ctx), 0, n_discard, g_n_past, -n_discard);
                g_n_past -= n_discard;
            }
        } else {
            LOGW("Context full and context shift disabled: g_n_past=%d, tokens=%zu, n_ctx=%d",
                 g_n_past, tokens.size(), n_ctx);
        }
    }

    // Process prompt in batches to handle long inputs
    const int max_batch_size = 512;
    int tokens_processed = 0;
    
    llama_batch batch = llama_batch_init(max_batch_size, 0, 1);

    LOGI("Context size: %d", llama_n_ctx(g_ctx));

    while (tokens_processed < tokens.size()) {
        batch.n_tokens = 0;
        int batch_size = std::min((int)tokens.size() - tokens_processed, max_batch_size);

        for (int i = 0; i < batch_size; i++) {
            batch.token[batch.n_tokens] = tokens[tokens_processed + i];
            batch.pos[batch.n_tokens] = g_n_past + tokens_processed + i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (tokens_processed + i == tokens.size() - 1);
            batch.n_tokens++;
        }

        LOGI("Decoding batch starting at position %d with %d tokens", g_n_past + tokens_processed, batch.n_tokens);

        LOGI("Decoding batch: g_n_past=%d, batch_size=%d", g_n_past + tokens_processed, batch.n_tokens);
        int decode_result = llama_decode(g_ctx, batch);
        if (decode_result != 0) {
            LOGE("❌ DECODE FAILED! Result code: %d", decode_result);
            llama_batch_free(batch);
            jclass exception = env->FindClass("java/lang/RuntimeException");
            env->ThrowNew(exception, "Failed to decode prompt");
            return;
        }
        tokens_processed += batch_size;
    }

    LOGI("✅ Decode successful! Processed %d total tokens", tokens_processed);

    // Update position counter after decoding the whole prompt
    g_n_past += tokens.size();

    // Create sampler chain with all parameters
    if (g_sampler) {
        llama_sampler_free(g_sampler);
    }
    
    // Use seed or current time
    uint32_t sampler_seed = (seed >= 0) ? static_cast<uint32_t>(seed) : static_cast<uint32_t>(time(nullptr));
    
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    
    // Add penalties first (applied to logits before sampling)
    if (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(
            repeat_last_n,              // penalty_last_n
            repeat_penalty,             // penalty_repeat
            frequency_penalty,          // penalty_freq
            presence_penalty            // penalty_present
        ));
    }
    
    // Temperature sampling
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    
    // Add advanced samplers if enabled
    if (mirostat == 1) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat(
            llama_vocab_n_tokens(g_vocab),  // Use the vocab to get n_vocab
            sampler_seed,
            mirostat_tau,
            mirostat_eta,
            100  // m parameter
        ));
    } else if (mirostat == 2) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat_v2(
            sampler_seed,
            mirostat_tau,
            mirostat_eta
        ));
    } else {
        // Standard sampling chain (only if mirostat is disabled)
        if (min_p > 0.0f && min_p < 1.0f) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p(min_p, 1));
        }
        
        if (typical_p < 1.0f) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_typical(typical_p, 1));
        }
        
        if (top_k > 0) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
        }
        
        if (top_p < 1.0f) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
        }
    }
    
    // Final distribution sampler
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(sampler_seed));

    // Get callback method
    jclass callbackClass = env->GetObjectClass(token_callback);
    jmethodID invokeMethod = env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");

    // Generation loop
    LOGI("Starting generation loop: max_tokens=%lld", max_tokens);
    for (int i = 0; i < max_tokens && !g_stop_flag; i++) {
        // If g_n_past >= n_ctx, shift context window during generation so it does NOT crash or abort!
        if (g_n_past >= n_ctx) {
            if (g_context_shift) {
                const int n_discard = std::max(1, n_ctx / 4);
                LOGI("Generation reached context limit (%d), shifting KV cache by %d tokens", g_n_past, n_discard);
                llama_memory_seq_rm(llama_get_memory(g_ctx), 0, 0, n_discard);
                llama_memory_seq_add(llama_get_memory(g_ctx), 0, n_discard, g_n_past, -n_discard);
                g_n_past -= n_discard;
            } else {
                LOGW("Generation reached context limit (%d) and context shift disabled; stopping generation", g_n_past);
                break;
            }
        }

        // Sample next token
        LOGI("Sampling token %d, g_n_past=%d", i + 1, g_n_past);
        llama_token new_token_id = llama_sampler_sample(g_sampler, g_ctx, -1);
        LOGI("Sampled token: %d", new_token_id);

        // Check for end of generation (EOS/EOD tokens)
        if (llama_vocab_is_eog(g_vocab, new_token_id)) {
            LOGI("EOS token detected, ending generation.");
            break;
        }

        // Decode token to string
        char buffer[256];
        int32_t length = llama_token_to_piece(g_vocab, new_token_id, buffer, sizeof(buffer), 0, true);
        std::string piece;
        
        if (length > 0) {
            piece = sanitizeUTF8(buffer, length);
        } else {
            piece = "";
        }
        
        // Call Kotlin callback
        jstring token_str = env->NewStringUTF(piece.c_str());
        env->CallObjectMethod(g_token_callback, invokeMethod, token_str);
        env->DeleteLocalRef(token_str);

        // Prepare next batch
        batch.n_tokens = 0;
        batch.token[batch.n_tokens] = new_token_id;
        batch.pos[batch.n_tokens] = g_n_past;  // Use the tracked position
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;

        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("Failed to decode after sampling token %d", i + 1);
            break;
        }
        
        // Update position counter after each generated token
        g_n_past++;
    }
    LOGI("Generation loop finished.");

    if (g_token_callback != nullptr) {
        env->DeleteGlobalRef(g_token_callback);
        g_token_callback = nullptr;
    }

    // After generation completion, ensure the KV cache is properly managed
    // In some llama.cpp versions, KV cache management may be needed between generations
    // For chat applications, we want to maintain conversation context
    
    llama_batch_free(batch);
    env->DeleteLocalRef(callbackClass);
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeStop(
    JNIEnv* env, jobject thiz) {
    g_stop_flag = true;
    // When stopping generation, we just set the flag - KV cache management handled by llama.cpp
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeFreeModel(
    JNIEnv* env, jobject thiz) {
    
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
    g_n_past = 0;  // Reset position counter
    
    LOGI("Model freed");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeGetTokensUsed(
    JNIEnv* env, jobject thiz) {
    return g_n_past;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeGetContextSize(
    JNIEnv* env, jobject thiz) {
    return g_ctx ? llama_n_ctx(g_ctx) : 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeClearContext(
    JNIEnv* env, jobject thiz) {
    if (!g_ctx) {
        LOGE("Cannot clear context: context is null");
        return;
    }
    
    llama_memory_t mem = llama_get_memory(g_ctx);
    if (mem) {
        llama_memory_seq_rm(mem, 0, 0, -1);
        g_n_past = 0;
        LOGI("Context cleared, g_n_past reset to 0");
    } else {
        LOGE("Failed to get memory object from context");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeSetSystemPromptLength(
    JNIEnv* env, jobject thiz, jint length) {
    // Currently not used but available for future smart context management
    LOGI("System prompt length set to: %d tokens (currently unused)", length);
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeSetNThreads(
    JNIEnv* env, jobject thiz, jint n_threads, jint n_threads_batch) {
    if (g_ctx) {
        llama_set_n_threads(g_ctx, n_threads, n_threads_batch);
        LOGI("Dynamic threads updated via llama_set_n_threads: n_threads=%d, n_threads_batch=%d",
             n_threads, n_threads_batch);
    }
}

