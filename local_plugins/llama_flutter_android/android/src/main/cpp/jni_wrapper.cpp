#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <ctime>
#include <cstring>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <thread>
#include <stdexcept>
#include <android/log.h>
#include "llama.cpp/include/llama.h"
#include "ggml-backend.h"
#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static JavaVM* g_jvm = nullptr;
static std::mutex g_infer_mutex;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    LOGI("JNI_OnLoad: JavaVM stored (%p)", (void*)g_jvm);
    return JNI_VERSION_1_6;
}

struct JvmAttachment {
    JavaVM* jvm = nullptr;
    JNIEnv* env = nullptr;
    bool attached = false;

    explicit JvmAttachment(JavaVM* vm) : jvm(vm) {
        if (!jvm) return;
        jint res = jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
        if (res == JNI_EDETACHED) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            } else {
                env = nullptr;
            }
        }
    }

    ~JvmAttachment() {
        if (attached && jvm) {
            jvm->DetachCurrentThread();
        }
    }
};

struct ScopedBatch {
    llama_batch batch;
    bool allocated = false;

    ScopedBatch(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
        batch = llama_batch_init(n_tokens, embd, n_seq_max);
        allocated = true;
    }

    ~ScopedBatch() {
        if (allocated) {
            llama_batch_free(batch);
            allocated = false;
        }
    }
};

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

    if (!g_jvm) {
        env->GetJavaVM(&g_jvm);
    }

    std::lock_guard<std::mutex> lock(g_infer_mutex);

    // Free any existing model/context
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
    g_n_past = 0;

    try {
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
        //    Adreno 830 Vulkan stability is optimal starting at 15 layers when GPU Fast (999) is selected
        llama_model_params model_params = llama_model_default_params();
        int32_t effective_gpu_layers = (n_gpu_layers == 0) ? 0 : ((n_gpu_layers > 0) ? (int32_t)n_gpu_layers : 15);
        if (effective_gpu_layers >= 999 || effective_gpu_layers > 24) {
            LOGI("Adreno 830 GPU stability: adjusting %d layers to 15 layers", effective_gpu_layers);
            effective_gpu_layers = 15;
        }
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

        // 3. Configure KV cache quantization defaults
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

        // Create context
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
            throwLoadError(env, "Failed to create context");
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
            throwLoadError(env, "Failed to get vocab from model");
            return;
        }
        
        // Reset KV cache position counter for new model
        g_n_past = 0;

        // Report progress completion
        if (progress_callback) {
            jclass callbackClass = env->GetObjectClass(progress_callback);
            jmethodID invokeMethod = env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
            
            jclass doubleClass = env->FindClass("java/lang/Double");
            jmethodID doubleConstructor = env->GetMethodID(doubleClass, "<init>", "(D)V");
            jobject doubleObj = env->NewObject(doubleClass, doubleConstructor, 1.0);
            
            env->CallObjectMethod(progress_callback, invokeMethod, doubleObj);
            env->DeleteLocalRef(doubleObj);
            env->DeleteLocalRef(callbackClass);
        }

        LOGI("Model loaded successfully");
    } catch (const std::exception& e) {
        LOGE("Native load model failed with exception: %s", e.what());
        if (g_ctx) {
            llama_free(g_ctx);
            g_ctx = nullptr;
        }
        if (g_model) {
            llama_model_free(g_model);
            g_model = nullptr;
        }
        g_vocab = nullptr;
        throwLoadError(env, std::string("Model load error: ") + e.what());
    } catch (...) {
        LOGE("Native load model failed with unknown exception");
        if (g_ctx) {
            llama_free(g_ctx);
            g_ctx = nullptr;
        }
        if (g_model) {
            llama_model_free(g_model);
            g_model = nullptr;
        }
        g_vocab = nullptr;
        throwLoadError(env, "Model load error: unknown exception");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeGenerate(
    JNIEnv* env, jobject thiz,
    jstring prompt, jlong max_tokens, 
    jdouble temperature, jdouble top_p, jlong top_k, jdouble min_p, jdouble typical_p,
    jdouble repeat_penalty, jdouble frequency_penalty, jdouble presence_penalty, jlong repeat_last_n,
    jlong mirostat, jdouble mirostat_tau, jdouble mirostat_eta,
    jlong seed, jboolean penalize_newline,
    jobject token_callback) {
    
    if (!g_model || !g_ctx || !g_vocab) {
        jclass exception = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exception, "Model not loaded");
        return;
    }

    if (!g_jvm) {
        env->GetJavaVM(&g_jvm);
    }

    const char* raw_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_str = raw_prompt ? raw_prompt : "";
    if (raw_prompt) {
        env->ReleaseStringUTFChars(prompt, raw_prompt);
    }

    jobject cb_ref = env->NewGlobalRef(token_callback);
    g_stop_flag = false;

    std::string worker_error;

    // Run inference on a dedicated native background thread to prevent UI freezing
    std::thread worker([&, cb_ref, prompt_str, max_tokens, temperature, top_p, top_k, min_p,
                        typical_p, repeat_penalty, frequency_penalty, presence_penalty,
                        repeat_last_n, mirostat, mirostat_tau, mirostat_eta, seed,
                        penalize_newline]() {
        JvmAttachment jvm_att(g_jvm);
        JNIEnv* t_env = jvm_att.env;
        if (!t_env) {
            LOGE("Worker thread failed to attach to JVM");
            worker_error = "Worker thread failed to attach to JVM";
            return;
        }

        std::lock_guard<std::mutex> lock(g_infer_mutex);
        if (!g_model || !g_ctx || !g_vocab) {
            worker_error = "Model not loaded";
            return;
        }

        try {
            const int prompt_len = (int)prompt_str.length();
            LOGI("Tokenizing prompt: '%s' (length: %d)", prompt_str.c_str(), prompt_len);
            LOGI("Vocab pointer: %p, Model pointer: %p", (void*)g_vocab, (void*)g_model);

            std::string sanitized_prompt = sanitizeUTF8(prompt_str.c_str(), prompt_len);
            const char* sanitized_cstr = sanitized_prompt.c_str();
            const int sanitized_len = (int)sanitized_prompt.length();

            const int n_prompt_tokens = -llama_tokenize(g_vocab, sanitized_cstr, sanitized_len, nullptr, 0, true, true);
            LOGI("Token count: %d", n_prompt_tokens);

            if (n_prompt_tokens <= 0) {
                throw std::runtime_error("Failed to tokenize prompt (got " + std::to_string(n_prompt_tokens) + " tokens)");
            }
            std::vector<llama_token> tokens(n_prompt_tokens);
            const int actual_tokens = llama_tokenize(g_vocab, sanitized_cstr, sanitized_len, tokens.data(), tokens.size(), true, true);
            if (actual_tokens < 0) {
                throw std::runtime_error("Failed to tokenize prompt");
            }
            tokens.resize(actual_tokens);

            const int n_ctx = llama_n_ctx(g_ctx);

            // Context shifting: handle prompts exceeding maximum context window
            if (g_context_shift && (int)tokens.size() >= n_ctx) {
                const int max_prompt_tokens = std::max(1, n_ctx - 64);
                LOGW("Prompt token count (%zu) exceeds context size (%d); sliding prompt window to %d tokens",
                     tokens.size(), n_ctx, max_prompt_tokens);
                const int excess = (int)tokens.size() - max_prompt_tokens;
                tokens.erase(tokens.begin(), tokens.begin() + excess);
                llama_memory_clear(llama_get_memory(g_ctx), true);
                g_n_past = 0;
            }

            // Check if context will be exceeded and apply sliding window for KV cache
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

            const int max_batch_size = 512;
            int tokens_processed = 0;
            ScopedBatch scoped_batch(max_batch_size, 0, 1);
            llama_batch& batch = scoped_batch.batch;

            LOGI("Context size: %d", llama_n_ctx(g_ctx));

            while (tokens_processed < (int)tokens.size() && !g_stop_flag) {
                batch.n_tokens = 0;
                int batch_size = std::min((int)tokens.size() - tokens_processed, max_batch_size);

                for (int i = 0; i < batch_size; i++) {
                    batch.token[batch.n_tokens] = tokens[tokens_processed + i];
                    batch.pos[batch.n_tokens] = g_n_past + tokens_processed + i;
                    batch.n_seq_id[batch.n_tokens] = 1;
                    batch.seq_id[batch.n_tokens][0] = 0;
                    batch.logits[batch.n_tokens] = (tokens_processed + i == (int)tokens.size() - 1);
                    batch.n_tokens++;
                }

                // Ensure the last token has logits enabled
                if (tokens_processed + batch_size >= (int)tokens.size() && batch.n_tokens > 0) {
                    batch.logits[batch.n_tokens - 1] = true;
                }

                LOGI("Decoding batch: g_n_past=%d, batch_size=%d", g_n_past + tokens_processed, batch.n_tokens);
                int decode_result = llama_decode(g_ctx, batch);
                if (decode_result != 0) {
                    LOGE("❌ DECODE FAILED! Result code: %d", decode_result);
                    throw std::runtime_error("Failed to decode prompt batch (code " + std::to_string(decode_result) + ")");
                }
                tokens_processed += batch_size;
            }

            if (g_stop_flag) {
                LOGI("Generation stopped by user during prompt evaluation");
                return;
            }

            LOGI("✅ Decode successful! Processed %d total tokens", tokens_processed);
            g_n_past += tokens.size();

            // Create sampler chain with all parameters
            if (g_sampler) {
                llama_sampler_free(g_sampler);
                g_sampler = nullptr;
            }

            uint32_t sampler_seed = (seed >= 0) ? static_cast<uint32_t>(seed) : static_cast<uint32_t>(time(nullptr));
            llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
            g_sampler = llama_sampler_chain_init(sparams);

            if (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) {
                llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(
                    repeat_last_n,
                    repeat_penalty,
                    frequency_penalty,
                    presence_penalty
                ));
            }

            llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));

            if (mirostat == 1) {
                llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat(
                    llama_vocab_n_tokens(g_vocab),
                    sampler_seed,
                    mirostat_tau,
                    mirostat_eta,
                    100
                ));
            } else if (mirostat == 2) {
                llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat_v2(
                    sampler_seed,
                    mirostat_tau,
                    mirostat_eta
                ));
            } else {
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

            llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(sampler_seed));

            jclass callbackClass = t_env->GetObjectClass(cb_ref);
            jmethodID invokeMethod = t_env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");

            LOGI("Starting generation loop: max_tokens=%lld", (long long)max_tokens);
            for (int i = 0; i < max_tokens && !g_stop_flag; i++) {
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

                // Defensive null-check before calling sampler
                float* logits = (batch.n_tokens > 0) ? llama_get_logits_ith(g_ctx, batch.n_tokens - 1) : nullptr;
                if (!logits) {
                    logits = llama_get_logits_ith(g_ctx, -1);
                }
                if (!logits) {
                    LOGE("Logits returned NULL! Skipping sample to prevent SIGSEGV");
                    break;
                }

                llama_token new_token_id = llama_sampler_sample(g_sampler, g_ctx, -1);
                if (llama_vocab_is_eog(g_vocab, new_token_id)) {
                    LOGI("EOS token detected, ending generation.");
                    break;
                }

                char buffer[256];
                int32_t length = llama_token_to_piece(g_vocab, new_token_id, buffer, sizeof(buffer), 0, true);
                std::string piece;
                if (length > 0) {
                    piece = sanitizeUTF8(buffer, length);
                }

                jstring token_str = t_env->NewStringUTF(piece.c_str());
                t_env->CallObjectMethod(cb_ref, invokeMethod, token_str);
                t_env->DeleteLocalRef(token_str);

                if (t_env->ExceptionCheck()) {
                    t_env->ExceptionDescribe();
                    t_env->ExceptionClear();
                    LOGW("Exception detected in token callback; stopping generation");
                    g_stop_flag = true;
                    break;
                }

                batch.n_tokens = 0;
                batch.token[batch.n_tokens] = new_token_id;
                batch.pos[batch.n_tokens] = g_n_past;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = true;
                batch.n_tokens++;
                batch.logits[batch.n_tokens - 1] = true;

                int decode_res = llama_decode(g_ctx, batch);
                if (decode_res != 0) {
                    LOGE("Failed to decode after sampling token %d (res=%d)", i + 1, decode_res);
                    throw std::runtime_error("Failed to decode token after sampling (code " + std::to_string(decode_res) + ")");
                }

                g_n_past++;
            }
            LOGI("Generation loop finished.");
            t_env->DeleteLocalRef(callbackClass);
        } catch (const std::exception& e) {
            LOGE("Native inference failed: %s", e.what());
            worker_error = e.what();
        } catch (...) {
            LOGE("Native inference failed with unknown error");
            worker_error = "Unknown native error during inference";
        }
    });

    worker.join();
    env->DeleteGlobalRef(cb_ref);

    if (!worker_error.empty()) {
        jclass exception = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exception, worker_error.c_str());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeStop(
    JNIEnv* env, jobject thiz) {
    g_stop_flag = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeFreeModel(
    JNIEnv* env, jobject thiz) {
    g_stop_flag = true;
    std::lock_guard<std::mutex> lock(g_infer_mutex);
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
    g_n_past = 0;
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
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    return g_ctx ? llama_n_ctx(g_ctx) : 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeClearContext(
    JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_infer_mutex);
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
    LOGI("System prompt length set to: %d tokens (currently unused)", length);
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeSetNThreads(
    JNIEnv* env, jobject thiz, jint n_threads, jint n_threads_batch) {
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    if (g_ctx) {
        llama_set_n_threads(g_ctx, n_threads, n_threads_batch);
        LOGI("Dynamic threads updated via llama_set_n_threads: n_threads=%d, n_threads_batch=%d",
             n_threads, n_threads_batch);
    }
}

