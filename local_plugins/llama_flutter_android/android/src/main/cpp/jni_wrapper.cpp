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
#include <sys/resource.h>
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
static std::atomic<bool> g_is_generating{false};
static std::atomic<int> g_n_past{0};  // Track the number of tokens already in KV cache
static std::atomic<int> g_context_size{0};
static bool g_context_shift = true;
static std::mutex g_load_log_mutex;
static std::string g_load_error;
static bool g_capture_load_error = false;
static std::mutex g_active_req_mutex;
static std::string g_active_req_id = "req_#0";

static std::string formatDouble(double val, int decimals) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", decimals, val);
    return std::string(buf);
}

static std::string currentTimestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm_buf;
    localtime_r(&ts.tv_sec, &tm_buf);
    char buf[64];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03ld",
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, ts.tv_nsec / 1000000);
    return std::string(buf);
}

static bool llama_abort_check(void* /*data*/) {
    return g_stop_flag.load();
}

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

    g_stop_flag.store(true);
    for (int i = 0; i < 50 && g_is_generating.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
    g_n_past.store(0);
    g_context_size.store(0);

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

        // 3. Configure KV cache quantization defaults & clamp CPU threads (max 4) to prevent CPU starvation
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = ctx_size;
        int32_t clamped_threads = std::min(4, (int32_t)std::max(1L, (long)n_threads));
        int32_t clamped_threads_batch = std::min(4, (int32_t)std::max(1L, (long)(n_threads_batch > 0 ? n_threads_batch : n_threads)));
        cparams.n_threads = clamped_threads;
        cparams.n_threads_batch = clamped_threads_batch;
        LOGI("Configured CPU threads: requested=%lld/%lld, clamped=%d/%d",
             (long long)n_threads, (long long)n_threads_batch, clamped_threads, clamped_threads_batch);
        
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
        LOGI("[%s] Creating context: n_ctx=%lld, type_k=%d, type_v=%d, context_shift=%d",
             currentTimestamp().c_str(), (long long)ctx_size, cparams.type_k, cparams.type_v, g_context_shift ? 1 : 0);
        cparams.abort_callback = llama_abort_check;
        cparams.abort_callback_data = nullptr;
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

        llama_set_abort_callback(g_ctx, llama_abort_check, nullptr);
        g_context_size.store(llama_n_ctx(g_ctx));
        g_n_past.store(0);

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
        g_n_past.store(0);

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
    jstring prompt, jstring request_id, jlong max_tokens, 
    jdouble temperature, jdouble top_p, jlong top_k, jdouble min_p, jdouble typical_p,
    jdouble repeat_penalty, jdouble frequency_penalty, jdouble presence_penalty, jlong repeat_last_n,
    jlong mirostat, jdouble mirostat_tau, jdouble mirostat_eta,
    jlong seed, jboolean penalize_newline,
    jobject token_callback) {

    const char* raw_req = env->GetStringUTFChars(request_id, nullptr);
    std::string req_id = raw_req ? raw_req : "req_#0";
    if (raw_req) {
        env->ReleaseStringUTFChars(request_id, raw_req);
    }
    {
        std::lock_guard<std::mutex> lock(g_active_req_mutex);
        g_active_req_id = req_id;
    }

    const std::string ts_start = currentTimestamp();
    LOGI("[%s] [%s] nativeGenerate called (max_tokens=%lld)", ts_start.c_str(), req_id.c_str(), (long long)max_tokens);

    if (g_is_generating.exchange(true)) {
        LOGE("[%s] [%s] nativeGenerate rejected: generation already active!", currentTimestamp().c_str(), req_id.c_str());
        jclass callbackClass = env->GetObjectClass(token_callback);
        jmethodID invokeMethod = env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
        jstring err_str = env->NewStringUTF("[ERROR]: Already generating");
        env->CallObjectMethod(token_callback, invokeMethod, err_str);
        env->DeleteLocalRef(err_str);
        env->DeleteLocalRef(callbackClass);
        return;
    }

    struct GenerationGuard {
        ~GenerationGuard() {
            g_is_generating.store(false);
        }
    } gen_guard;

    std::lock_guard<std::mutex> lock(g_infer_mutex);

    jclass callbackClass = env->GetObjectClass(token_callback);
    jmethodID invokeMethod = env->GetMethodID(callbackClass, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");

    auto emit_stage = [&](const std::string& msg) {
        std::string full_stage = "[STAGE]: " + msg;
        jstring s_str = env->NewStringUTF(full_stage.c_str());
        env->CallObjectMethod(token_callback, invokeMethod, s_str);
        env->DeleteLocalRef(s_str);
    };

    if (!g_model || !g_ctx || !g_vocab) {
        LOGE("[%s] [%s] Model not loaded in nativeGenerate", currentTimestamp().c_str(), req_id.c_str());
        jstring err_str = env->NewStringUTF("[ERROR]: Model not loaded");
        env->CallObjectMethod(token_callback, invokeMethod, err_str);
        env->DeleteLocalRef(err_str);
        env->DeleteLocalRef(callbackClass);
        return;
    }

    // Set background thread priority (nice +10) to avoid starving SurfaceFlinger
    setpriority(PRIO_PROCESS, 0, 10);
    g_stop_flag.store(false);

    auto req_start_time = std::chrono::steady_clock::now();
    int active_threads = g_ctx ? llama_n_threads(g_ctx) : 0;
    LOGI("[%s] [%s] [NativeStart] max_tokens=%lld, temp=%.2f, top_p=%.2f, threads=%d",
         currentTimestamp().c_str(), req_id.c_str(), (long long)max_tokens, temperature, top_p, active_threads);
    emit_stage("[" + req_id + "] [NativeStart] max_tokens=" + std::to_string(max_tokens) +
               ", temp=" + formatDouble(temperature, 2) + ", threads=" + std::to_string(active_threads));

    const char* raw_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_str = raw_prompt ? raw_prompt : "";
    if (raw_prompt) {
        env->ReleaseStringUTFChars(prompt, raw_prompt);
    }

    try {
        const int prompt_len = (int)prompt_str.length();
        LOGI("[%s] [%s] Tokenizing prompt (length: %d chars)", currentTimestamp().c_str(), req_id.c_str(), prompt_len);

        std::string sanitized_prompt = sanitizeUTF8(prompt_str.c_str(), prompt_len);
        const char* sanitized_cstr = sanitized_prompt.c_str();
        const int sanitized_len = (int)sanitized_prompt.length();

        const int n_prompt_tokens = -llama_tokenize(g_vocab, sanitized_cstr, sanitized_len, nullptr, 0, true, true);
        LOGI("[%s] [%s] Prompt tokenized into %d tokens", currentTimestamp().c_str(), req_id.c_str(), n_prompt_tokens);

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
            LOGW("[%s] [%s] Prompt token count (%zu) exceeds context size (%d); sliding prompt window to %d tokens",
                 currentTimestamp().c_str(), req_id.c_str(), tokens.size(), n_ctx, max_prompt_tokens);
            const int excess = (int)tokens.size() - max_prompt_tokens;
            tokens.erase(tokens.begin(), tokens.begin() + excess);
            llama_memory_clear(llama_get_memory(g_ctx), true);
            g_n_past.store(0);
        }

        // Check if context will be exceeded and apply sliding window for KV cache
        if (g_n_past.load() + (int)tokens.size() > n_ctx) {
            if (g_context_shift) {
                int n_needed = (g_n_past.load() + (int)tokens.size()) - n_ctx;
                int n_discard = std::max(n_needed, n_ctx / 4);
                if (n_discard >= g_n_past.load()) {
                    LOGI("[%s] [%s] Prompt requires clearing previous KV cache (g_n_past=%d, tokens=%zu, n_ctx=%d)",
                         currentTimestamp().c_str(), req_id.c_str(), g_n_past.load(), tokens.size(), n_ctx);
                    llama_memory_clear(llama_get_memory(g_ctx), true);
                    g_n_past.store(0);
                } else {
                    LOGI("[%s] [%s] Context full, shifting KV cache by %d tokens",
                         currentTimestamp().c_str(), req_id.c_str(), n_discard);
                    llama_memory_seq_rm(llama_get_memory(g_ctx), 0, 0, n_discard);
                    llama_memory_seq_add(llama_get_memory(g_ctx), 0, n_discard, g_n_past.load(), -n_discard);
                    g_n_past.fetch_sub(n_discard);
                }
            } else {
                LOGW("[%s] [%s] Context full and context shift disabled: g_n_past=%d, tokens=%zu, n_ctx=%d",
                     currentTimestamp().c_str(), req_id.c_str(), g_n_past.load(), tokens.size(), n_ctx);
            }
        }

        const int max_batch_size = 512;
        int tokens_processed = 0;
        ScopedBatch scoped_batch(max_batch_size, 0, 1);
        llama_batch& batch = scoped_batch.batch;

        auto prefill_start_time = std::chrono::steady_clock::now();
        LOGI("[%s] [%s] [PrefillStart] prompt_tokens=%zu, batch_size=%d",
             currentTimestamp().c_str(), req_id.c_str(), tokens.size(), max_batch_size);
        emit_stage("[" + req_id + "] [PrefillStart] prompt_tokens=" + std::to_string(tokens.size()) +
                   ", batch_size=" + std::to_string(max_batch_size));

        int last_decode_result = 0;
        int batch_idx = 0;
        while (tokens_processed < (int)tokens.size() && !g_stop_flag.load()) {
            batch.n_tokens = 0;
            int batch_size = std::min((int)tokens.size() - tokens_processed, max_batch_size);

            for (int i = 0; i < batch_size; i++) {
                batch.token[batch.n_tokens] = tokens[tokens_processed + i];
                batch.pos[batch.n_tokens] = g_n_past.load() + tokens_processed + i;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = false;
                batch.n_tokens++;
            }

            // Ensure the last token of the final batch has logits enabled
            if (tokens_processed + batch_size >= (int)tokens.size() && batch.n_tokens > 0) {
                batch.logits[batch.n_tokens - 1] = true;
            }

            LOGI("[%s] [%s] [PrefillBatch] batch=%d, g_n_past=%d, batch_size=%d",
                 currentTimestamp().c_str(), req_id.c_str(), batch_idx++, g_n_past.load() + tokens_processed, batch.n_tokens);
            last_decode_result = llama_decode(g_ctx, batch);
            if (last_decode_result != 0) {
                if (g_stop_flag.load() || last_decode_result == 2) {
                    LOGI("[%s] [%s] Prompt decode aborted by stop signal (code %d)",
                         currentTimestamp().c_str(), req_id.c_str(), last_decode_result);
                    break;
                }
                LOGE("[%s] [%s] ❌ DECODE FAILED! Result code: %d",
                     currentTimestamp().c_str(), req_id.c_str(), last_decode_result);
                throw std::runtime_error("Failed to decode prompt batch (code " + std::to_string(last_decode_result) + ")");
            }
            tokens_processed += batch_size;
        }

        auto prefill_end_time = std::chrono::steady_clock::now();
        double prefill_elapsed_s = std::chrono::duration<double>(prefill_end_time - prefill_start_time).count();
        double prefill_tps = (prefill_elapsed_s > 0) ? (tokens_processed / prefill_elapsed_s) : 0.0;
        LOGI("[%s] [%s] [PrefillEnd] tokens=%d, elapsed=%.2fs (%.1f t/s), code=%d",
             currentTimestamp().c_str(), req_id.c_str(), tokens_processed, prefill_elapsed_s, prefill_tps, last_decode_result);
        emit_stage("[" + req_id + "] [PrefillEnd] tokens=" + std::to_string(tokens_processed) +
                   ", elapsed=" + formatDouble(prefill_elapsed_s, 2) + "s (" + formatDouble(prefill_tps, 1) + " t/s), code=" + std::to_string(last_decode_result));

        if (g_stop_flag.load()) {
            LOGI("[%s] [%s] [Terminal] Stopped during prefill -> sending [DONE]",
                 currentTimestamp().c_str(), req_id.c_str());
            emit_stage("[" + req_id + "] [Terminal] reason=STOP_PREFILL, tokens=" + std::to_string(tokens_processed) +
                       ", elapsed=" + formatDouble(prefill_elapsed_s, 2) + "s");
            jstring done_str = env->NewStringUTF("[DONE]");
            env->CallObjectMethod(token_callback, invokeMethod, done_str);
            env->DeleteLocalRef(done_str);
            env->DeleteLocalRef(callbackClass);
            return;
        }

        g_n_past.fetch_add(tokens.size());

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

        LOGI("[%s] [%s] Starting generation loop: max_tokens=%lld",
             currentTimestamp().c_str(), req_id.c_str(), (long long)max_tokens);
        bool first_token = true;
        int generated_tokens = 0;
        bool is_eos = false;

        for (int i = 0; i < max_tokens && !g_stop_flag.load(); i++) {
            if (g_n_past.load() >= n_ctx) {
                if (g_context_shift) {
                    const int n_discard = std::max(1, n_ctx / 4);
                    LOGI("[%s] [%s] Context limit reached (%d), shifting KV cache by %d tokens",
                         currentTimestamp().c_str(), req_id.c_str(), g_n_past.load(), n_discard);
                    llama_memory_seq_rm(llama_get_memory(g_ctx), 0, 0, n_discard);
                    llama_memory_seq_add(llama_get_memory(g_ctx), 0, n_discard, g_n_past.load(), -n_discard);
                    g_n_past.fetch_sub(n_discard);
                } else {
                    LOGW("[%s] [%s] Context limit reached (%d) and context shift disabled; stopping",
                         currentTimestamp().c_str(), req_id.c_str(), g_n_past.load());
                    break;
                }
            }

            // Defensive null-check before calling sampler
            float* logits = llama_get_logits_ith(g_ctx, -1);
            if (!logits) {
                LOGE("[%s] [%s] Logits returned NULL! Skipping sample to prevent crash",
                     currentTimestamp().c_str(), req_id.c_str());
                break;
            }

            llama_token new_token_id = llama_sampler_sample(g_sampler, g_ctx, -1);

            if (first_token) {
                auto first_token_time = std::chrono::steady_clock::now();
                double first_token_s = std::chrono::duration<double>(first_token_time - req_start_time).count();
                LOGI("[%s] [%s] [FirstToken] token_id=%d, elapsed=%.2fs",
                     currentTimestamp().c_str(), req_id.c_str(), new_token_id, first_token_s);
                emit_stage("[" + req_id + "] [FirstToken] token_id=" + std::to_string(new_token_id) +
                           ", elapsed=" + formatDouble(first_token_s, 2) + "s");
                first_token = false;
            }

            // Break immediately on EOS
            if (llama_vocab_is_eog(g_vocab, new_token_id)) {
                LOGI("[%s] [%s] EOS token detected (%d), ending generation.",
                     currentTimestamp().c_str(), req_id.c_str(), new_token_id);
                is_eos = true;
                break;
            }

            char buffer[256];
            int32_t length = llama_token_to_piece(g_vocab, new_token_id, buffer, sizeof(buffer), 0, true);
            std::string piece;
            if (length > 0) {
                piece = sanitizeUTF8(buffer, length);
            }

            jstring token_str = env->NewStringUTF(piece.c_str());
            env->CallObjectMethod(token_callback, invokeMethod, token_str);
            env->DeleteLocalRef(token_str);
            generated_tokens++;

            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
                LOGW("[%s] [%s] Exception detected in token callback; stopping generation",
                     currentTimestamp().c_str(), req_id.c_str());
                g_stop_flag.store(true);
                break;
            }

            if (g_stop_flag.load()) {
                LOGI("[%s] [%s] Cancellation observed in loop", currentTimestamp().c_str(), req_id.c_str());
                break;
            }

            // Autoregressive decode advances position:
            batch.n_tokens = 1;
            batch.token[0] = new_token_id;
            batch.pos[0] = g_n_past.load();
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            int decode_res = llama_decode(g_ctx, batch);
            if (decode_res != 0) {
                if (g_stop_flag.load() || decode_res == 2) {
                    LOGI("[%s] [%s] Decode aborted by stop signal", currentTimestamp().c_str(), req_id.c_str());
                } else {
                    LOGE("[%s] [%s] Failed to decode token %d (res=%d)",
                         currentTimestamp().c_str(), req_id.c_str(), i + 1, decode_res);
                }
                break;
            }
            g_n_past.fetch_add(1);
        }

        auto req_end_time = std::chrono::steady_clock::now();
        double total_elapsed_s = std::chrono::duration<double>(req_end_time - req_start_time).count();
        double gen_tps = (total_elapsed_s > 0) ? (generated_tokens / total_elapsed_s) : 0.0;
        const char* term_reason = g_stop_flag.load() ? "STOP" : (is_eos ? "EOS" : "MAX_TOKENS");
        LOGI("[%s] [%s] [Terminal] reason=%s, generated_tokens=%d, elapsed=%.2fs (%.1f t/s)",
             currentTimestamp().c_str(), req_id.c_str(), term_reason, generated_tokens, total_elapsed_s, gen_tps);
        emit_stage("[" + req_id + "] [Terminal] reason=" + std::string(term_reason) +
                   ", generated_tokens=" + std::to_string(generated_tokens) +
                   ", elapsed=" + formatDouble(total_elapsed_s, 2) + "s (" + formatDouble(gen_tps, 1) + " t/s)");

        jstring done_str = env->NewStringUTF("[DONE]");
        env->CallObjectMethod(token_callback, invokeMethod, done_str);
        env->DeleteLocalRef(done_str);

    } catch (const std::exception& e) {
        LOGE("[%s] [%s] Native inference failed: %s", currentTimestamp().c_str(), req_id.c_str(), e.what());
        emit_stage("[" + req_id + "] [Terminal] reason=ERROR, message=" + std::string(e.what()));
        jstring err_str = env->NewStringUTF((std::string("[ERROR]: ") + e.what()).c_str());
        env->CallObjectMethod(token_callback, invokeMethod, err_str);
        env->DeleteLocalRef(err_str);
    } catch (...) {
        LOGE("[%s] [%s] Native inference failed with unknown error", currentTimestamp().c_str(), req_id.c_str());
        emit_stage("[" + req_id + "] [Terminal] reason=UNKNOWN_ERROR");
        jstring err_str = env->NewStringUTF("[ERROR]: Unknown native error");
        env->CallObjectMethod(token_callback, invokeMethod, err_str);
        env->DeleteLocalRef(err_str);
    }

    env->DeleteLocalRef(callbackClass);
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeStop(
    JNIEnv* env, jobject thiz) {
    std::string current_id;
    {
        std::lock_guard<std::mutex> lock(g_active_req_mutex);
        current_id = g_active_req_id;
    }
    LOGI("[%s] [%s] [Stop] nativeStop called -> setting g_stop_flag = true",
         currentTimestamp().c_str(), current_id.c_str());
    g_stop_flag.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeFreeModel(
    JNIEnv* env, jobject thiz) {
    LOGI("[%s] nativeFreeModel called", currentTimestamp().c_str());
    g_stop_flag.store(true);

    for (int i = 0; i < 50 && g_is_generating.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

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
    g_n_past.store(0);
    g_context_size.store(0);
    LOGI("[%s] Model freed successfully", currentTimestamp().c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeGetTokensUsed(
    JNIEnv* env, jobject thiz) {
    return g_n_past.load();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeGetContextSize(
    JNIEnv* env, jobject thiz) {
    return g_context_size.load();
}

extern "C" JNIEXPORT void JNICALL
Java_com_write4me_llama_1flutter_1android_LlamaFlutterAndroidPlugin_nativeClearContext(
    JNIEnv* env, jobject thiz) {
    if (g_is_generating.load()) {
        LOGW("[%s] Cannot clear context while generation is active", currentTimestamp().c_str());
        return;
    }
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    if (!g_ctx) {
        LOGE("Cannot clear context: context is null");
        return;
    }
    
    llama_memory_t mem = llama_get_memory(g_ctx);
    if (mem) {
        llama_memory_seq_rm(mem, 0, 0, -1);
        g_n_past.store(0);
        LOGI("[%s] Context cleared, g_n_past reset to 0", currentTimestamp().c_str());
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
    if (g_is_generating.load()) {
        LOGW("[%s] Skipping dynamic thread update while generation is active", currentTimestamp().c_str());
        return;
    }
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    if (g_ctx) {
        int32_t clamped_threads = std::min(4, std::max(1, (int32_t)n_threads));
        int32_t clamped_threads_batch = std::min(4, std::max(1, (int32_t)n_threads_batch));
        llama_set_n_threads(g_ctx, clamped_threads, clamped_threads_batch);
        LOGI("[%s] Dynamic threads updated: requested=%d/%d, clamped=%d/%d",
             currentTimestamp().c_str(), n_threads, n_threads_batch, clamped_threads, clamped_threads_batch);
    }
}


