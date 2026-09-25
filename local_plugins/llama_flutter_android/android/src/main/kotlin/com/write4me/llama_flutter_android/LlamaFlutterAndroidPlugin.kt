package com.write4me.llama_flutter_android

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import io.flutter.embedding.engine.plugins.FlutterPlugin
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

class LlamaFlutterAndroidPlugin : FlutterPlugin, LlamaHostApi {
    private lateinit var context: Context
    private lateinit var flutterApi: LlamaFlutterApi
    private val mainHandler = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var generationJob: Job? = null
    private val isModelLoaded = AtomicBoolean(false)
    private val isGenerating = AtomicBoolean(false)
    private val isStopping = AtomicBoolean(false)
    private var currentModelPath: String? = null
    private var nativeLoadError: Throwable? = null

    private fun timestamp(): String {
        return SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(Date())
    }

    companion object {
        private const val TAG = "LlamaFlutterPlugin"
    }

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        Log.i(TAG, "Attaching llama Flutter plugin")
        context = binding.applicationContext
        flutterApi = LlamaFlutterApi(binding.binaryMessenger)
        LlamaHostApi.setUp(binding.binaryMessenger, this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        scope.cancel()
        if (isModelLoaded.get()) {
            nativeFreeModel()
        }
        LlamaHostApi.setUp(binding.binaryMessenger, null)
    }

    override fun loadModel(config: ModelConfig, callback: (Result<Unit>) -> Unit) {
        ensureNativeLoaded()?.let {
            callback(Result.failure(it))
            return
        }

        scope.launch {
            try {
                // Start foreground service for long-running task
                val intent = Intent(context, InferenceService::class.java)
                ContextCompat.startForegroundService(context, intent)

                // Load model with progress callback
                nativeLoadModel(
                    config.modelPath,
                    config.nThreads,
                    config.contextSize,
                    config.nGpuLayers ?: 0L,
                    config.kvQuantization ?: "q8_0",
                    config.contextShift ?: true,
                    config.nThreadsBatch ?: config.nThreads,
                    config.nBatch ?: 512L,
                    config.nUbatch ?: config.nBatch ?: 512L
                ) { progress ->
                    scope.launch {
                        withContext(Dispatchers.Main) {
                            flutterApi.onLoadProgress(progress) { result ->
                                // Handle result if needed
                            }
                        }
                    }
                }

                currentModelPath = config.modelPath
                isModelLoaded.set(true)
                withContext(Dispatchers.Main) {
                    callback(Result.success(Unit))
                }
            } catch (e: Exception) {
                scope.launch {
                    withContext(Dispatchers.Main) {
                        flutterApi.onError(e.message ?: "Failed to load model") { result ->
                            // Handle result if needed
                        }
                        callback(Result.failure(e))
                    }
                }
            }
        }
    }

    override fun generate(request: GenerateRequest, callback: (Result<Unit>) -> Unit) {
        ensureNativeLoaded()?.let {
            callback(Result.failure(it))
            return
        }

        if (!isModelLoaded.get()) {
            callback(Result.failure(IllegalStateException("Model not loaded")))
            return
        }

        if (!isGenerating.compareAndSet(false, true)) {
            Log.w(TAG, "[${timestamp()}] Generate rejected: already generating")
            callback(Result.failure(IllegalStateException("Already generating")))
            return
        }

        isStopping.set(false)
        generationJob = scope.launch(Dispatchers.Default) {
            try {
                Log.i(TAG, "[${timestamp()}] Send / nativeGenerate started")
                nativeGenerate(
                    request.prompt,
                    request.maxTokens,
                    request.temperature,
                    request.topP,
                    request.topK,
                    request.minP,
                    request.typicalP,
                    request.repeatPenalty,
                    request.frequencyPenalty,
                    request.presencePenalty,
                    request.repeatLastN,
                    request.mirostat,
                    request.mirostatTau,
                    request.mirostatEta,
                    request.seed ?: -1L,  // Use -1 for random seed
                    request.penalizeNewline
                ) { token ->
                    val isTerminal = token == "[DONE]" || token.startsWith("[ERROR]: ")
                    val stopped = isStopping.get()
                    if (!stopped || isTerminal) {
                        mainHandler.post {
                            if (token == "[DONE]") {
                                Log.i(TAG, "[${timestamp()}] [DONE] callback dispatched to Flutter")
                                flutterApi.onDone { }
                            } else if (token.startsWith("[ERROR]: ")) {
                                val err = token.removePrefix("[ERROR]: ")
                                Log.e(TAG, "[${timestamp()}] [ERROR] callback dispatched to Flutter: $err")
                                flutterApi.onError(err) { }
                            } else {
                                flutterApi.onToken(token) { }
                            }
                        }
                    }
                }

                withContext(Dispatchers.Main) {
                    Log.i(TAG, "[${timestamp()}] nativeGenerate finished successfully")
                    callback(Result.success(Unit))
                }
            } catch (e: Exception) {
                Log.e(TAG, "[${timestamp()}] Generation exception: ${e.message}", e)
                mainHandler.post {
                    flutterApi.onError(e.message ?: "Generation failed") { }
                }
                withContext(Dispatchers.Main) {
                    callback(Result.failure(e))
                }
            } finally {
                isGenerating.set(false)
            }
        }
    }

    override fun stop(callback: (Result<Unit>) -> Unit) {
        Log.i(TAG, "[${timestamp()}] Stop received from Flutter")
        isStopping.set(true)
        if (nativeLoadError == null) {
            nativeStop()
        }
        callback(Result.success(Unit))
    }

    override fun dispose(callback: (Result<Unit>) -> Unit) {
        Log.i(TAG, "[${timestamp()}] dispose() called")
        scope.launch(Dispatchers.Default) {
            try {
                stop { }
                if (isModelLoaded.get()) {
                    nativeFreeModel()
                    isModelLoaded.set(false)
                }
                
                // Stop foreground service
                val intent = Intent(context, InferenceService::class.java)
                context.stopService(intent)
                
                withContext(Dispatchers.Main) {
                    callback(Result.success(Unit))
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback(Result.failure(e))
                }
            }
        }
    }

    override fun generateChat(request: ChatRequest, callback: (Result<Unit>) -> Unit) {
        ensureNativeLoaded()?.let {
            callback(Result.failure(it))
            return
        }

        if (!isModelLoaded.get()) {
            callback(Result.failure(IllegalStateException("Model not loaded")))
            return
        }

        if (!isGenerating.compareAndSet(false, true)) {
            Log.w(TAG, "[${timestamp()}] GenerateChat rejected: already generating")
            callback(Result.failure(IllegalStateException("Already generating")))
            return
        }

        isStopping.set(false)
        generationJob = scope.launch(Dispatchers.Default) {
            try {
                Log.i(TAG, "[${timestamp()}] Send / nativeGenerateChat started")
                // Format the chat messages using the template manager
                val formattedPrompt = ChatTemplateManager.formatMessages(
                    request.messages.map { msg -> TemplateChatMessage(msg.role, msg.content) },
                    request.template,
                    currentModelPath
                )

                nativeGenerate(
                    formattedPrompt,
                    request.maxTokens.toLong(),
                    request.temperature.toDouble(),
                    request.topP.toDouble(),
                    request.topK.toLong(),
                    request.minP.toDouble(),
                    request.typicalP.toDouble(),
                    request.repeatPenalty.toDouble(),
                    request.frequencyPenalty.toDouble(),
                    request.presencePenalty.toDouble(),
                    request.repeatLastN.toLong(),
                    request.mirostat.toLong(),
                    request.mirostatTau.toDouble(),
                    request.mirostatEta.toDouble(),
                    request.seed ?: -1L,  // Use -1 for random seed
                    request.penalizeNewline
                ) { token ->
                    val isTerminal = token == "[DONE]" || token.startsWith("[ERROR]: ")
                    val stopped = isStopping.get()
                    if (!stopped || isTerminal) {
                        mainHandler.post {
                            if (token == "[DONE]") {
                                Log.i(TAG, "[${timestamp()}] [DONE] callback dispatched to Flutter")
                                flutterApi.onDone { }
                            } else if (token.startsWith("[ERROR]: ")) {
                                val err = token.removePrefix("[ERROR]: ")
                                Log.e(TAG, "[${timestamp()}] [ERROR] callback dispatched to Flutter: $err")
                                flutterApi.onError(err) { }
                            } else {
                                flutterApi.onToken(token) { }
                            }
                        }
                    }
                }

                withContext(Dispatchers.Main) {
                    Log.i(TAG, "[${timestamp()}] nativeGenerateChat finished successfully")
                    callback(Result.success(Unit))
                }
            } catch (e: Exception) {
                Log.e(TAG, "[${timestamp()}] Generation exception: ${e.message}", e)
                mainHandler.post {
                    flutterApi.onError(e.message ?: "Generation failed") { }
                }
                withContext(Dispatchers.Main) {
                    callback(Result.failure(e))
                }
            } finally {
                isGenerating.set(false)
            }
        }
    }

    override fun getSupportedTemplates(): List<String> {
        return ChatTemplateManager.getSupportedTemplates()
    }

    override fun isModelLoaded(): Boolean {
        return isModelLoaded.get()
    }

    override fun getContextInfo(): ContextInfo {
        ensureNativeLoaded()?.let { throw it }

        val tokensUsed = nativeGetTokensUsed().toLong()
        val contextSize = nativeGetContextSize().toLong()
        val usagePercentage = if (contextSize > 0) {
            (tokensUsed.toDouble() / contextSize.toDouble() * 100.0)
        } else {
            0.0
        }
        
        return ContextInfo(
            tokensUsed = tokensUsed,
            contextSize = contextSize,
            usagePercentage = usagePercentage
        )
    }

    override fun clearContext(callback: (Result<Unit>) -> Unit) {
        ensureNativeLoaded()?.let {
            callback(Result.failure(it))
            return
        }

        scope.launch {
            try {
                nativeClearContext()
                withContext(Dispatchers.Main) {
                    callback(Result.success(Unit))
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback(Result.failure(e))
                }
            }
        }
    }

    override fun setSystemPromptLength(length: Long) {
        ensureNativeLoaded()?.let { throw it }
        nativeSetSystemPromptLength(length.toInt())
    }

    /**
     * Register a custom chat template
     * Allows users to provide their own template format at runtime
     */
    override fun registerCustomTemplate(name: String, content: String) {
        ChatTemplateManager.registerCustomTemplate(name, content)
    }

    /**
     * Unregister a custom chat template
     * Removes a previously registered custom template
     */
    override fun unregisterCustomTemplate(name: String) {
        ChatTemplateManager.unregisterCustomTemplate(name)
    }

    override fun detectGpu(callback: (Result<GpuInfo>) -> Unit) {
        ensureNativeLoaded()?.let {
            callback(Result.failure(it))
            return
        }

        scope.launch {
            try {
                val outStats = LongArray(2) { -1L }
                val gpuName: String? = nativeDetectGpu(outStats)
                val vulkanSupported = gpuName != null
                val apiVersion = outStats[0]
                val deviceLocalMemory = outStats[1]

                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memInfo)
                val freeRamBytes = memInfo.availMem

                val recommendedGpuLayers = computeRecommendedLayers(
                    vulkanSupported = vulkanSupported,
                    gpuName = gpuName ?: "",
                    freeRamBytes = freeRamBytes,
                    deviceLocalMemoryBytes = deviceLocalMemory
                )

                withContext(Dispatchers.Main) {
                    callback(Result.success(GpuInfo(
                        vulkanSupported = vulkanSupported,
                        gpuName = gpuName ?: "None",
                        vulkanApiVersion = apiVersion,
                        deviceLocalMemoryBytes = deviceLocalMemory,
                        freeRamBytes = freeRamBytes,
                        recommendedGpuLayers = recommendedGpuLayers.toLong()
                    )))
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback(Result.success(GpuInfo(
                        vulkanSupported = false,
                        gpuName = "None",
                        vulkanApiVersion = -1L,
                        deviceLocalMemoryBytes = -1L,
                        freeRamBytes = -1L,
                        recommendedGpuLayers = 0L
                    )))
                }
            }
        }
    }

    override fun setNThreads(threads: Long, batchThreads: Long, callback: (Result<Unit>) -> Unit) {
        ensureNativeLoaded()?.let {
            callback(Result.failure(it))
            return
        }
        if (!isModelLoaded.get()) {
            callback(Result.success(Unit))
            return
        }
        try {
            nativeSetNThreads(threads.toInt(), batchThreads.toInt())
            callback(Result.success(Unit))
        } catch (e: Exception) {
            callback(Result.failure(e))
        }
    }

    private fun ensureNativeLoaded(): Throwable? {
        nativeLoadError?.let { return it }

        return try {
            System.loadLibrary("llama_jni")
            null
        } catch (t: Throwable) {
            nativeLoadError = t
            Log.e(TAG, "Failed to load llama native library", t)
            IllegalStateException("Failed to load llama native library: ${t.message}", t)
        }
    }

    private fun computeRecommendedLayers(
        vulkanSupported: Boolean,
        gpuName: String,
        freeRamBytes: Long,
        deviceLocalMemoryBytes: Long
    ): Int {
        val GB = 1_073_741_824L
        val safeRam = (freeRamBytes * 0.7).toLong()
        return when {
            !vulkanSupported -> 0
            gpuName.contains("Mali", ignoreCase = true) -> 0
            safeRam < GB -> 0                                          // < 1 GB — truly too low
            safeRam < 2 * GB && deviceLocalMemoryBytes < 3 * GB -> 0  // low RAM + low VRAM
            safeRam < 3 * GB || deviceLocalMemoryBytes < 2 * GB -> 16 // partial offload
            else -> 99                                                  // full offload
        }
    }

    // Native methods
    private external fun nativeLoadModel(
        path: String,
        nThreads: Long,
        contextSize: Long,
        nGpuLayers: Long,
        kvQuantization: String,
        contextShift: Boolean,
        nThreadsBatch: Long,
        nBatch: Long,
        nUbatch: Long,
        progressCallback: (Double) -> Unit
    )

    private external fun nativeSetNThreads(nThreads: Int, nThreadsBatch: Int)

    private external fun nativeGenerate(
        prompt: String,
        maxTokens: Long,
        temperature: Double,
        topP: Double,
        topK: Long,
        minP: Double,
        typicalP: Double,
        repeatPenalty: Double,
        frequencyPenalty: Double,
        presencePenalty: Double,
        repeatLastN: Long,
        mirostat: Long,
        mirostatTau: Double,
        mirostatEta: Double,
        seed: Long,
        penalizeNewline: Boolean,
        tokenCallback: (String) -> Unit
    )

    private external fun nativeStop()
    private external fun nativeFreeModel()
    private external fun nativeGetTokensUsed(): Int
    private external fun nativeGetContextSize(): Int
    private external fun nativeClearContext()
    private external fun nativeSetSystemPromptLength(length: Int)
    private external fun nativeDetectGpu(outStats: LongArray): String?
    // outStats[0] = vulkanApiVersion, outStats[1] = deviceLocalMemoryBytes
}
