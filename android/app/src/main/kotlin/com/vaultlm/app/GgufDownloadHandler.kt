package com.vaultlm.app

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.database.Cursor
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.ConcurrentHashMap

class GgufDownloadHandler(
    private val context: Context,
    messenger: BinaryMessenger
) : MethodChannel.MethodCallHandler, EventChannel.StreamHandler {

    companion object {
        private const val TAG = "GgufDownloadHandler"
        private const val METHOD_CHANNEL_NAME = "com.vaultlm.app/downloader"
        private const val EVENT_CHANNEL_NAME = "com.vaultlm.app/downloader_events"
        private const val POLL_INTERVAL_MS = 500L
    }

    private data class TrackedDownload(
        val downloadId: Long,
        val fileName: String,
        val file: File,
        val url: String
    )

    private val methodChannel = MethodChannel(messenger, METHOD_CHANNEL_NAME)
    private val eventChannel = EventChannel(messenger, EVENT_CHANNEL_NAME)
    private val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private val activeDownloads = ConcurrentHashMap<Long, TrackedDownload>()

    private var eventSink: EventChannel.EventSink? = null
    private var isPolling = false
    private var isReceiverRegistered = false

    private val pollRunnable = object : Runnable {
        override fun run() {
            if (!isPolling) return
            pollProgress()
            if (activeDownloads.isNotEmpty() && isPolling) {
                mainHandler.postDelayed(this, POLL_INTERVAL_MS)
            } else {
                isPolling = false
            }
        }
    }

    private val downloadCompleteReceiver = object : BroadcastReceiver() {
        override fun onReceive(receiverContext: Context?, intent: Intent?) {
            if (intent?.action == DownloadManager.ACTION_DOWNLOAD_COMPLETE) {
                val downloadId = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1L)
                if (downloadId != -1L) {
                    mainHandler.post {
                        handleDownloadComplete(downloadId)
                    }
                }
            }
        }
    }

    init {
        methodChannel.setMethodCallHandler(this)
        eventChannel.setStreamHandler(this)
        registerCompleteReceiver()
    }

    private fun registerCompleteReceiver() {
        if (!isReceiverRegistered) {
            try {
                val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
                ContextCompat.registerReceiver(
                    context,
                    downloadCompleteReceiver,
                    filter,
                    ContextCompat.RECEIVER_EXPORTED
                )
                isReceiverRegistered = true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to register download complete receiver: ${e.message}", e)
            }
        }
    }

    private fun unregisterCompleteReceiver() {
        if (isReceiverRegistered) {
            try {
                context.unregisterReceiver(downloadCompleteReceiver)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to unregister download complete receiver: ${e.message}", e)
            } finally {
                isReceiverRegistered = false
            }
        }
    }

    fun getModelDirectoryFile(): File {
        val modelsDir = File(context.getExternalFilesDir(null), "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }
        return modelsDir
    }

    fun getModelDirectory(): String {
        return getModelDirectoryFile().absolutePath
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "getModelDirectory" -> {
                try {
                    result.success(getModelDirectory())
                } catch (e: Exception) {
                    result.error("DIR_ERROR", e.message ?: "Failed to get model directory", null)
                }
            }
            "enqueueDownload" -> {
                val url = call.argument<String>("url")
                val fileName = call.argument<String>("fileName") ?: call.argument<String>("filename")
                val hfToken = call.argument<String>("hfToken")
                if (url.isNullOrBlank() || fileName.isNullOrBlank()) {
                    result.error("INVALID_ARGS", "url and fileName are required.", null)
                    return
                }
                try {
                    val downloadId = enqueueDownload(url, fileName, hfToken)
                    result.success(downloadId)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to enqueue download for $fileName from $url", e)
                    result.error("ENQUEUE_FAILED", e.message ?: e.toString(), null)
                }
            }
            "cancelDownload" -> {
                val downloadId = (call.argument<Any>("downloadId") as? Number)?.toLong()
                if (downloadId == null) {
                    result.error("INVALID_ARGS", "downloadId is required.", null)
                    return
                }
                try {
                    val cancelled = cancelDownload(downloadId)
                    result.success(cancelled)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to cancel download $downloadId", e)
                    result.error("CANCEL_FAILED", e.message ?: e.toString(), null)
                }
            }
            "queryDownload" -> {
                val downloadId = (call.argument<Any>("downloadId") as? Number)?.toLong()
                if (downloadId == null) {
                    result.error("INVALID_ARGS", "downloadId is required.", null)
                    return
                }
                try {
                    val statusMap = queryDownload(downloadId)
                    result.success(statusMap)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to query download $downloadId", e)
                    result.error("QUERY_FAILED", e.message ?: e.toString(), null)
                }
            }
            else -> result.notImplemented()
        }
    }

    fun enqueueDownload(url: String, fileName: String, hfToken: String?): Long {
        val modelsDir = getModelDirectoryFile()
        val destFile = File(modelsDir, fileName)

        // Clear any existing stale partial file before enqueuing
        if (destFile.exists()) {
            destFile.delete()
        }

        val request = DownloadManager.Request(Uri.parse(url)).apply {
            setTitle(fileName)
            setDescription("Downloading model $fileName")
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setDestinationUri(Uri.fromFile(destFile))
            setAllowedOverMetered(true)
            setAllowedOverRoaming(true)
            if (!hfToken.isNullOrBlank()) {
                addRequestHeader("Authorization", "Bearer $hfToken")
            }
        }

        val downloadId = downloadManager.enqueue(request)
        val tracked = TrackedDownload(downloadId, fileName, destFile, url)
        activeDownloads[downloadId] = tracked

        // Send initial PENDING telemetry
        val initialTelemetry = mapOf(
            "downloadId" to downloadId,
            "downloadedBytes" to 0L,
            "totalBytes" to 0L,
            "status" to "PENDING",
            "reason" to "",
            "filePath" to destFile.absolutePath
        )
        sendEvent(initialTelemetry)

        startPolling()
        return downloadId
    }

    fun cancelDownload(downloadId: Long): Boolean {
        val tracked = activeDownloads.remove(downloadId)
        val removed = try {
            downloadManager.remove(downloadId) > 0
        } catch (e: Exception) {
            Log.e(TAG, "Error removing download from DownloadManager: $downloadId", e)
            false
        }

        // Clean partial file
        if (tracked != null && tracked.file.exists()) {
            tracked.file.delete()
        }

        // Emit cancelled/failed status telemetry
        val telemetry = mapOf(
            "downloadId" to downloadId,
            "downloadedBytes" to 0L,
            "totalBytes" to 0L,
            "status" to "FAILED",
            "reason" to "CANCELLED",
            "filePath" to (tracked?.file?.absolutePath ?: "")
        )
        sendEvent(telemetry)

        if (activeDownloads.isEmpty()) {
            stopPolling()
        }
        return removed || tracked != null
    }

    fun queryDownload(downloadId: Long): Map<String, Any?>? {
        val query = DownloadManager.Query().setFilterById(downloadId)
        var cursor: Cursor? = null
        try {
            cursor = downloadManager.query(query)
            if (cursor != null && cursor.moveToFirst()) {
                val tracked = activeDownloads[downloadId]
                val defaultPath = tracked?.file?.absolutePath
                    ?: File(getModelDirectoryFile(), "unknown_$downloadId.gguf").absolutePath
                return extractTelemetry(cursor, downloadId, defaultPath)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying downloadId $downloadId", e)
        } finally {
            cursor?.close()
        }
        return null
    }

    private fun startPolling() {
        if (!isPolling) {
            isPolling = true
            mainHandler.removeCallbacks(pollRunnable)
            mainHandler.post(pollRunnable)
        }
    }

    private fun stopPolling() {
        isPolling = false
        mainHandler.removeCallbacks(pollRunnable)
    }

    private fun pollProgress() {
        if (activeDownloads.isEmpty()) return

        for ((downloadId, tracked) in activeDownloads) {
            val query = DownloadManager.Query().setFilterById(downloadId)
            var cursor: Cursor? = null
            try {
                cursor = downloadManager.query(query)
                if (cursor != null && cursor.moveToFirst()) {
                    val telemetry = extractTelemetry(cursor, downloadId, tracked.file.absolutePath)
                    sendEvent(telemetry)

                    val status = telemetry["status"] as? String
                    if (status == "SUCCESS" || status == "FAILED") {
                        activeDownloads.remove(downloadId)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error polling download $downloadId", e)
            } finally {
                cursor?.close()
            }
        }
    }

    private fun handleDownloadComplete(downloadId: Long) {
        val tracked = activeDownloads[downloadId] ?: return
        val query = DownloadManager.Query().setFilterById(downloadId)
        var cursor: Cursor? = null
        try {
            cursor = downloadManager.query(query)
            if (cursor != null && cursor.moveToFirst()) {
                val telemetry = extractTelemetry(cursor, downloadId, tracked.file.absolutePath)
                sendEvent(telemetry)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling download complete for $downloadId", e)
        } finally {
            cursor?.close()
            activeDownloads.remove(downloadId)
            if (activeDownloads.isEmpty()) {
                stopPolling()
            }
        }
    }

    private fun extractTelemetry(cursor: Cursor, downloadId: Long, fallbackFilePath: String): Map<String, Any?> {
        val downloadedIdx = cursor.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
        val totalIdx = cursor.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
        val statusIdx = cursor.getColumnIndex(DownloadManager.COLUMN_STATUS)
        val reasonIdx = cursor.getColumnIndex(DownloadManager.COLUMN_REASON)
        val localUriIdx = cursor.getColumnIndex(DownloadManager.COLUMN_LOCAL_URI)

        val downloadedBytes = if (downloadedIdx >= 0) cursor.getLong(downloadedIdx) else 0L
        val totalBytes = if (totalIdx >= 0) cursor.getLong(totalIdx) else 0L
        val statusCode = if (statusIdx >= 0) cursor.getInt(statusIdx) else -1
        val reasonCode = if (reasonIdx >= 0) cursor.getInt(reasonIdx) else 0

        val statusStr = when (statusCode) {
            DownloadManager.STATUS_PENDING -> "PENDING"
            DownloadManager.STATUS_RUNNING -> "RUNNING"
            DownloadManager.STATUS_PAUSED -> "PAUSED"
            DownloadManager.STATUS_SUCCESSFUL -> "SUCCESS"
            DownloadManager.STATUS_FAILED -> "FAILED"
            else -> "UNKNOWN"
        }

        var filePath = fallbackFilePath
        if (localUriIdx >= 0) {
            val localUriStr = cursor.getString(localUriIdx)
            if (!localUriStr.isNullOrBlank()) {
                val parsed = Uri.parse(localUriStr)
                if (parsed.scheme == "file") {
                    parsed.path?.let { filePath = it }
                }
            }
        }

        val reasonStr = if (statusStr == "FAILED" || statusStr == "PAUSED") {
            reasonCode.toString()
        } else {
            ""
        }

        return mapOf(
            "downloadId" to downloadId,
            "downloadedBytes" to downloadedBytes,
            "totalBytes" to totalBytes,
            "status" to statusStr,
            "reason" to reasonStr,
            "filePath" to filePath
        )
    }

    private fun sendEvent(event: Map<String, Any?>) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            eventSink?.success(event)
        } else {
            mainHandler.post {
                eventSink?.success(event)
            }
        }
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        this.eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        this.eventSink = null
    }

    fun dispose() {
        stopPolling()
        unregisterCompleteReceiver()
        activeDownloads.clear()
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        eventSink = null
    }
}
