package com.vaultlm.app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

open class MainActivity : FlutterActivity() {
    protected var downloadHandler: GgufDownloadHandler? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        if (downloadHandler == null) {
            downloadHandler = GgufDownloadHandler(this, flutterEngine.dartExecutor.binaryMessenger)
        }
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        downloadHandler?.dispose()
        downloadHandler = null
        super.cleanUpFlutterEngine(flutterEngine)
    }

    override fun onDestroy() {
        downloadHandler?.dispose()
        downloadHandler = null
        super.onDestroy()
    }
}
