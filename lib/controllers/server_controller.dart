import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

import '../core/constants.dart';
import '../models/server_log_entry.dart';
import '../services/app_log_service.dart';
import '../services/hive_service.dart';
import '../services/inference_service.dart';
import '../services/openai_server_service.dart';
import '../services/server_persistence_service.dart';

class ServerController extends GetxController {
  final HiveService _hive = Get.find<HiveService>();
  final InferenceService inference = Get.find<InferenceService>();
  final OpenAiServerService _server = OpenAiServerService();

  final isRunning = false.obs;
  final isStarting = false.obs;
  final localUrl = RxnString();
  final serverStatus = 'Server stopped'.obs;
  final lastError = RxnString();

  final useApiKey = false.obs;
  final apiKey = ''.obs;

  final currentTokensPerSec = 0.0.obs;
  final sessionTokens = 0.obs;
  final serverInferenceStatus = 'Idle'.obs;
  final requestLogs = <ServerLogEntry>[].obs;

  late final TextEditingController apiKeyCtrl;

  static const int port = 8080;

  @override
  void onInit() {
    super.onInit();
    useApiKey.value =
        _hive.getSetting<bool>(AppConstants.keyServerUseApiKey) ?? false;
    apiKey.value = _hive.getSetting<String>(AppConstants.keyServerApiKey) ?? '';

    apiKeyCtrl = TextEditingController(text: apiKey.value);
  }

  bool get hasLocalModel => inference.isModelLoaded.value;

  String get modelName => inference.loadedModelName.value.isEmpty
      ? 'No model loaded'
      : inference.loadedModelName.value;

  Future<void> toggleServer(bool enabled) async {
    if (enabled) {
      await startServer();
    } else {
      await stopServer();
    }
  }

  Future<void> startServer() async {
    if (isRunning.value || isStarting.value) return;
    lastError.value = null;
    if (!hasLocalModel) {
      lastError.value = 'Load a local GGUF or LiteRT-LM model first.';
      Get.snackbar('Server not started', lastError.value!);
      return;
    }

    isStarting.value = true;
    serverStatus.value = 'Starting server...';
    await saveSettings();

    try {
      await _server.start(
        port: port,
        apiKey: useApiKey.value ? apiKey.value : null,
        onLog: (message) => serverStatus.value = message,
      );
      localUrl.value = _server.localUrl;
      isRunning.value = true;
      serverStatus.value = 'Server running';

      if (Get.isRegistered<ServerPersistenceService>()) {
        final uri = _server.localUrl != null ? Uri.tryParse(_server.localUrl!) : null;
        final host = (uri?.host.isNotEmpty == true) ? uri!.host : 'localhost';
        final serverPort = uri?.hasPort == true ? uri!.port : port;
        await Get.find<ServerPersistenceService>().start(
          host: host,
          port: serverPort,
        );
      }
    } catch (e) {
      lastError.value = '$e';
      serverStatus.value = 'Server failed';
      Get.find<AppLogService>().error('API server failed', details: e);
      Get.snackbar('Server failed', '$e');
    } finally {
      isStarting.value = false;
    }
  }

  Future<void> stopServer() async {
    isStarting.value = false;
    await _server.stop();
    if (Get.isRegistered<ServerPersistenceService>()) {
      await Get.find<ServerPersistenceService>().stop();
    }
    isRunning.value = false;
    localUrl.value = null;
    serverStatus.value = 'Server stopped';
    currentTokensPerSec.value = 0.0;
    serverInferenceStatus.value = 'Idle';
  }

  void clearLogs() {
    requestLogs.clear();
  }

  void logRequest({
    required String method,
    required String path,
    required int statusCode,
    required Duration duration,
    int tokenCount = 0,
    double tokensPerSecond = 0.0,
  }) {
    final entry = ServerLogEntry(
      timestamp: DateTime.now(),
      method: method,
      path: path,
      statusCode: statusCode,
      duration: duration,
      tokenCount: tokenCount,
      tokensPerSecond: tokensPerSecond,
    );
    if (requestLogs.length >= 100) {
      requestLogs.removeAt(0);
    }
    requestLogs.add(entry);
    if (tokenCount > 0) {
      sessionTokens.value += tokenCount;
    }
  }

  Future<void> saveSettings() async {
    await _hive.setSetting(AppConstants.keyServerUseApiKey, useApiKey.value);
    await _hive.setSetting(AppConstants.keyServerApiKey, apiKey.value.trim());
  }

  Future<void> generateApiKey() async {
    final random = Random.secure();
    final bytes = List<int>.generate(24, (_) => random.nextInt(256));
    apiKey.value =
        'aichat_${bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join()}';
    useApiKey.value = true;
    await saveSettings();
  }

  Future<void> copyText(String text, String label) async {
    await Clipboard.setData(ClipboardData(text: text));
    Get.snackbar('Copied', '$label copied.');
  }

  String get baseUrl => localUrl.value ?? 'http://localhost:$port';

  String get openAiBaseUrl => '$baseUrl/v1';

  @override
  void onClose() {
    apiKeyCtrl.dispose();
    unawaited(stopServer());
    super.onClose();
  }
}
