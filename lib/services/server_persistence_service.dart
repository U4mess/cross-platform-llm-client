import 'dart:async';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:get/get.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'app_log_service.dart';

@pragma('vm:entry-point')
void startForegroundTaskCallback() {
  FlutterForegroundTask.setTaskHandler(_ServerTaskHandler());
}

class _ServerTaskHandler extends TaskHandler {
  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {}

  @override
  void onRepeatEvent(DateTime timestamp) {}

  @override
  Future<void> onDestroy(DateTime timestamp) async {}
}

class ServerPersistenceService extends GetxService {
  bool _initialized = false;

  /// Initialize the foreground task and notification channel
  Future<void> init() async {
    if (kIsWeb) return;
    if (_initialized) return;

    try {
      FlutterForegroundTask.initCommunicationPort();
      FlutterForegroundTask.init(
        androidNotificationOptions: AndroidNotificationOptions(
          channelId: 'vaultlm_server_channel',
          channelName: 'VaultLM Server Channel',
          channelDescription: 'Notification channel for VaultLM API Server',
          channelImportance: NotificationChannelImportance.LOW,
          priority: NotificationPriority.LOW,
        ),
        iosNotificationOptions: const IOSNotificationOptions(
          showNotification: false,
          playSound: false,
        ),
        foregroundTaskOptions: ForegroundTaskOptions(
          eventAction: ForegroundTaskEventAction.nothing(),
          autoRunOnBoot: false,
          allowWakeLock: true,
        ),
      );
      _initialized = true;
    } catch (e) {
      Get.find<AppLogService>().warning('Failed to init ServerPersistenceService: $e');
    }
  }

  /// Start foreground service and acquire wakelock
  Future<void> start({required String host, required int port}) async {
    if (kIsWeb) return;

    if (!_initialized) {
      await init();
    }

    try {
      await WakelockPlus.enable();
    } catch (e) {
      Get.find<AppLogService>().warning('Failed to enable WakelockPlus: $e');
    }

    try {
      final isRunning = await FlutterForegroundTask.isRunningService;
      if (isRunning) {
        await FlutterForegroundTask.updateService(
          notificationTitle: 'VaultLM API Active',
          notificationText: 'http://$host:$port',
        );
      } else {
        await FlutterForegroundTask.startService(
          serviceId: 256,
          notificationTitle: 'VaultLM API Active',
          notificationText: 'http://$host:$port',
          callback: startForegroundTaskCallback,
        );
      }
    } catch (e) {
      Get.find<AppLogService>().warning('Failed to start FlutterForegroundTask: $e');
    }
  }

  /// Stop foreground service and release wakelock
  Future<void> stop() async {
    if (kIsWeb) return;

    try {
      await WakelockPlus.disable();
    } catch (e) {
      Get.find<AppLogService>().warning('Failed to disable WakelockPlus: $e');
    }

    try {
      final isRunning = await FlutterForegroundTask.isRunningService;
      if (isRunning) {
        await FlutterForegroundTask.stopService();
      }
    } catch (e) {
      Get.find<AppLogService>().warning('Failed to stop FlutterForegroundTask: $e');
    }
  }
}
