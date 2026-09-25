import 'dart:async';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import '../controllers/model_controller.dart';

/// Status of a native background download transfer.
enum DownloadStatus {
  pending,
  running,
  paused,
  success,
  failed,
  unknown;

  static DownloadStatus fromString(String? str) {
    switch (str?.toUpperCase()) {
      case 'PENDING':
        return DownloadStatus.pending;
      case 'RUNNING':
        return DownloadStatus.running;
      case 'PAUSED':
        return DownloadStatus.paused;
      case 'SUCCESS':
        return DownloadStatus.success;
      case 'FAILED':
        return DownloadStatus.failed;
      default:
        return DownloadStatus.unknown;
    }
  }
}

/// Typed telemetry payload for real-time download progress.
class DownloadProgressPayload {
  final int downloadId;
  final int downloadedBytes;
  final int totalBytes;
  final DownloadStatus status;
  final String reason;
  final String filePath;
  final String fileName;

  DownloadProgressPayload({
    required this.downloadId,
    required this.downloadedBytes,
    required this.totalBytes,
    required this.status,
    required this.reason,
    required this.filePath,
    String? fileName,
  }) : fileName = fileName ??
            (filePath.isNotEmpty ? filePath.split('/').last : 'model.gguf');

  factory DownloadProgressPayload.fromMap(
    Map<dynamic, dynamic> map, {
    String? defaultFileName,
  }) {
    final downloadId = (map['downloadId'] as num?)?.toInt() ?? -1;
    final downloadedBytes = (map['downloadedBytes'] as num?)?.toInt() ?? 0;
    final totalBytes = (map['totalBytes'] as num?)?.toInt() ?? 0;
    final statusStr = map['status'] as String?;
    final reason = map['reason']?.toString() ?? '';
    final filePath = map['filePath']?.toString() ?? '';
    final name = (filePath.isNotEmpty ? filePath.split('/').last : null) ??
        defaultFileName;

    return DownloadProgressPayload(
      downloadId: downloadId,
      downloadedBytes: downloadedBytes,
      totalBytes: totalBytes,
      status: DownloadStatus.fromString(statusStr),
      reason: reason,
      filePath: filePath,
      fileName: name,
    );
  }

  double get progress =>
      totalBytes > 0 ? (downloadedBytes / totalBytes).clamp(0.0, 1.0) : 0.0;

  bool get isCompleted => status == DownloadStatus.success;
  bool get isFailed => status == DownloadStatus.failed;
  bool get isRunning => status == DownloadStatus.running;
  bool get isPending => status == DownloadStatus.pending;
  bool get isPaused => status == DownloadStatus.paused;

  String get formattedDownloaded => _formatBytes(downloadedBytes);
  String get formattedTotal => _formatBytes(totalBytes);

  static String _formatBytes(int bytes) {
    if (bytes <= 0) return '0 B';
    const gb = 1024 * 1024 * 1024;
    const mb = 1024 * 1024;
    if (bytes >= gb) {
      return '${(bytes / gb).toStringAsFixed(2)} GB';
    }
    return '${(bytes / mb).toStringAsFixed(1)} MB';
  }
}

/// Service bridging to native Android DownloadManager for background-resilient GGUF downloads.
class HfDownloadService extends GetxService {
  static const MethodChannel _methodChannel =
      MethodChannel('com.vaultlm.app/downloader');
  static const EventChannel _eventChannel =
      EventChannel('com.vaultlm.app/downloader_events');

  final activeTransfers = <int, DownloadProgressPayload>{}.obs;
  final _trackedFileNames = <int, String>{};

  StreamSubscription? _eventSubscription;
  final _progressStreamController =
      StreamController<DownloadProgressPayload>.broadcast();

  /// Broadcast stream of real-time download telemetry.
  Stream<DownloadProgressPayload> get progressStream =>
      _progressStreamController.stream;

  @override
  void onInit() {
    super.onInit();
    _initEventStream();
  }

  @override
  void onClose() {
    _eventSubscription?.cancel();
    _progressStreamController.close();
    super.onClose();
  }

  void _initEventStream() {
    try {
      _eventSubscription = _eventChannel
          .receiveBroadcastStream()
          .listen(_onEventReceived, onError: (error) {
        // Ignored or logged
      });
    } catch (_) {
      // Non-Android platforms or test runner
    }
  }

  void _onEventReceived(dynamic event) {
    if (event is Map) {
      final downloadId = (event['downloadId'] as num?)?.toInt() ?? -1;
      final knownName = _trackedFileNames[downloadId];
      final payload =
          DownloadProgressPayload.fromMap(event, defaultFileName: knownName);

      activeTransfers[payload.downloadId] = payload;
      _progressStreamController.add(payload);

      if (payload.status == DownloadStatus.success) {
        // Trigger re-scan of the models directory so the downloaded GGUF appears immediately
        try {
          Get.find<ModelController>().refreshDownloaded();
        } catch (_) {}
      }
    }
  }

  /// Enqueues a download via native Android DownloadManager.
  Future<int> enqueueDownload(
    String url,
    String fileName, {
    String? hfToken,
  }) async {
    try {
      final downloadId =
          await _methodChannel.invokeMethod<num>('enqueueDownload', {
        'url': url,
        'fileName': fileName,
        'hfToken': hfToken,
      });
      final id = downloadId?.toInt() ?? -1;
      _trackedFileNames[id] = fileName;
      activeTransfers[id] = DownloadProgressPayload(
        downloadId: id,
        downloadedBytes: 0,
        totalBytes: 0,
        status: DownloadStatus.pending,
        reason: '',
        filePath: '',
        fileName: fileName,
      );
      return id;
    } on PlatformException catch (e) {
      throw Exception('Failed to enqueue download: ${e.message}');
    }
  }

  /// Cancels an active download and deletes partial files.
  Future<bool> cancelDownload(int downloadId) async {
    try {
      final bool? result =
          await _methodChannel.invokeMethod<bool>('cancelDownload', {
        'downloadId': downloadId,
      });
      activeTransfers.remove(downloadId);
      _trackedFileNames.remove(downloadId);
      return result ?? false;
    } on PlatformException catch (e) {
      throw Exception('Failed to cancel download: ${e.message}');
    }
  }

  /// Returns the absolute POSIX path of the models directory on app-scoped external storage.
  Future<String> getModelDirectory() async {
    try {
      final String? dir =
          await _methodChannel.invokeMethod<String>('getModelDirectory');
      return dir ?? '';
    } on PlatformException catch (e) {
      throw Exception('Failed to get model directory: ${e.message}');
    }
  }

  /// Queries the single-shot status of a download.
  Future<DownloadProgressPayload?> queryDownload(int downloadId) async {
    try {
      final map = await _methodChannel.invokeMethod<Map>('queryDownload', {
        'downloadId': downloadId,
      });
      if (map != null) {
        final knownName = _trackedFileNames[downloadId];
        return DownloadProgressPayload.fromMap(map, defaultFileName: knownName);
      }
      return null;
    } catch (_) {
      return null;
    }
  }
}
