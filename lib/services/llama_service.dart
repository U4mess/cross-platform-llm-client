import 'dart:async';
import 'package:get/get.dart';
import '../core/constants.dart';
import 'hive_service.dart';
import 'inference_service.dart';
import 'app_log_service.dart';

/// Service responsible for managing llama.cpp context parameters,
/// thread tuning, and batch sizing for optimal generation throughput.
class LlamaService extends GetxService {
  final HiveService _hive = Get.find<HiveService>();

  /// Configured CPU generation / decode threads (1-8, default: 4)
  int get cpuThreads =>
      _hive.getSetting<int>(
        AppConstants.keyCpuThreads,
        defaultValue: AppConstants.defaultCpuThreads,
      ) ??
      AppConstants.defaultCpuThreads;

  /// Configured prompt prefill / batch processing threads (1-8, default: 6)
  int get batchThreads =>
      _hive.getSetting<int>(
        AppConstants.keyBatchThreads,
        defaultValue: AppConstants.defaultBatchThreads,
      ) ??
      AppConstants.defaultBatchThreads;

  /// Configured batch size for n_batch / n_ubatch evaluation (default: 512)
  int get batchSize =>
      _hive.getSetting<int>(
        AppConstants.keyBatchSize,
        defaultValue: AppConstants.defaultBatchSize,
      ) ??
      AppConstants.defaultBatchSize;

  /// Returns current context configuration map for llama context initialization.
  Map<String, dynamic> buildContextConfig({int? contextSize}) {
    final effectiveCtx = contextSize ??
        (_hive.getSetting<int>(
              AppConstants.keyContextSize,
              defaultValue: AppConstants.defaultContextSize,
            ) ??
            AppConstants.defaultContextSize);

    return {
      'n_ctx': effectiveCtx,
      'n_threads': cpuThreads,
      'n_threads_batch': batchThreads,
      'n_batch': batchSize,
      'n_ubatch': batchSize,
    };
  }

  /// Dynamically updates active llama.cpp execution threads via llama_set_n_threads.
  Future<void> applyThreadSettings({
    int? cpuThreads,
    int? batchThreads,
  }) async {
    try {
      final inference = Get.find<InferenceService>();
      if (inference.isModelLoaded.value &&
          inference.loadedModelRuntime.value == 'llama') {
        await inference.updateThreads(
          cpuThreads: cpuThreads ?? this.cpuThreads,
          batchThreads: batchThreads ?? this.batchThreads,
        );
      }
    } catch (e) {
      Get.find<AppLogService>().warning(
        'Failed to dynamically apply llama threads',
        details: e,
      );
    }
  }

  /// Cleanly reloads the native llama.cpp model context when structural parameters
  /// such as batch size or context length change.
  Future<String> reloadContext() async {
    try {
      final inference = Get.find<InferenceService>();
      if (inference.isModelLoaded.value &&
          inference.loadedModelRuntime.value == 'llama') {
        return await inference.reloadModel();
      }
      return 'No active Llama model loaded.';
    } catch (e) {
      Get.find<AppLogService>().error(
        'Failed to reload llama context',
        details: e,
      );
      return 'ERROR: $e';
    }
  }
}
