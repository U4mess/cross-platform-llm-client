import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:get/get.dart';
import 'package:path_provider/path_provider.dart';

import '../controllers/server_controller.dart';
import '../controllers/settings_controller.dart';
import '../core/constants.dart';
import 'inference_service.dart';

class OpenAiServerService {
  HttpServer? _server;
  final _InferenceLock _inferenceLock = _InferenceLock();
  bool get isBusy => _inferenceLock.isLocked;
  String? _apiKey;
  void Function(String)? _onLog;

  bool get isRunning => _server != null;
  String? get localUrl {
    final port = _server?.port;
    if (port == null) return null;
    final host = _lastReachableAddress;
    return 'http://${host ?? 'localhost'}:$port';
  }

  String? _lastReachableAddress;

  static const int maxBodyBytes = 18 * 1024 * 1024;
  static const int maxDecodedAttachmentBytes = 12 * 1024 * 1024;

  Future<void> start({
    int port = 8080,
    String? apiKey,
    void Function(String)? onLog,
  }) async {
    if (_server != null) return;
    _apiKey = apiKey?.trim();
    _onLog = onLog;
    _lastReachableAddress = await _reachableIpv4Address();
    _server = await HttpServer.bind(InternetAddress.anyIPv4, port);
    final displayUrl = localUrl ?? 'http://localhost:$port';
    _onLog?.call('Server listening on $displayUrl');
    unawaited(_serve(_server!));
  }

  Future<void> stop() async {
    final server = _server;
    _server = null;
    await server?.close(force: true);
    _inferenceLock.clear();
    _onLog?.call('Server stopped');
  }

  Future<void> _serve(HttpServer server) async {
    await for (final request in server) {
      unawaited(_handle(request));
    }
  }

  Future<void> _handle(HttpRequest request) async {
    final stopwatch = Stopwatch()..start();
    try {
      _addCorsHeaders(request.response);
      if (request.method == 'OPTIONS') {
        request.response.statusCode = HttpStatus.noContent;
        await request.response.close();
        return;
      }

      final path = request.uri.path;
      if (request.method == 'GET' && (path == '/' || path.isEmpty || path == '/health')) {
        final inference = Get.isRegistered<InferenceService>()
            ? Get.find<InferenceService>()
            : null;
        await _json(request, {
          'status': 'online',
          'name': 'VaultLM Local Server',
          'version': '1.0.8',
          'active_model': inference?.loadedModelName.value ?? '',
        });
        _log(request: request, statusCode: HttpStatus.ok, duration: stopwatch.elapsed);
        return;
      }

      if (path.startsWith('/v1/') && !_isAuthorized(request)) {
        await _json(request, {'error': 'Unauthorized'},
            status: HttpStatus.unauthorized);
        _log(request: request, statusCode: HttpStatus.unauthorized, duration: stopwatch.elapsed);
        return;
      }

      if (request.method == 'GET' && path == '/v1/models') {
        await _handleModels(request);
        _log(request: request, statusCode: HttpStatus.ok, duration: stopwatch.elapsed);
        return;
      }
      if (request.method == 'GET' && path == '/v1/server/capabilities') {
        await _handleCapabilities(request);
        _log(request: request, statusCode: HttpStatus.ok, duration: stopwatch.elapsed);
        return;
      }
      if (request.method == 'POST' && path == '/v1/chat/completions') {
        await _handleChatCompletions(request, stopwatch: stopwatch);
        return;
      }
      if (request.method == 'POST' && path == '/v1/completions') {
        await _handleCompletions(request, stopwatch: stopwatch);
        return;
      }

      await _json(request, {'error': 'Not found'}, status: HttpStatus.notFound);
      _log(request: request, statusCode: HttpStatus.notFound, duration: stopwatch.elapsed);
    } catch (error) {
      _onLog?.call('Request failed: $error');
      try {
        await _json(
          request,
          {'error': 'Internal server error', 'message': '$error'},
          status: HttpStatus.internalServerError,
        );
      } catch (_) {
        await request.response.close();
      }
      _log(request: request, statusCode: HttpStatus.internalServerError, duration: stopwatch.elapsed);
    }
  }

  void _log({
    required HttpRequest request,
    required int statusCode,
    required Duration duration,
    int tokenCount = 0,
    double tokensPerSecond = 0.0,
  }) {
    if (Get.isRegistered<ServerController>()) {
      Get.find<ServerController>().logRequest(
        method: request.method,
        path: request.uri.path,
        statusCode: statusCode,
        duration: duration,
        tokenCount: tokenCount,
        tokensPerSecond: tokensPerSecond,
      );
    }
  }

  bool _isAuthorized(HttpRequest request) {
    final key = _apiKey;
    if (key == null || key.isEmpty) return true;
    final header = request.headers.value(HttpHeaders.authorizationHeader) ?? '';
    return header.trim() == 'Bearer $key';
  }

  Future<void> _handleModels(HttpRequest request) async {
    final inference = Get.find<InferenceService>();
    final hasModel = inference.isModelLoaded.value;
    await _json(request, {
      'object': 'list',
      'data': [
        if (hasModel)
          {
            'id': inference.loadedModelName.value,
            'object': 'model',
            'created': DateTime.now().millisecondsSinceEpoch ~/ 1000,
            'owned_by': 'local',
          }
      ],
    });
  }

  Future<void> _handleCapabilities(HttpRequest request) async {
    final inference = Get.find<InferenceService>();
    final hasModel = inference.isModelLoaded.value;
    final isLiteRt = hasModel && inference.loadedModelRuntime.value == 'litert';
    await _json(request, {
      'server': 'AI Chat Local OpenAI API',
      'running': true,
      'model': hasModel ? inference.loadedModelName.value : null,
      'runtime': inference.loadedModelRuntime.value,
      'requires_litert': false,
      'capabilities': {
        'text': hasModel,
        'image': isLiteRt && inference.isVisionLoaded.value,
        'audio': isLiteRt,
        'streaming': hasModel,
        'gguf': hasModel && !isLiteRt,
      },
    });
  }

  Future<void> _handleChatCompletions(
    HttpRequest request, {
    required Stopwatch stopwatch,
  }) async {
    final body = await _readJson(request);
    final inference = Get.find<InferenceService>();
    final modelError = _localModelError(inference);
    if (modelError != null) {
      await _json(request, {'error': modelError},
          status: HttpStatus.badRequest);
      _log(request: request, statusCode: HttpStatus.badRequest, duration: stopwatch.elapsed);
      return;
    }

    final parsed = await _parseChatRequest(body);
    if (parsed.error != null) {
      await _json(request, {'error': parsed.error},
          status: HttpStatus.badRequest);
      _log(request: request, statusCode: HttpStatus.badRequest, duration: stopwatch.elapsed);
      return;
    }

    final rawModel = (body['model'] as String?)?.trim();
    final effectiveModel = (rawModel != null && rawModel.isNotEmpty)
        ? rawModel
        : inference.loadedModelName.value;

    final acquired = await _inferenceLock.acquire();
    if (!acquired) {
      await _json(
        request,
        {'error': 'Too Many Requests - inference engine busy'},
        status: HttpStatus.tooManyRequests,
      );
      await parsed.cleanup();
      _log(request: request, statusCode: HttpStatus.tooManyRequests, duration: stopwatch.elapsed);
      return;
    }

    final stream = body['stream'] == true;
    var tokenCount = 0;
    try {
      while (inference.isGenerating.value) {
        await Future.delayed(const Duration(milliseconds: 100));
      }

      if (stream) {
        tokenCount = await _streamChatResponse(
          request,
          inference,
          parsed,
          modelName: effectiveModel,
          stopwatch: stopwatch,
        );
      } else {
        if (Get.isRegistered<ServerController>()) {
          Get.find<ServerController>().serverInferenceStatus.value = 'Streaming';
        }
        final text = await inference.generate(
          prompt: parsed.prompt,
          systemPrompt: parsed.systemPrompt ??
              _defaultSystemPrompt(inference.loadedModelName.value),
          conversationHistory: parsed.history,
          source: 'server',
          imagePath: parsed.imagePath,
          audioPath: parsed.audioPath,
          onToken: (token) {
            tokenCount++;
            final elapsedSeconds = stopwatch.elapsedMilliseconds / 1000.0;
            if (elapsedSeconds > 0.05 && Get.isRegistered<ServerController>()) {
              Get.find<ServerController>().currentTokensPerSec.value =
                  tokenCount / elapsedSeconds;
            }
          },
        );
        if (tokenCount == 0 && text.isNotEmpty) {
          tokenCount = max(1, text.split(RegExp(r'\s+')).length);
        }
        await _json(
          request,
          _chatResponse(effectiveModel, text),
        );
      }
      final duration = stopwatch.elapsed;
      final finalTps = duration.inMilliseconds > 50
          ? (tokenCount / (duration.inMilliseconds / 1000.0))
          : 0.0;
      _log(
        request: request,
        statusCode: HttpStatus.ok,
        duration: duration,
        tokenCount: tokenCount,
        tokensPerSecond: finalTps,
      );
    } finally {
      if (Get.isRegistered<ServerController>()) {
        Get.find<ServerController>().currentTokensPerSec.value = 0.0;
        Get.find<ServerController>().serverInferenceStatus.value = 'Idle';
      }
      _inferenceLock.release();
      await parsed.cleanup();
    }
  }

  Future<void> _handleCompletions(
    HttpRequest request, {
    required Stopwatch stopwatch,
  }) async {
    final body = await _readJson(request);
    final inference = Get.find<InferenceService>();
    final modelError = _localModelError(inference);
    if (modelError != null) {
      await _json(request, {'error': modelError},
          status: HttpStatus.badRequest);
      _log(request: request, statusCode: HttpStatus.badRequest, duration: stopwatch.elapsed);
      return;
    }

    final prompt = body['prompt'];
    if (prompt is! String || prompt.trim().isEmpty) {
      await _json(request, {'error': 'prompt is required'},
          status: HttpStatus.badRequest);
      _log(request: request, statusCode: HttpStatus.badRequest, duration: stopwatch.elapsed);
      return;
    }

    final rawModel = (body['model'] as String?)?.trim();
    final effectiveModel = (rawModel != null && rawModel.isNotEmpty)
        ? rawModel
        : inference.loadedModelName.value;

    final acquired = await _inferenceLock.acquire();
    if (!acquired) {
      await _json(
        request,
        {'error': 'Too Many Requests - inference engine busy'},
        status: HttpStatus.tooManyRequests,
      );
      _log(request: request, statusCode: HttpStatus.tooManyRequests, duration: stopwatch.elapsed);
      return;
    }

    final stream = body['stream'] == true;
    var tokenCount = 0;
    try {
      while (inference.isGenerating.value) {
        await Future.delayed(const Duration(milliseconds: 100));
      }

      if (stream) {
        tokenCount = await _streamCompletionResponse(
          request,
          inference,
          prompt,
          modelName: effectiveModel,
          stopwatch: stopwatch,
        );
      } else {
        if (Get.isRegistered<ServerController>()) {
          Get.find<ServerController>().serverInferenceStatus.value = 'Streaming';
        }
        final text = await inference.generate(
          prompt: prompt,
          systemPrompt: _defaultSystemPrompt(inference.loadedModelName.value),
          source: 'server',
          onToken: (token) {
            tokenCount++;
            final elapsedSeconds = stopwatch.elapsedMilliseconds / 1000.0;
            if (elapsedSeconds > 0.05 && Get.isRegistered<ServerController>()) {
              Get.find<ServerController>().currentTokensPerSec.value =
                  tokenCount / elapsedSeconds;
            }
          },
        );
        if (tokenCount == 0 && text.isNotEmpty) {
          tokenCount = max(1, text.split(RegExp(r'\s+')).length);
        }
        await _json(request, {
          'id': 'cmpl-${_id()}',
          'object': 'text_completion',
          'created': DateTime.now().millisecondsSinceEpoch ~/ 1000,
          'model': effectiveModel,
          'choices': [
            {'index': 0, 'text': text, 'finish_reason': 'stop'}
          ],
        });
      }
      final duration = stopwatch.elapsed;
      final finalTps = duration.inMilliseconds > 50
          ? (tokenCount / (duration.inMilliseconds / 1000.0))
          : 0.0;
      _log(
        request: request,
        statusCode: HttpStatus.ok,
        duration: duration,
        tokenCount: tokenCount,
        tokensPerSecond: finalTps,
      );
    } finally {
      if (Get.isRegistered<ServerController>()) {
        Get.find<ServerController>().currentTokensPerSec.value = 0.0;
        Get.find<ServerController>().serverInferenceStatus.value = 'Idle';
      }
      _inferenceLock.release();
    }
  }

  String? _localModelError(InferenceService inference) {
    if (!inference.isModelLoaded.value) {
      return 'No local model loaded. Load a GGUF or LiteRT-LM model first.';
    }
    return null;
  }

  Future<_ParsedChatRequest> _parseChatRequest(
      Map<String, dynamic> body) async {
    final rawMessages = body['messages'];
    if (rawMessages is! List || rawMessages.isEmpty) {
      return _ParsedChatRequest.error('messages must be a non-empty array');
    }

    final systemParts = <String>[];
    final history = <Map<String, String>>[];
    var lastUserText = '';
    String? imagePath;
    String? audioPath;
    final tempFiles = <File>[];

    for (var i = 0; i < rawMessages.length; i++) {
      final raw = rawMessages[i];
      if (raw is! Map)
        return _ParsedChatRequest.error('message[$i] must be an object');
      final role = '${raw['role'] ?? ''}';
      if (role != 'system' && role != 'user' && role != 'assistant') {
        return _ParsedChatRequest.error('message[$i].role is unsupported');
      }
      final contentResult = await _parseContent(raw['content']);
      if (contentResult.error != null)
        return _ParsedChatRequest.error(contentResult.error!);
      tempFiles.addAll(contentResult.tempFiles);
      imagePath ??= contentResult.imagePath;
      audioPath ??= contentResult.audioPath;

      if (role == 'system') {
        if (contentResult.text.trim().isNotEmpty)
          systemParts.add(contentResult.text.trim());
        continue;
      }
      if (i == rawMessages.length - 1 && role == 'user') {
        lastUserText = contentResult.text.trim();
      } else {
        history.add({'role': role, 'content': contentResult.text});
      }
    }

    if (lastUserText.isEmpty && imagePath == null && audioPath == null) {
      return _ParsedChatRequest.error(
          'last user message must contain text, image, or audio');
    }

    return _ParsedChatRequest(
      prompt: lastUserText.isEmpty ? 'Describe this attachment.' : lastUserText,
      systemPrompt: systemParts.isEmpty ? null : systemParts.join('\n'),
      history: history,
      imagePath: imagePath,
      audioPath: audioPath,
      tempFiles: tempFiles,
    );
  }

  Future<_ContentResult> _parseContent(dynamic content) async {
    if (content is String) return _ContentResult(text: content);
    if (content is! List)
      return _ContentResult.error(
          'message.content must be a string or content array');

    final text = StringBuffer();
    String? imagePath;
    String? audioPath;
    final tempFiles = <File>[];

    for (final part in content) {
      if (part is! Map)
        return _ContentResult.error('content part must be an object');
      final type = '${part['type'] ?? ''}';
      if (type == 'text') {
        text.write('${part['text'] ?? ''}');
      } else if (type == 'image_url') {
        if (imagePath != null)
          return _ContentResult.error(
              'only one image is supported per request');
        final imageUrl = part['image_url'];
        final url = imageUrl is Map ? '${imageUrl['url'] ?? ''}' : '';
        final file = await _dataUrlToTempFile(url, 'image');
        if (file.error != null) return _ContentResult.error(file.error!);
        imagePath = file.path;
        tempFiles.add(file.file!);
      } else if (type == 'input_audio' || type == 'audio_url') {
        if (audioPath != null)
          return _ContentResult.error(
              'only one audio file is supported per request');
        final raw = part[type == 'input_audio' ? 'input_audio' : 'audio_url'];
        final data = raw is Map ? '${raw['data'] ?? raw['url'] ?? ''}' : '';
        final file = await _dataUrlToTempFile(data, 'audio');
        if (file.error != null) return _ContentResult.error(file.error!);
        audioPath = file.path;
        tempFiles.add(file.file!);
      } else {
        return _ContentResult.error('unsupported content part type: $type');
      }
    }

    return _ContentResult(
      text: text.toString(),
      imagePath: imagePath,
      audioPath: audioPath,
      tempFiles: tempFiles,
    );
  }

  Future<_TempFileResult> _dataUrlToTempFile(String value, String kind) async {
    if (!value.startsWith('data:')) {
      return _TempFileResult.error(
          'Only base64 data URLs are accepted for $kind input');
    }
    final comma = value.indexOf(',');
    if (comma <= 0 || !value.substring(0, comma).contains(';base64')) {
      return _TempFileResult.error('$kind input must be a base64 data URL');
    }
    final meta = value.substring(5, comma).toLowerCase();
    final encoded = value.substring(comma + 1);
    if (encoded.length > maxDecodedAttachmentBytes * 2) {
      return _TempFileResult.error('$kind input is too large');
    }
    late List<int> bytes;
    try {
      bytes = base64Decode(encoded);
    } catch (_) {
      return _TempFileResult.error('$kind input has invalid base64');
    }
    if (bytes.length > maxDecodedAttachmentBytes) {
      return _TempFileResult.error('$kind input is too large');
    }
    final ext = _extensionForMime(meta, kind);
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/server_${kind}_${_id()}.$ext');
    await file.writeAsBytes(bytes, flush: true);
    return _TempFileResult(file);
  }

  String _extensionForMime(String mime, String kind) {
    if (mime.contains('png')) return 'png';
    if (mime.contains('webp')) return 'webp';
    if (mime.contains('jpg') || mime.contains('jpeg')) return 'jpg';
    if (mime.contains('wav')) return 'wav';
    if (mime.contains('mp3') || mime.contains('mpeg')) return 'mp3';
    if (mime.contains('m4a') || mime.contains('mp4')) return 'm4a';
    return kind == 'image' ? 'png' : 'wav';
  }

  Future<int> _streamChatResponse(
    HttpRequest request,
    InferenceService inference,
    _ParsedChatRequest parsed, {
    String? modelName,
    required Stopwatch stopwatch,
  }) async {
    final response = request.response;
    _addCorsHeaders(response);
    response.statusCode = HttpStatus.ok;
    response.headers.contentType =
        ContentType('text', 'event-stream', charset: 'utf-8');
    response.headers.set(HttpHeaders.cacheControlHeader, 'no-cache');
    final id = 'chatcmpl-${_id()}';
    final created = DateTime.now().millisecondsSinceEpoch ~/ 1000;
    final reportedModel = (modelName != null && modelName.isNotEmpty)
        ? modelName
        : inference.loadedModelName.value;
    var emitted = false;
    var tokenCount = 0;

    void emitContent(String text) {
      if (text.isEmpty) return;
      emitted = true;
      final payload = {
        'id': id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': reportedModel,
        'choices': [
          {
            'index': 0,
            'delta': {'content': text},
            'finish_reason': null,
          }
        ],
      };
      response.write('data: ${jsonEncode(payload)}\n\n');
    }

    void emitReasoning(String text) {
      if (text.isEmpty) return;
      emitted = true;
      final payload = {
        'id': id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': reportedModel,
        'choices': [
          {
            'index': 0,
            'delta': {'reasoning_content': text},
            'finish_reason': null,
          }
        ],
      };
      response.write('data: ${jsonEncode(payload)}\n\n');
    }

    void emitFinish() {
      final payload = {
        'id': id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': reportedModel,
        'choices': [
          {
            'index': 0,
            'delta': <String, dynamic>{},
            'finish_reason': 'stop',
          }
        ],
      };
      response.write('data: ${jsonEncode(payload)}\n\n');
    }

    final parser = _ThinkingStreamParser(
      onContent: emitContent,
      onReasoning: emitReasoning,
    );

    if (Get.isRegistered<ServerController>()) {
      Get.find<ServerController>().serverInferenceStatus.value = 'Streaming';
    }

    final result = await inference.generate(
      prompt: parsed.prompt,
      systemPrompt: parsed.systemPrompt ??
          _defaultSystemPrompt(inference.loadedModelName.value),
      conversationHistory: parsed.history,
      source: 'server',
      imagePath: parsed.imagePath,
      audioPath: parsed.audioPath,
      onToken: (token) {
        tokenCount++;
        final elapsedSeconds = stopwatch.elapsedMilliseconds / 1000.0;
        if (elapsedSeconds > 0.05 && Get.isRegistered<ServerController>()) {
          Get.find<ServerController>().currentTokensPerSec.value =
              tokenCount / elapsedSeconds;
        }
        parser.feed(token);
      },
    );
    parser.flush();

    if (!emitted && result.isNotEmpty) {
      tokenCount++;
      parser.feed(result);
      parser.flush();
    }

    emitFinish();
    response.write('data: [DONE]\n\n');
    await response.close();
    return tokenCount;
  }

  Future<int> _streamCompletionResponse(
    HttpRequest request,
    InferenceService inference,
    String prompt, {
    String? modelName,
    required Stopwatch stopwatch,
  }) async {
    final response = request.response;
    _addCorsHeaders(response);
    response.statusCode = HttpStatus.ok;
    response.headers.contentType =
        ContentType('text', 'event-stream', charset: 'utf-8');
    response.headers.set(HttpHeaders.cacheControlHeader, 'no-cache');
    final id = 'cmpl-${_id()}';
    final created = DateTime.now().millisecondsSinceEpoch ~/ 1000;
    final reportedModel = (modelName != null && modelName.isNotEmpty)
        ? modelName
        : inference.loadedModelName.value;
    var emitted = false;
    var tokenCount = 0;

    void emit(String token) {
      emitted = true;
      response.write('data: ${jsonEncode({
            'id': id,
            'object': 'text_completion',
            'created': created,
            'model': reportedModel,
            'choices': [
              {'index': 0, 'text': token, 'finish_reason': null}
            ],
          })}\n\n');
    }

    if (Get.isRegistered<ServerController>()) {
      Get.find<ServerController>().serverInferenceStatus.value = 'Streaming';
    }

    final result = await inference.generate(
      prompt: prompt,
      systemPrompt: _defaultSystemPrompt(inference.loadedModelName.value),
      source: 'server',
      onToken: (token) {
        tokenCount++;
        final elapsedSeconds = stopwatch.elapsedMilliseconds / 1000.0;
        if (elapsedSeconds > 0.05 && Get.isRegistered<ServerController>()) {
          Get.find<ServerController>().currentTokensPerSec.value =
              tokenCount / elapsedSeconds;
        }
        emit(token);
      },
    );
    if (!emitted && result.isNotEmpty) {
      tokenCount++;
      emit(result);
    }
    response.write('data: [DONE]\n\n');
    await response.close();
    return tokenCount;
  }

  Map<String, dynamic> _chatResponse(String model, String text) {
    final thinkRegex = RegExp(r'<think>(.*?)</think>', dotAll: true);
    final match = thinkRegex.firstMatch(text);
    final Map<String, dynamic> message;

    if (match != null) {
      final reasoning = match.group(1) ?? '';
      var remaining = text.substring(0, match.start) + text.substring(match.end);
      if (remaining.startsWith('\n\n')) {
        remaining = remaining.substring(2);
      } else if (remaining.startsWith('\n')) {
        remaining = remaining.substring(1);
      }
      message = {
        'role': 'assistant',
        'content': remaining,
        'reasoning_content': reasoning,
      };
    } else if (text.contains('<think>')) {
      final unclosedMatch =
          RegExp(r'<think>(.*)', dotAll: true).firstMatch(text);
      final reasoning = unclosedMatch?.group(1) ?? '';
      final before = text.substring(0, unclosedMatch?.start ?? 0);
      message = {
        'role': 'assistant',
        'content': before,
        'reasoning_content': reasoning,
      };
    } else {
      message = {
        'role': 'assistant',
        'content': text,
      };
    }

    return {
      'id': 'chatcmpl-${_id()}',
      'object': 'chat.completion',
      'created': DateTime.now().millisecondsSinceEpoch ~/ 1000,
      'model': model,
      'choices': [
        {
          'index': 0,
          'message': message,
          'finish_reason': 'stop',
        }
      ],
    };
  }

  Future<Map<String, dynamic>> _readJson(HttpRequest request) async {
    final builder = BytesBuilder(copy: false);
    await for (final chunk in request) {
      builder.add(chunk);
      if (builder.length > maxBodyBytes) {
        throw const HttpException('Request body is too large');
      }
    }
    final body = utf8.decode(builder.takeBytes());
    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('JSON body must be an object');
    }
    return decoded;
  }

  Future<void> _json(
    HttpRequest request,
    Map<String, dynamic> data, {
    int status = HttpStatus.ok,
  }) async {
    _addCorsHeaders(request.response);
    request.response.statusCode = status;
    request.response.headers.contentType = ContentType.json;
    request.response.write(jsonEncode(data));
    await request.response.close();
  }

  void _addCorsHeaders(HttpResponse response) {
    response.headers.set(HttpHeaders.accessControlAllowOriginHeader, '*');
    response.headers.set(
      HttpHeaders.accessControlAllowMethodsHeader,
      'GET, POST, OPTIONS',
    );
    response.headers.set(
      HttpHeaders.accessControlAllowHeadersHeader,
      'Authorization, Content-Type, Accept, Origin, User-Agent',
    );
  }

  Future<String?> _reachableIpv4Address() async {
    try {
      final interfaces = await NetworkInterface.list(
        type: InternetAddressType.IPv4,
        includeLoopback: false,
      );
      for (final interface in interfaces) {
        for (final address in interface.addresses) {
          if (!address.isLoopback) return address.address;
        }
      }
    } catch (_) {}
    return null;
  }

  String _id() {
    final rng = Random.secure();
    return List.generate(12, (_) => rng.nextInt(16).toRadixString(16)).join();
  }

  String _defaultSystemPrompt(String modelName) {
    if (Get.isRegistered<SettingsController>()) {
      return Get.find<SettingsController>().effectiveSystemPromptForModel(
        modelName,
      );
    }
    if (AppConstants.isUncensoredModelName(modelName)) {
      return AppConstants.uncensoredSystemPrompt;
    }
    return AppConstants.systemPrompt;
  }
}

class _ParsedChatRequest {
  final String prompt;
  final String? systemPrompt;
  final List<Map<String, String>> history;
  final String? imagePath;
  final String? audioPath;
  final List<File> tempFiles;
  final String? error;

  _ParsedChatRequest({
    required this.prompt,
    required this.systemPrompt,
    required this.history,
    required this.imagePath,
    required this.audioPath,
    required this.tempFiles,
  }) : error = null;

  _ParsedChatRequest.error(this.error)
      : prompt = '',
        systemPrompt = null,
        history = const [],
        imagePath = null,
        audioPath = null,
        tempFiles = const [];

  Future<void> cleanup() async {
    for (final file in tempFiles) {
      try {
        if (await file.exists()) await file.delete();
      } catch (_) {}
    }
  }
}

class _ContentResult {
  final String text;
  final String? imagePath;
  final String? audioPath;
  final List<File> tempFiles;
  final String? error;

  _ContentResult({
    required this.text,
    this.imagePath,
    this.audioPath,
    this.tempFiles = const [],
  }) : error = null;

  _ContentResult.error(this.error)
      : text = '',
        imagePath = null,
        audioPath = null,
        tempFiles = const [];
}

class _TempFileResult {
  final File? file;
  final String? error;

  _TempFileResult(this.file) : error = null;
  _TempFileResult.error(this.error) : file = null;

  String get path => file!.path;
}

class _InferenceLock {
  bool _isLocked = false;
  final List<Completer<void>> _queue = [];
  static const int maxWaitingQueueSize = 3;

  bool get isLocked => _isLocked;
  int get waitingCount => _queue.length;

  Future<bool> acquire() async {
    if (!_isLocked) {
      _isLocked = true;
      return true;
    }

    if (_queue.length >= maxWaitingQueueSize) {
      return false;
    }

    final completer = Completer<void>();
    _queue.add(completer);
    try {
      await completer.future;
      return true;
    } catch (_) {
      _queue.remove(completer);
      return false;
    }
  }

  void release() {
    if (_queue.isNotEmpty) {
      final next = _queue.removeAt(0);
      if (!next.isCompleted) {
        next.complete();
      }
    } else {
      _isLocked = false;
    }
  }

  void clear() {
    while (_queue.isNotEmpty) {
      final next = _queue.removeAt(0);
      if (!next.isCompleted) {
        next.completeError(const HttpException('Server stopped'));
      }
    }
    _isLocked = false;
  }
}

class _ThinkingStreamParser {
  final void Function(String text) onContent;
  final void Function(String text) onReasoning;

  bool _inThinking = false;
  bool _thinkingDone = false;
  String _buffer = '';

  static const String _thinkOpen = '<think>';
  static const String _thinkClose = '</think>';

  _ThinkingStreamParser({
    required this.onContent,
    required this.onReasoning,
  });

  void feed(String token) {
    if (token.isEmpty) return;
    _buffer += token;
    _process();
  }

  void _process() {
    while (_buffer.isNotEmpty) {
      if (!_inThinking && !_thinkingDone) {
        final openIdx = _buffer.indexOf(_thinkOpen);
        if (openIdx != -1) {
          if (openIdx > 0) {
            onContent(_buffer.substring(0, openIdx));
          }
          _inThinking = true;
          _buffer = _buffer.substring(openIdx + _thinkOpen.length);
          continue;
        }

        final partialLen = _matchingPrefixLength(_buffer, _thinkOpen);
        if (partialLen > 0) {
          final emitLen = _buffer.length - partialLen;
          if (emitLen > 0) {
            onContent(_buffer.substring(0, emitLen));
            _buffer = _buffer.substring(emitLen);
          }
          break;
        } else {
          onContent(_buffer);
          _buffer = '';
          break;
        }
      } else if (_inThinking) {
        final closeIdx = _buffer.indexOf(_thinkClose);
        if (closeIdx != -1) {
          if (closeIdx > 0) {
            onReasoning(_buffer.substring(0, closeIdx));
          }
          _inThinking = false;
          _thinkingDone = true;
          var after = _buffer.substring(closeIdx + _thinkClose.length);
          if (after.startsWith('\n\n')) {
            after = after.substring(2);
          } else if (after.startsWith('\n')) {
            after = after.substring(1);
          }
          _buffer = after;
          continue;
        }

        final partialLen = _matchingPrefixLength(_buffer, _thinkClose);
        if (partialLen > 0) {
          final emitLen = _buffer.length - partialLen;
          if (emitLen > 0) {
            onReasoning(_buffer.substring(0, emitLen));
            _buffer = _buffer.substring(emitLen);
          }
          break;
        } else {
          onReasoning(_buffer);
          _buffer = '';
          break;
        }
      } else {
        onContent(_buffer);
        _buffer = '';
        break;
      }
    }
  }

  void flush() {
    if (_buffer.isNotEmpty) {
      if (_inThinking) {
        onReasoning(_buffer);
      } else {
        onContent(_buffer);
      }
      _buffer = '';
    }
  }

  int _matchingPrefixLength(String text, String target) {
    final maxLen = text.length < target.length ? text.length : target.length - 1;
    for (var len = maxLen; len > 0; len--) {
      if (text.endsWith(target.substring(0, len))) {
        return len;
      }
    }
    return 0;
  }
}
