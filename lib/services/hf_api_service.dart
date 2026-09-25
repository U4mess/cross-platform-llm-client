import 'dart:convert';
import 'package:http/http.dart' as http;

/// Represents a Hugging Face model repository.
class HfRepository {
  final String id;
  final int downloads;
  final int likes;
  final String? pipelineTag;
  final List<String> tags;
  final DateTime? createdAt;

  HfRepository({
    required this.id,
    this.downloads = 0,
    this.likes = 0,
    this.pipelineTag,
    this.tags = const [],
    this.createdAt,
  });

  factory HfRepository.fromJson(Map<String, dynamic> json) {
    return HfRepository(
      id: json['id'] as String? ?? json['modelId'] as String? ?? '',
      downloads: (json['downloads'] as num?)?.toInt() ?? 0,
      likes: (json['likes'] as num?)?.toInt() ?? 0,
      pipelineTag: json['pipeline_tag'] as String?,
      tags: (json['tags'] as List?)?.map((e) => e.toString()).toList() ??
          const [],
      createdAt: json['createdAt'] != null
          ? DateTime.tryParse(json['createdAt'] as String)
          : null,
    );
  }

  String get author => id.contains('/') ? id.split('/').first : '';
  String get name => id.contains('/') ? id.split('/').last : id;
}

/// Represents a GGUF file within a Hugging Face model repository.
class HfGgufFile {
  final String path;
  final String filename;
  final int sizeBytes;
  final String sizeFormatted;
  final String quantTag;
  final String downloadUrl;

  HfGgufFile({
    required this.path,
    required this.filename,
    required this.sizeBytes,
    required this.sizeFormatted,
    required this.quantTag,
    required this.downloadUrl,
  });

  factory HfGgufFile.fromTreeJson(Map<String, dynamic> json, String repoId) {
    final path = json['path'] as String? ?? '';
    final filename = path.split('/').last;
    final sizeBytes = (json['size'] as num?)?.toInt() ?? 0;
    final quant = extractQuantTag(filename);
    final sizeStr = formatBytes(sizeBytes);
    final downloadUrl =
        'https://huggingface.co/$repoId/resolve/main/${Uri.encodeComponent(path)}';

    return HfGgufFile(
      path: path,
      filename: filename,
      sizeBytes: sizeBytes,
      sizeFormatted: sizeStr,
      quantTag: quant,
      downloadUrl: downloadUrl,
    );
  }

  static String formatBytes(int bytes) {
    if (bytes <= 0) return 'Unknown size';
    const gb = 1024 * 1024 * 1024;
    const mb = 1024 * 1024;
    if (bytes >= gb) {
      return '${(bytes / gb).toStringAsFixed(2)} GB';
    }
    return '${(bytes / mb).toStringAsFixed(1)} MB';
  }

  static String extractQuantTag(String filename) {
    // Match common quantization patterns: e.g. Q4_K_M, Q8_0, IQ4_XS, BF16, FP16, etc.
    final match = RegExp(
      r'(IQ\d_[A-Z0-9_]+|Q\d_[A-Z0-9_]+|BF16|FP16|F16|F32|Q\d_\d)',
      caseSensitive: false,
    ).firstMatch(filename);
    if (match != null) {
      return match.group(1)!.toUpperCase();
    }
    final parts = filename.split('.');
    if (parts.length > 2) {
      return parts[parts.length - 2].toUpperCase();
    }
    return 'GGUF';
  }
}

/// Service to query Hugging Face API for repositories and GGUF files.
class HfApiService {
  static const String _baseUrl = 'https://huggingface.co/api';

  /// Searches for repositories matching [query] with GGUF weights.
  static Future<List<HfRepository>> searchRepositories(
    String query, {
    String? token,
    int limit = 20,
  }) async {
    final trimmed = query.trim();
    final searchParam = trimmed.isEmpty ? 'gguf' : Uri.encodeComponent(trimmed);
    final url = Uri.parse(
      '$_baseUrl/models?search=$searchParam&filter=gguf&limit=$limit&full=false',
    );

    final headers = <String, String>{
      'Accept': 'application/json',
    };
    if (token != null && token.trim().isNotEmpty) {
      headers['Authorization'] = 'Bearer ${token.trim()}';
    }

    final response = await http.get(url, headers: headers);
    if (response.statusCode != 200) {
      throw Exception(
        'Hugging Face API search error (${response.statusCode}): ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body);
    if (decoded is! List) return [];

    return decoded
        .map((e) => HfRepository.fromJson(Map<String, dynamic>.from(e as Map)))
        .where((r) => r.id.isNotEmpty)
        .toList();
  }

  /// Fetches GGUF files in the repository tree at main.
  static Future<List<HfGgufFile>> fetchRepoGgufFiles(
    String repoId, {
    String? token,
  }) async {
    final url = Uri.parse('$_baseUrl/models/$repoId/tree/main');
    final headers = <String, String>{
      'Accept': 'application/json',
    };
    if (token != null && token.trim().isNotEmpty) {
      headers['Authorization'] = 'Bearer ${token.trim()}';
    }

    final response = await http.get(url, headers: headers);
    if (response.statusCode != 200) {
      throw Exception(
        'Failed to load files from $repoId (${response.statusCode}): ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body);
    if (decoded is! List) return [];

    final ggufFiles = <HfGgufFile>[];
    for (final item in decoded) {
      if (item is Map) {
        final map = Map<String, dynamic>.from(item);
        final path = (map['path'] as String? ?? '').toLowerCase();
        final type = map['type'] as String? ?? '';
        if (type == 'file' && path.endsWith('.gguf')) {
          ggufFiles.add(HfGgufFile.fromTreeJson(map, repoId));
        }
      }
    }

    // Sort by file size ascending
    ggufFiles.sort((a, b) => a.sizeBytes.compareTo(b.sizeBytes));
    return ggufFiles;
  }
}
