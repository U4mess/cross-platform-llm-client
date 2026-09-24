class ServerLogEntry {
  final DateTime timestamp;
  final String method;
  final String path;
  final int statusCode;
  final Duration duration;
  final int tokenCount;
  final double tokensPerSecond;

  const ServerLogEntry({
    required this.timestamp,
    required this.method,
    required this.path,
    required this.statusCode,
    required this.duration,
    this.tokenCount = 0,
    this.tokensPerSecond = 0.0,
  });

  String get formattedLine =>
      '[${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}:${timestamp.second.toString().padLeft(2, '0')}] $method $path $statusCode (${(duration.inMilliseconds / 1000).toStringAsFixed(1)}s | $tokenCount tok | ${tokensPerSecond.toStringAsFixed(1)} t/s)';
}
