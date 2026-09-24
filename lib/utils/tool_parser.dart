import 'dart:convert';
import '../widgets/tool_status_disclosure.dart';

class ToolParseResult {
  final List<ToolExecutionRecord> tools;
  final String cleanText;

  const ToolParseResult({
    required this.tools,
    required this.cleanText,
  });
}

ToolParseResult parseToolTags(String text) {
  final tools = <ToolExecutionRecord>[];
  final toolCallExp = RegExp(r'<tool_call>(.*?)</tool_call>', dotAll: true);
  final toolResponseExp =
      RegExp(r'<tool_response>(.*?)</tool_response>', dotAll: true);

  final callMatches = toolCallExp.allMatches(text).toList();
  final responseMatches = toolResponseExp.allMatches(text).toList();

  for (var i = 0; i < callMatches.length; i++) {
    final callJson = callMatches[i].group(1)?.trim() ?? '';
    String toolName = 'tool';
    String query = '';
    try {
      final parsed = jsonDecode(callJson) as Map<String, dynamic>;
      toolName = (parsed['name'] ?? 'tool').toString();
      query = (parsed['query'] ??
              parsed['url'] ??
              parsed['expression'] ??
              '')
          .toString();
    } catch (_) {}

    String? snippet;
    var state = ToolExecutionState.executing;
    if (i < responseMatches.length) {
      snippet = responseMatches[i].group(1)?.trim();
      final lower = (snippet ?? '').toLowerCase();
      if (lower.startsWith('error') ||
          lower.startsWith('failed to') ||
          lower.startsWith('invalid tool')) {
        state = ToolExecutionState.error;
      } else {
        state = ToolExecutionState.finished;
      }
    }

    tools.add(ToolExecutionRecord(
      toolName: toolName,
      query: query,
      state: state,
      snippet: snippet,
    ));
  }

  // Remove tool tags from displayed text
  var clean = text
      .replaceAll(toolCallExp, '')
      .replaceAll(toolResponseExp, '')
      .trim();

  return ToolParseResult(tools: tools, cleanText: clean);
}
