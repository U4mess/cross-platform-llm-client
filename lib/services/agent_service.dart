import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'app_log_service.dart';

/// Lightweight mathematical expression parser and evaluator for basic math:
/// supports +, -, *, /, %, ^, parentheses, unary signs, and decimals.
class MathEvaluator {
  final String _input;
  int _pos = 0;

  MathEvaluator(this._input);

  static num evaluate(String expression) {
    final cleaned = expression.replaceAll(' ', '');
    if (cleaned.isEmpty) {
      throw const FormatException('Empty expression');
    }
    final evaluator = MathEvaluator(cleaned);
    final result = evaluator._parseExpression();
    if (evaluator._pos < evaluator._input.length) {
      throw FormatException('Unexpected character: ${evaluator._input[evaluator._pos]}');
    }
    return result;
  }

  num _parseExpression() {
    num value = _parseTerm();
    while (_pos < _input.length) {
      final op = _input[_pos];
      if (op == '+') {
        _pos++;
        value += _parseTerm();
      } else if (op == '-') {
        _pos++;
        value -= _parseTerm();
      } else {
        break;
      }
    }
    return value;
  }

  num _parseTerm() {
    num value = _parsePower();
    while (_pos < _input.length) {
      final op = _input[_pos];
      if (op == '*') {
        _pos++;
        value *= _parsePower();
      } else if (op == '/') {
        _pos++;
        final divisor = _parsePower();
        if (divisor == 0) {
          throw UnsupportedError('Division by zero');
        }
        value /= divisor;
      } else if (op == '%') {
        _pos++;
        final divisor = _parsePower();
        if (divisor == 0) {
          throw UnsupportedError('Modulo by zero');
        }
        value %= divisor;
      } else {
        break;
      }
    }
    return value;
  }

  num _parsePower() {
    num value = _parseFactor();
    if (_pos < _input.length && _input[_pos] == '^') {
      _pos++;
      final exponent = _parseFactor();
      value = math.pow(value, exponent);
    }
    return value;
  }

  num _parseFactor() {
    if (_pos >= _input.length) {
      throw const FormatException('Unexpected end of expression');
    }
    final ch = _input[_pos];
    if (ch == '+') {
      _pos++;
      return _parseFactor();
    }
    if (ch == '-') {
      _pos++;
      return -_parseFactor();
    }
    if (ch == '(') {
      _pos++;
      final val = _parseExpression();
      if (_pos >= _input.length || _input[_pos] != ')') {
        throw const FormatException('Missing closing parenthesis');
      }
      _pos++; // skip ')'
      return val;
    }

    final start = _pos;
    while (_pos < _input.length &&
        ((_input.codeUnitAt(_pos) >= 48 && _input.codeUnitAt(_pos) <= 57) ||
            _input[_pos] == '.')) {
      _pos++;
    }
    if (start == _pos) {
      throw FormatException('Expected number at position $start');
    }
    final numStr = _input.substring(start, _pos);
    final parsed = num.tryParse(numStr);
    if (parsed == null) {
      throw FormatException('Invalid number: $numStr');
    }
    return parsed;
  }
}

/// AgentService provides lightweight local and web tools for LLM agent execution.
class AgentService extends GetxService {
  /// Maximum sequential tool calls allowed per turn
  static const int maxToolCallsPerTurn = 2;

  /// Web search via DuckDuckGo Instant Answers API
  Future<String> webSearch(String query) async {
    final cleanQuery = query.trim();
    if (cleanQuery.isEmpty) return 'No query provided for web search.';

    try {
      final uri = Uri.parse(
        'https://api.duckduckgo.com/?q=${Uri.encodeComponent(cleanQuery)}&format=json&no_html=1',
      );
      final response = await http.get(uri, headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) PrivateLM/1.0',
        'Accept': 'application/json',
      }).timeout(const Duration(seconds: 10));

      if (response.statusCode != 200) {
        return 'Search error: HTTP ${response.statusCode}';
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final abstractText = (data['AbstractText'] as String?)?.trim() ?? '';
      final answer = (data['Answer'] as String?)?.trim() ?? '';
      final heading = (data['Heading'] as String?)?.trim() ?? '';

      if (answer.isNotEmpty) {
        return answer;
      }
      if (abstractText.isNotEmpty) {
        return heading.isNotEmpty ? '$heading: $abstractText' : abstractText;
      }

      final relatedTopics = data['RelatedTopics'] as List<dynamic>? ?? [];
      final topicTexts = <String>[];
      for (final topic in relatedTopics) {
        if (topic is Map<String, dynamic>) {
          final text = (topic['Text'] as String?)?.trim();
          if (text != null && text.isNotEmpty) {
            topicTexts.add(text);
          } else if (topic['Topics'] is List) {
            for (final subTopic in topic['Topics'] as List) {
              if (subTopic is Map<String, dynamic>) {
                final subText = (subTopic['Text'] as String?)?.trim();
                if (subText != null && subText.isNotEmpty) {
                  topicTexts.add(subText);
                }
              }
            }
          }
        }
        if (topicTexts.length >= 4) break;
      }

      if (topicTexts.isNotEmpty) {
        return topicTexts.join('\n\n');
      }

      return 'No direct answers found for "$cleanQuery".';
    } catch (e) {
      Get.find<AppLogService>().warning('AgentService web_search error', details: e);
      return 'Search failed: $e';
    }
  }

  /// Evaluates simple math expressions (+, -, *, /, parentheses)
  String calculate(String expression) {
    final cleanExpr = expression.trim();
    if (cleanExpr.isEmpty) return 'Error: Empty expression.';

    try {
      final result = MathEvaluator.evaluate(cleanExpr);
      if (result is double &&
          result == result.roundToDouble() &&
          !result.isInfinite &&
          !result.isNaN) {
        return result.toInt().toString();
      }
      return result.toString();
    } catch (e) {
      return 'Calculation error: $e';
    }
  }

  /// Reads text from the system clipboard
  Future<String> readClipboard() async {
    try {
      final data = await Clipboard.getData(Clipboard.kTextPlain);
      final text = data?.text?.trim();
      if (text == null || text.isEmpty) {
        return 'Clipboard is empty.';
      }
      return text;
    } catch (e) {
      Get.find<AppLogService>().warning('AgentService read_clipboard error', details: e);
      return 'Failed to read clipboard: $e';
    }
  }

  /// Dispatches a tool execution by name and arguments
  Future<String> executeTool(String name, Map<String, dynamic> args) async {
    final normalized = name.toLowerCase().trim();
    switch (normalized) {
      case 'web_search':
      case 'websearch':
      case 'search':
        final query = (args['query'] ?? args['q'] ?? args['input'] ?? '').toString();
        return await webSearch(query);
      case 'calculate':
      case 'calculator':
      case 'calc':
        final expr = (args['expression'] ?? args['expr'] ?? args['input'] ?? '').toString();
        return calculate(expr);
      case 'read_clipboard':
      case 'readclipboard':
      case 'clipboard':
        return await readClipboard();
      default:
        return 'Unknown tool: $name';
    }
  }

  /// Parses a JSON string tool call and executes it
  Future<String> executeToolCallJson(String jsonString) async {
    try {
      final clean = jsonString.trim();
      final parsed = jsonDecode(clean);
      if (parsed is Map<String, dynamic>) {
        final name = (parsed['name'] ?? '').toString();
        return await executeTool(name, parsed);
      }
      return 'Invalid tool call format: expected JSON object';
    } catch (e) {
      return 'Invalid tool call JSON: $e';
    }
  }
}
