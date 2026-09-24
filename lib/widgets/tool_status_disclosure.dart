import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

enum ToolExecutionState {
  executing,
  finished,
  error,
}

class ToolExecutionRecord {
  final String toolName;
  final String query;
  final ToolExecutionState state;
  final String? snippet;

  const ToolExecutionRecord({
    required this.toolName,
    required this.query,
    required this.state,
    this.snippet,
  });

  ToolExecutionRecord copyWith({
    String? toolName,
    String? query,
    ToolExecutionState? state,
    String? snippet,
  }) {
    return ToolExecutionRecord(
      toolName: toolName ?? this.toolName,
      query: query ?? this.query,
      state: state ?? this.state,
      snippet: snippet ?? this.snippet,
    );
  }
}

class ToolStatusDisclosure extends StatefulWidget {
  final ToolExecutionRecord record;

  const ToolStatusDisclosure({
    super.key,
    required this.record,
  });

  @override
  State<ToolStatusDisclosure> createState() => _ToolStatusDisclosureState();
}

class _ToolStatusDisclosureState extends State<ToolStatusDisclosure>
    with SingleTickerProviderStateMixin {
  late bool _expanded;
  late AnimationController _animController;
  late Animation<double> _expandAnimation;

  @override
  void initState() {
    super.initState();
    _expanded = false;
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 220),
      value: _expanded ? 1.0 : 0.0,
    );
    _expandAnimation = CurvedAnimation(
      parent: _animController,
      curve: Curves.easeOutCubic,
    );
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  void _toggle() {
    setState(() => _expanded = !_expanded);
    if (_expanded) {
      _animController.forward();
    } else {
      _animController.reverse();
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final record = widget.record;
    final isExecuting = record.state == ToolExecutionState.executing;
    final isError = record.state == ToolExecutionState.error;

    final Color statusColor;
    final Widget leadingWidget;
    final String label;

    if (isExecuting) {
      statusColor = isDark ? const Color(0xFF0A84FF) : const Color(0xFF007AFF);
      leadingWidget = Padding(
        padding: const EdgeInsets.only(right: 8),
        child: SizedBox(
          width: 12,
          height: 12,
          child: CircularProgressIndicator(
            strokeWidth: 1.5,
            color: statusColor,
          ),
        ),
      );
      final queryText = record.query.trim();
      label = queryText.isNotEmpty
          ? '🔍 Executing: ${record.toolName} ($queryText)...'
          : '🔍 Executing: ${record.toolName}...';
    } else if (isError) {
      statusColor = const Color(0xFFFF9500);
      leadingWidget = const Padding(
        padding: EdgeInsets.only(right: 6),
        child: Icon(
          Icons.warning_amber_rounded,
          size: 14,
          color: Color(0xFFFF9500),
        ),
      );
      label = '⚠ Tool failed: fallback to general inference';
    } else {
      statusColor = const Color(0xFF34C759);
      leadingWidget = const Padding(
        padding: EdgeInsets.only(right: 6),
        child: Icon(
          Icons.check_circle_outline_rounded,
          size: 14,
          color: Color(0xFF34C759),
        ),
      );
      label = '✓ Finished: ${record.toolName}';
    }

    final hasSnippet =
        record.snippet != null && record.snippet!.trim().isNotEmpty;
    final canExpand = !isExecuting && hasSnippet;

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: isDark
            ? Colors.white.withValues(alpha: 0.05)
            : Colors.black.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: statusColor.withValues(alpha: 0.25),
          width: 0.5,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header / Status Pill
          InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: canExpand ? _toggle : null,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  leadingWidget,
                  Flexible(
                    child: Text(
                      label,
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: statusColor,
                        fontWeight: FontWeight.w500,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (canExpand) ...[
                    const SizedBox(width: 4),
                    AnimatedRotation(
                      turns: _expanded ? 0.25 : 0.0,
                      duration: const Duration(milliseconds: 200),
                      child: Icon(
                        Icons.chevron_right_rounded,
                        size: 15,
                        color: statusColor.withValues(alpha: 0.7),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
          // Collapsible Snippet
          if (hasSnippet)
            SizeTransition(
              sizeFactor: _expandAnimation,
              axisAlignment: -1.0,
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: isDark
                        ? Colors.black.withValues(alpha: 0.3)
                        : Colors.white.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SelectableText(
                    record.snippet!.trim(),
                    style: GoogleFonts.firaCode(
                      fontSize: 11,
                      height: 1.4,
                      color: isDark ? Colors.white70 : Colors.black87,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
