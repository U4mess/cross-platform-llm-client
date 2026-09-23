import 'package:flutter/material.dart';
import '../../views/settings_view.dart';

export '../../views/settings_view.dart';

/// Compatibility alias for [SettingsView].
class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SettingsView();
  }
}
