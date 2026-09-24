# VaultLM

[![Platform: Android](https://img.shields.io/badge/Platform-Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://github.com/U4mess/cross-platform-llm-client)
[![Version](https://img.shields.io/badge/Version-1.0.8-blueviolet?style=for-the-badge)](https://github.com/U4mess/cross-platform-llm-client/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A privacy-focused, cross-platform AI assistant built with Flutter. It unifies local on-device LLM inference (Android Vulkan-accelerated via `llama.cpp`) with cloud API access and autonomous agent capabilities.

![VaultLM Interface](PrivateLM.png)
_On-device local inference with dynamic hardware tier configuration and agent tooling_

![Image generation tested on mobile hardware](IMG_2390.png)
_On-device visual and text generation performance_

---

## Key Features

- **Local Inference on Android** — Download and run GGUF models directly on your phone using GPU-accelerated inference (Vulkan) via `llama.cpp`. Operates fully offline without network transmission.
- **Autonomous Micro-Agent Engine** — Equipped with built-in Dart tools for zero-API DuckDuckGo web searches, exact math evaluation, clipboard reading, and web page content scraping (`fetch_page_content`).
- **Dynamic Clock & Environment Grounding** — Real-time local date, day of the week, and device timestamps are automatically injected into inference sessions to prevent temporal hallucinations.
- **One-Tap Share Action Chips** — Direct integration with the Android Share sheet. Sharing text or links surfaces instant contextual chips: *Summarize*, *Proofread*, *Action Items*, *Translate*, and *Fetch & Summarize*.
- **Hardware-Aware Context Control** — Dynamic context size selection (2048, 4096, 8192 tokens) tailored to available RAM.
- **Cloud API Fallback** — Switch seamlessly to OpenAI, Anthropic Claude, Google Gemini, or Kimi (Moonshot AI) when larger parameter models are required.
- **Multimodal Support** — Multimodal vision conversations supported across both local models (such as Qwen2-VL) and cloud endpoints.
- **Local-First Storage** — Chat logs, task sessions, and configuration keys are stored on-device using Hive.

---

## Technical Architecture

### Stack

- **Framework:** Flutter 3.x (Dart >=3.3.0)
- **Local Engine:** `llama_flutter_android` (Native `llama.cpp` + Vulkan backend)
- **State Management:** GetX
- **Local Storage:** Hive
- **Networking:** Dio + `package:http`
- **Background Execution:** `flutter_background_service` + `flutter_local_notifications`

### Inference & Agent Pipeline
