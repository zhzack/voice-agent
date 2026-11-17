# VoiceStudio & Realtime Voice Agent

基于 **AMD Ryzen 7 9800X3D (CPU)** 与 **AMD Radeon RX 9070 XT (GPU 16GB)** 构建的完全本地化、纯离线、支持多大模型编排、多语音合成引擎与实时麦克风语音识别的高性能语音平台。

## ✨ 核心能力

- **🎙️ 实时语音转文字 (STT / Live Mic ASR)**：内置 `SenseVoice-Small INT8 + Silero VAD` 离线引擎，支持文件转写、单次定时录音识别、SSE/WebSocket 实时边录音边识别，以及本地终端一键监听。
- **🔊 多引擎语音合成 (TTS Studio)**：统一接入 `Edge-TTS`、`Kokoro`、`Piper`、`MeloTTS`、`ChatTTS`、`F5-TTS`（支持零样本声音克隆）。
- **🤖 多大模型编排 (LLM Hub)**：支持本地 `Qwen2.5`、本地 `Ollama (Qwen3.8-27B)` 与 `Cloudflare Workers AI`。
- **📞 全双工实时语音通话 (Realtime Call)**：完整打通 `VAD + SenseVoice ASR → LLM 流式生成 → TTS 切句合成`。
- **📊 自动化客观评测 (Benchmark Suite)**：基于标准语料库与 ASR 回测自动计算 CER / 准确率与 RTF 实时率。

---

## 📖 平台全景文档导航

本项目遵循清晰的分层工程架构，所有详细规范与文档均已归档于 `docs/` 目录：

- 🚀 **[平台快速上手与全景手册 (docs/README.md)](docs/README.md)**：包含 Web 控制台启动、多引擎配置、音色矩阵、实时通话及基准评测。
- 🏛️ **[系统分层架构设计说明书 (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)**：微内核插件架构、异构流水线、数据流与持久化详细技术规范。
- 📊 **[TTS 八大维度自动化基准评测规范 (docs/TTS_TEST_SPECIFICATION.md)](docs/TTS_TEST_SPECIFICATION.md)**：标准语料库与 ASR/CER 客观打分标准。

---

## ⚡ 极速启动

### 1. 安装依赖

```powershell
cd voice_agent
pip install -r requirements.txt
```

> **模型说明**：离线模型权重（约 800MB+，包含 `model.int8.onnx`、`tokens.txt`、`silero_vad.onnx` 以及 Kokoro/Piper/MeloTTS 权重）已通过 `.gitignore` 排除，请下载并放入 `models/` 目录。如需下载本地 Qwen3.8-27B GGUF，可运行 `python scripts/download_qwen27b.py`。

### 2. 启动 Web 控制台与 API 服务

```powershell
# 标准 FastAPI 启动 (支持热重载)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- 🌐 **Web 控制台地址**：`http://localhost:8000`
- 📖 **Swagger API 接口文档**：`http://localhost:8000/docs`

### 3. 麦克风实时边录音边识别

```powershell
# 终端一键调起本地麦克风实时识别
python live_mic_stt.py
```

常用 STT HTTP/WebSocket 接口：
- `GET /api/v1/stt/devices` — 查询可用麦克风设备列表
- `GET /api/v1/stt/mic/stream` — SSE 实时流式调起麦克风边录音边识别
- `POST /api/v1/stt/mic/record` — 定时录制指定秒数并返回文字
- `POST /api/v1/stt/transcribe` — 上传音频文件直接转文字
- `WS /api/v1/stt/ws` — WebSocket 实时双向流式识别

---

## 🧪 运行测试

```powershell
pytest tests
```

---

## 📂 核心目录架构

```text
voice_agent/
├── app/                  # 平台生产级核心架构 (API、Core、Providers、Services、Web Static)
├── assets/audio/         # 音色试听与音频测试样例 (*.wav)
├── benchmark_results/    # 基准评测报告 (*.md)
├── deploy/               # Ollama Modelfile 等本地部署配置
├── docs/                 # 系统架构、评测规范与核心技术文档
├── legacy/               # 历史独立 CLI 脚本与评测工具
├── model_plugins/        # 模型插件清单定义
├── models/               # 纯离线 ONNX / GGUF 模型存储目录 (gitignored)
├── scripts/              # 模型下载与环境初始化脚本
├── speakers/             # 预设音色样本库 (edge / 克隆参考音)
├── tests/                # FastAPI 接口与核心模块单元测试
├── live_mic_stt.py       # 麦克风实时边录音边识别 CLI 入口
└── requirements.txt      # Python 依赖清单
```
