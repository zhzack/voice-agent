# VoiceStudio V3.0 最终架构设计方案

**项目名称：** VoiceStudio  
**文档类型：** 系统架构设计说明书（Architecture Design Document）  
**版本：** V3.0  
**设计目标：** 本地化、多模型、插件化 AI 语音实验与制作平台  
**硬件基准：** AMD Ryzen 7 9800X3D + AMD Radeon RX 9070 XT 16GB  
**技术方向：** Vue 3 + FastAPI + Python + PyTorch/ROCm + SQLite/PostgreSQL  
**部署原则：** 本地优先、模型环境隔离、GPU 统一调度、模块低耦合、可扩展

---

# 1. 项目定位

## 1.1 项目目标

VoiceStudio 是一个面向本地 AI 语音模型的综合管理与实验平台。

平台以 TTS（Text-to-Speech，文本转语音）为核心，同时提供 STT（语音转文本）、LLM（大语言模型）、音频处理、音色管理、模型评测、有声书制作和实时语音交互等能力。

平台不是简单的语音合成接口集合，而是一个能够持续接入不同厂商、不同架构、不同运行环境的语音模型实验平台。

核心目标：

1. 支持大量不同类型的 TTS 模型接入。
2. 支持模型安装、配置、启动、停止、卸载和版本管理。
3. 支持 CPU、AMD GPU（ROCm）、其他推理后端及可选云端服务。
4. 支持不同模型使用同一文本、相同参考音频进行公平对比。
5. 支持试听、盲听、人工评分和自动化评测。
6. 支持人物音色档案和跨模型音色对照。
7. 支持小说、有声书、多人角色配音和章节管理。
8. 支持 LLM 自动分析文本情绪、语气、人物和表达方式。
9. 支持用户修改 LLM 分析结果，并将其转换为具体模型的控制参数。
10. 支持长期保存模型实验记录、生成音频和评测报告。

## 1.2 设计原则

### 原则一：微内核与插件化

平台核心不直接依赖具体 TTS 模型。

所有模型通过统一的 Provider 接口接入，模型自身的专属功能通过能力扩展接口提供。

增加新模型时，尽量只需要增加：
- 模型适配器
- 模型描述文件
- 独立运行环境
- 必要的安装和启动配置
- 测试用例

不应该因为增加一个模型就修改平台核心业务逻辑。

### 原则二：模型与运行环境分离

模型权重、模型适配代码、运行环境、模型运行实例分别管理。

不能把所有模型的依赖都安装在同一个 Python 环境中。

### 原则三：统一管理，不强制统一所有参数

所有 TTS 模型拥有统一的基础调用协议，但模型专属能力必须保留。

例如：
- Qwen3-TTS 的声音设计指令。
- F5-TTS 的参考音频和参考文本。
- IndexTTS 的情绪控制参数。
- CosyVoice 的指令控制。
- Fish Speech 的文本表达标记。

不能为了统一接口而丢弃这些能力。

### 原则四：实验必须可复现

每次实验都必须保存：
- 模型名称及版本。
- 权重版本和文件标识。
- 推理环境。
- 硬件信息。
- 生成参数。
- 输入文本。
- 参考音频及其哈希。
- 输出音频。
- 性能指标。
- 评分结果。

### 原则五：本地优先

本地模型应当是默认运行方式。

云端 API 作为可选 Provider，必须明确标识网络依赖、数据传输范围和服务费用。

---

# 2. 系统总体架构

## 2.1 总体分层

系统分为六个主要层次。

```text
┌──────────────────────────────────────────────────────────┐
│                    前端表现层                            │
│                                                          │
│ Dashboard / Model Market / TTS Playground                │
│ Benchmark Lab / Voice Library / Audiobook Studio         │
│ LLM Hub / STT Lab / Realtime Call / History              │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    API 接入层                             │
│                                                          │
│ FastAPI / REST API / WebSocket / 文件上传 / 任务状态      │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    核心业务层                             │
│                                                          │
│ Model Registry       Runtime Manager                     │
│ Task Manager         GPU Scheduler                       │
│ Voice Manager        Benchmark Engine                    │
│ Audiobook Manager    History Manager                     │
│ LLM Orchestrator     Audio Processing                    │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    模型适配层                             │
│                                                          │
│ Base TTS Provider / Base STT Provider / Base LLM Provider│
│ Model Manifest / Capability Registry / Parameter Mapping │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    模型运行层                             │
│                                                          │
│ Qwen3-TTS / CosyVoice / IndexTTS / MOSS / Fish Speech    │
│ F5-TTS / VoxCPM2 / GLM-TTS / Piper / Kokoro / ...        │
│                                                          │
│ CPU Runtime / ROCm Runtime / Other Runtime / Cloud       │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    数据存储层                             │
│                                                          │
│ SQLite / PostgreSQL / Model Storage / Audio Storage      │
│ Benchmark Datasets / Logs / Configuration / Reports      │
└──────────────────────────────────────────────────────────┘
```

## 2.2 主要业务模块

| 模块 | 核心职责 |
|---|---|
| Dashboard | 系统状态、CPU、GPU、内存、任务状态 |
| Model Market | 模型发现、安装、版本、许可证和兼容性 |
| TTS Playground | 单模型调音和快速试听 |
| Benchmark Lab | 多模型对照、批量测试、盲听和评分 |
| Voice Library | 人物档案、音色、参考音频管理 |
| Audiobook Studio | 小说导入、章节、角色和音频制作 |
| LLM Hub | 本地大模型、提示词和模型编排 |
| STT Lab | 语音识别和自动转写 |
| Realtime Call | 实时语音对话和打断 |
| History Lab | 实验历史、生成记录和收藏 |
| System Settings | 运行环境、路径、资源和系统配置 |

---

# 3. 模型插件系统

这是 V3.0 最核心的架构改造。

## 3.1 模型分类

模型按任务类型注册，而不是全部放在 TTS 目录中。

```text
Model Registry
│
├── TTS
│   ├── Traditional TTS
│   ├── Flow Matching TTS
│   ├── Autoregressive TTS
│   ├── Voice Cloning
│   ├── Voice Design
│   └── Multi-Speaker TTS
│
├── STT
│   ├── ASR
│   ├── Streaming ASR
│   └── Alignment
│
├── LLM
│   ├── Text LLM
│   ├── Multimodal LLM
│   └── Emotion Analysis
│
└── AUDIO
    ├── VAD
    ├── Denoise
    ├── Separation
    └── Audio Evaluation
```

## 3.2 三个核心对象

必须严格区分：

**Model：模型**

例如：
- Qwen3-TTS 1.7B
- CosyVoice 3
- IndexTTS-2.5

**Runtime：运行实例**

模型实际运行时的实例。

一个模型可以有多个运行配置，但是否能同时加载必须经过资源检查。

**Voice：音色**

模型可使用的声音资源，包括预设音色、参考音频、克隆音色和声音设计配置。

一个人物可以对应多个模型的音色实现。

## 3.3 模型 Manifest

每个模型必须拥有一份描述文件，供 Model Registry 读取。

示例：

```yaml
schema_version: "1.0"

id: qwen3-tts-1.7b
name: Qwen3-TTS 1.7B
provider: qwen3
task: tts

version:
  adapter: "1.0.0"
  model: "1.7b"
  upstream_revision: null

source:
  repository: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  local_path: "models/qwen3-tts-1.7b"

runtime:
  backend: pytorch
  device: auto
  precision: auto
  environment: qwen3-tts
  startup_command: "python serve.py"

capabilities:
  languages:
    - zh
    - en

  voice_cloning: true
  voice_design: true
  emotion_control: true
  speed_control: true
  streaming: true
  multi_speaker: false
  reference_audio: true
  reference_text: true

parameters:
  temperature:
    type: float
    default: 0.9
    min: 0.1
    max: 2.0

license:
  code_license: Apache-2.0
  weight_license: verify
  commercial_use: unknown

hardware:
  supported_devices:
    - cpu
    - rocm
  minimum_vram_mb: null

status:
  experimental: true
```

说明：

- 上述参数只是 Manifest 结构示例。
- 真实参数必须由具体模型适配器定义。
- 不支持的能力不能伪装成支持。
- `null` 表示尚未验证，不代表无限制或不需要资源。
- 许可证必须分别记录代码和模型权重。

## 3.4 能力描述体系

所有模型需要提供标准能力描述。

| 能力 | 说明 |
|---|---|
| languages | 支持语言 |
| voice_cloning | 是否支持声音克隆 |
| voice_design | 是否支持自然语言设计音色 |
| emotion_control | 是否支持情绪控制 |
| speed_control | 是否支持语速控制 |
| pitch_control | 是否支持音调控制 |
| streaming | 是否支持流式输出 |
| multi_speaker | 是否支持多角色 |
| reference_audio | 是否支持参考音频 |
| reference_text | 是否需要或支持参考文本 |
| pronunciation_control | 是否支持发音控制 |
| duration_control | 是否支持时长控制 |

能力分为三种状态：

- `supported`：已确认支持。
- `unsupported`：明确不支持。
- `unknown`：尚未验证。

前端根据能力动态生成控件。

## 3.5 Provider 抽象接口

基础接口：

```python
class BaseTTSProvider:

    async def initialize(self, config):
        """初始化模型环境和配置"""
        raise NotImplementedError

    async def load_model(self):
        """加载模型"""
        raise NotImplementedError

    async def unload_model(self):
        """卸载模型"""
        raise NotImplementedError

    async def health_check(self):
        """检查模型运行状态"""
        raise NotImplementedError

    async def get_capabilities(self):
        """返回模型能力"""
        raise NotImplementedError

    async def get_parameters(self):
        """返回模型参数定义"""
        raise NotImplementedError

    async def synthesize(self, request):
        """执行语音合成"""
        raise NotImplementedError

    async def cancel(self, task_id):
        """取消任务"""
        raise NotImplementedError
```

模型扩展接口：

```python
class VoiceDesignCapability:
    async def design_voice(self, request):
        raise NotImplementedError


class VoiceCloneCapability:
    async def clone_voice(self, request):
        raise NotImplementedError


class EmotionCapability:
    async def configure_emotion(self, request):
        raise NotImplementedError


class StreamingCapability:
    async def stream_synthesize(self, request):
        raise NotImplementedError
```

并不是每个模型都必须实现全部扩展接口。

平台通过能力注册表判断具体模型支持什么。

## 3.6 模型插件目录

```text
model_plugins/
│
├── tts/
│   ├── qwen3/
│   │   ├── manifest.yaml
│   │   ├── provider.py
│   │   ├── mapper.py
│   │   ├── runtime.py
│   │   └── tests/
│   │
│   ├── cosyvoice/
│   ├── indextts/
│   ├── moss/
│   ├── fish_speech/
│   ├── voxcpm/
│   ├── glm_tts/
│   ├── fireredtts/
│   ├── f5/
│   ├── spark_tts/
│   ├── chattts/
│   ├── kokoro/
│   ├── melo/
│   └── piper/
│
├── stt/
│   ├── sensevoice/
│   ├── whisper/
│   └── funasr/
│
└── llm/
    ├── transformers/
    └── openai_compatible/
```

新增模型不得要求修改核心调度器。

---

# 4. 模型运行环境管理

## 4.1 设计目标

模型依赖彼此隔离。

例如：

- Qwen3-TTS 使用独立 Python 环境。
- CosyVoice 使用独立 Python 环境。
- IndexTTS 使用独立 Python 环境。
- Fish Speech 使用独立运行环境。
- CPU 模型可以共享经过验证的基础环境。

模型运行环境不能由主 FastAPI 进程直接导入所有模型代码。

## 4.2 运行方式

平台支持三种运行方式：

| 类型 | 适用场景 |
|---|---|
| Local Process | 轻量模型、开发调试 |
| Isolated Worker | GPU 模型和复杂依赖 |
| Container Runtime | 可容器化且依赖隔离的模型 |

优先采用独立 Worker 进程或容器，通过内部 HTTP API 或本地 IPC 与核心服务通信。

统一运行接口：

```text
POST   /runtime/start
POST   /runtime/stop
POST   /runtime/restart
GET    /runtime/status
GET    /runtime/health
GET    /runtime/capabilities
POST   /runtime/unload
```

运行状态：

```text
NOT_INSTALLED
INSTALLED
STOPPED
STARTING
LOADING
READY
BUSY
UNLOADING
ERROR
```

## 4.3 模型安装与卸载

Model Market 应支持：

1. 查看模型详情。
2. 查看模型来源。
3. 查看模型许可证。
4. 查看模型文件大小。
5. 查看 AMD GPU 兼容性。
6. 下载模型权重。
7. 安装模型运行环境。
8. 检查依赖。
9. 加载模型。
10. 卸载模型。
11. 删除模型权重。
12. 更新模型版本。

删除模型权重与删除运行环境必须是两个独立操作。

平台必须支持模型下载断点续传、文件完整性校验和失败恢复。

---

# 5. 硬件资源管理与 GPU 调度

## 5.1 硬件基准

| 硬件 | 规格 |
|---|---|
| CPU | Ryzen 7 9800X3D |
| GPU | Radeon RX 9070 XT |
| GPU 架构 | RDNA 4 / gfx1201 |
| VRAM | 16GB |
| GPU 后端 | AMD ROCm |
| CPU 线程 | 16 线程 |

平台不能把显存当作普通内存使用。

## 5.2 资源分配原则

CPU 适合承担：

- Web 服务。
- 音频转码。
- 任务调度。
- 轻量 TTS。
- 轻量 STT。
- 数据库操作。
- 音频拼接。

GPU 适合承担：

- 重型 TTS。
- 大语言模型。
- 高质量声音克隆。
- 复杂语音生成。

实际分配由模型兼容性测试和资源监控决定，不在配置中写死性能数值。

## 5.3 GPU Scheduler

GPU Scheduler 负责：

- 查询显存使用量。
- 检查模型加载需求。
- 控制 GPU 任务并发。
- 排队执行任务。
- 控制模型加载和卸载。
- 处理显存不足。
- 处理任务取消。
- 记录实际 GPU 资源使用情况。

建议初始采用单 GPU 独占调度：

```text
                 GPU Scheduler
                       │
               Resource Manager
                       │
                 GPU Task Queue
                       │
             ┌─────────┴─────────┐
             │                   │
         TTS Task             LLM Task
             │                   │
             └─────────┬─────────┘
                       │
                 GPU Worker
                       │
                RX 9070 XT
```

默认不允许两个未经验证的重型模型同时占用 GPU。

后续可以增加模型并行和显存预算功能，但必须经过真实压力测试。

## 5.4 显存管理策略

支持：

- 模型加载前显存检查。
- 模型运行期间显存监控。
- 空闲模型自动卸载。
- GPU 任务优先级。
- GPU 内存不足告警。
- CPU 推理降级（仅限模型支持时）。
- 模型切换前检查当前任务。

16GB 显存下建议优先采用一个重型 GPU 模型运行，完成任务后再切换模型。

模型的最低显存要求必须通过实机测试获得，不根据参数量简单推断。

---

# 6. TTS 统一接口设计

## 6.1 基础请求协议

```json
{
  "model_id": "qwen3-tts-1.7b",
  "text": "我真的很高兴见到你。",
  "voice_profile_id": "voice_female_001",
  "language": "zh",
  "output_format": "wav",
  "generation_config": {},
  "model_config": {}
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| model_id | 模型标识 |
| text | 输入文本 |
| voice_profile_id | 人物音色档案 |
| language | 语言 |
| output_format | 输出格式 |
| generation_config | 通用生成参数 |
| model_config | 模型专属参数 |

## 6.2 生成参数分层

第一层：通用参数

```json
{
  "speed": 1.0,
  "volume": 1.0,
  "seed": null
}
```

第二层：音色条件

```json
{
  "reference_audio_id": "audio_ref_001",
  "reference_text": "参考音频对应的文本",
  "voice_prompt": "成熟、温柔、略带磁性的女声"
}
```

第三层：表达控制

```json
{
  "emotion": "sad",
  "emotion_intensity": 0.7,
  "speaking_style": "narration",
  "pause_control": {}
}
```

第四层：模型专属参数

```json
{
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9
}
```

以上参数结构是平台统一数据格式示例。具体模型是否支持，需要由 Provider 映射和验证。

## 6.3 参数映射机制

Provider 负责把统一参数映射到模型原生参数。

```text
Unified Request
      │
      ▼
Parameter Validator
      │
      ▼
Capability Checker
      │
      ▼
Model Parameter Mapper
      │
      ▼
Native Model Request
      │
      ▼
TTS Model
```

如果用户选择了模型不支持的参数：

- 明确标记不支持。
- 不允许静默伪造参数效果。
- 如果支持近似映射，必须标记为 approximate。
- 记录最终实际传递给模型的参数。

## 6.4 统一生成结果

```json
{
  "task_id": "tts_task_001",
  "status": "completed",
  "model_id": "qwen3-tts-1.7b",
  "audio": {
    "file_id": "audio_001",
    "format": "wav",
    "sample_rate": 24000,
    "duration": 8.4
  },
  "performance": {
    "generation_time": 2.8,
    "rtf": 0.333,
    "peak_vram_mb": 8500
  },
  "effective_config": {},
  "warnings": []
}
```

---

# 7. TTS Playground

## 7.1 功能

Playground 是单模型实验区。

主要功能：

- 选择模型。
- 选择模型运行实例。
- 选择音色。
- 输入文本。
- 设置通用参数。
- 设置模型专属参数。
- 上传参考音频。
- 录制参考音频。
- 生成语音。
- 播放和下载。
- 查看生成性能。
- 保存实验记录。
- 将当前配置保存为预设。

## 7.2 动态参数界面

前端不能针对每个模型写一套固定表单。

参数控件由 Provider 返回的参数定义动态生成。

支持控件类型：

- String
- Textarea
- Number
- Slider
- Boolean
- Select
- Audio Upload
- Audio Recorder
- JSON Editor

每个参数需要提供：

- 名称。
- 描述。
- 类型。
- 默认值。
- 最小值。
- 最大值。
- 可选值。
- 是否必填。
- 是否支持动态修改。

## 7.3 实验预设

用户可以保存：

```text
preset/
├── chinese_narration.json
├── emotional_dialogue.json
├── voice_cloning.json
└── audiobook_female.json
```

预设包含模型、音色、生成参数和测试文本，但不能保存无法复现的临时运行状态。

---

# 8. Benchmark 多模型评测中心

这是 VoiceStudio 的核心功能之一。

## 8.1 实验模式

支持两种比较模式：

**模式 A：相同音色条件**

所有模型使用：
- 相同目标文本。
- 相同参考音频。
- 相同参考文本（如果模型需要）。
- 相同语言。
- 尽可能相同的音频输出规格。

每个模型仍保留必要的原生参数。

**模式 B：各模型最佳效果**

允许每个模型使用不同的原生参数、情绪指令和声音设计方式。

两种模式必须独立统计，不能把结果放在同一个排行榜中。

## 8.2 首批实验模型

第一批：

1. Qwen3-TTS 1.7B。
2. CosyVoice 3。
3. IndexTTS-2.5。

第二批：

1. MOSS-TTS。
2. F5-TTS。
3. VoxCPM2。
4. Fish Audio S2-Pro。
5. GLM-TTS。
6. FireRedTTS-2。

第三批：

1. Kokoro。
2. Piper。
3. MeloTTS。
4. ChatTTS。
5. Spark-TTS。

模型是否正式进入可运行列表，必须经过实际安装和兼容性测试。

## 8.3 测试任务分类

| 分类 | 内容 |
|---|---|
| daily | 日常口语 |
| narration | 小说旁白 |
| dialogue | 人物对话 |
| emotion | 情绪表达 |
| style | 语气变化 |
| pronunciation | 中文发音 |
| numbers | 日期、数字和金额 |
| multilingual | 多语言和中英混读 |
| punctuation | 停顿和断句 |
| longtext | 长文本 |
| consistency | 音色一致性 |
| multi_speaker | 多角色对话 |

## 8.4 标准实验流程

```text
创建实验
   │
   ▼
选择参与模型
   │
   ▼
选择测试集
   │
   ▼
设置统一参考音频
   │
   ▼
检查模型能力和兼容性
   │
   ▼
生成各模型的参数配置
   │
   ▼
创建生成任务
   │
   ▼
GPU Scheduler 排队执行
   │
   ▼
保存生成音频和性能指标
   │
   ▼
执行 ASR 和自动化检测
   │
   ▼
随机化音频顺序并隐藏模型名称
   │
   ▼
人工盲听评分
   │
   ▼
生成实验报告
```

## 8.5 公平性控制

必须记录：

- 参考音频哈希。
- 参考文本。
- 目标文本。
- 模型版本。
- 模型原生参数。
- 生成随机种子（如果支持）。
- 音频采样率。
- 音频声道数。
- 音频格式。
- 后处理配置。

特别注意：

不同模型不一定能用同一套参数实现完全相同的音色条件。

因此本实验评价的是：

**相同参考条件下，各模型实现目标声音的能力。**

不能把它解释为三个模型拥有完全相同的音色。

## 8.6 自动化评分

自动化指标：

| 指标 | 说明 |
|---|---|
| CER | 字符错误率 |
| WER | 词错误率，适用于相应语言 |
| RTF | 生成时间 / 音频时长 |
| TTFA | 首个音频片段到达时间 |
| Generation Time | 完整生成耗时 |
| Peak VRAM | GPU 峰值显存 |
| Peak RAM | CPU 内存峰值 |
| Failure Rate | 任务失败率 |

性能指标必须记录测试环境和测量方法。

## 8.7 人工评分

建议初始权重：

| 评分项目 | 权重 |
|---|---:|
| 发音准确性 | 20% |
| 自然度 | 20% |
| 情绪表现 | 20% |
| 音色一致性 | 15% |
| 断句和韵律 | 15% |
| 生成速度和稳定性 | 10% |

每项采用 1～5 分制。

总分：

```text
Total Score =
    Pronunciation × 20%
  + Naturalness × 20%
  + Emotion × 20%
  + Voice Consistency × 15%
  + Prosody × 15%
  + Performance × 10%
```

评分规则：

- 1：非常差。
- 2：较差。
- 3：合格。
- 4：优秀。
- 5：非常优秀。

所有评分必须支持按测试场景配置权重。

对于小说旁白，可以提高自然度、音色一致性和韵律权重。

对于实时语音交互，可以提高延迟和稳定性权重。

## 8.8 盲听功能

系统生成音频后：

1. 随机生成试听编号。
2. 隐藏模型名称。
3. 随机化播放顺序。
4. 记录用户评分。
5. 完成评分后才允许揭示模型信息。

需要防止同一实验的模型名称通过文件名、音频元数据或界面提示泄露。

## 8.9 实验报告

报告应包含：

- 实验基本信息。
- 参与模型。
- 模型版本。
- 测试文本。
- 参考音频。
- 生成参数。
- 生成音频。
- 性能数据。
- CER/WER。
- 人工评分。
- 综合得分。
- 用户评语。
- 模型优缺点。
- 实验环境。
- 可复现配置。

支持 JSON、CSV、Markdown 和 HTML 导出。

---

# 9. Voice Library 音色管理中心

## 9.1 核心设计

音色管理必须独立于具体模型。

平台中的人物档案是统一的人物定义，而不同模型的实际音色实现通过映射表关联。

例如：

```text
人物：小说女主角
│
├── 人物基本信息
│   ├── 性别设定
│   ├── 年龄感
│   ├── 声音特征
│   ├── 语言
│   └── 表达风格
│
├── Qwen3-TTS 实现
│   └── 声音设计提示词
│
├── CosyVoice 实现
│   └── 参考音频和模型配置
│
├── IndexTTS 实现
│   └── 参考音频和情绪配置
│
└── F5-TTS 实现
    └── 参考音频和参考文本
```

## 9.2 音色数据

每个 Voice Profile 包含：

- 人物名称。
- 人物描述。
- 参考音频。
- 参考文本。
- 语言。
- 声音特征。
- 适用场景。
- 音色来源。
- 授权信息。
- 创建时间。
- 版本。

## 9.3 音色授权

上传或克隆声音时必须记录来源及授权状态。

不得默认用户上传的音频就允许用于任何声音克隆用途。

支持：
- 自有声音。
- 已获得授权的声音。
- 授权范围有限的声音。
- 未验证来源的参考音频。

未验证授权状态的音色不应被默认用于公开分享或商业导出。

---

# 10. Audiobook Studio 有声书工作台

## 10.1 目标

支持从小说文本到完整有声书的制作流程。

## 10.2 主要功能

- TXT、Markdown、DOCX 导入。
- 章节识别。
- 段落切分。
- 对话识别。
- 人物识别。
- 角色音色分配。
- 旁白音色分配。
- 情绪分析。
- 用户修改情绪和语气。
- 批量 TTS 生成。
- 音频试听和重生成。
- 章节音频拼接。
- 最终音频导出。

## 10.3 LLM 情绪分析流程

```text
导入小说
   │
   ▼
章节解析
   │
   ▼
段落切分
   │
   ▼
人物和说话人识别
   │
   ▼
上下文分析
   │
   ▼
情绪与语气分析
   │
   ▼
生成结构化标注
   │
   ▼
用户检查和修改
   │
   ▼
模型能力映射
   │
   ▼
TTS 生成
   │
   ▼
人工试听与修订
```

## 10.4 情绪标注结构

示例：

```json
{
  "speaker_id": "character_001",
  "text": "你真的决定要离开了吗？",
  "emotion": "sad",
  "emotion_intensity": 0.75,
  "speaking_style": "dialogue",
  "delivery": "soft",
  "speed": 0.9,
  "pause_before_ms": 300,
  "pause_after_ms": 500,
  "confidence": 0.92
}
```

这里的情绪标签是平台内部统一表达，不意味着每个模型都支持全部字段。

## 10.5 用户修改优先级

参数优先级：

```text
用户手动修改
      ↓
用户预设
      ↓
LLM 自动分析
      ↓
模型默认参数
```

用户手动修改必须覆盖自动分析结果。

每次修改都要保留历史版本，便于撤销和对比。

## 10.6 长文本处理

长篇内容不能简单地一次性提交给 TTS。

需要：

1. 章节切分。
2. 段落切分。
3. 文本长度检查。
4. 上下文保留。
5. 角色状态继承。
6. 分段生成。
7. 片段试听。
8. 音频拼接。
9. 章节导出。

同一人物跨章节必须复用同一人物音色配置，避免无意间更换音色。

---

# 11. LLM Hub

## 11.1 主要功能

- 本地 LLM 管理。
- 可选云端 LLM。
- 模型切换。
- 提示词管理。
- 文本分析。
- 人物识别。
- 情绪分析。
- 结构化输出。
- 对话历史。

## 11.2 LLM 与 TTS 解耦

LLM 不直接调用具体 TTS 模型。

LLM 负责生成结构化表达意图。

核心服务负责：
1. 校验输出。
2. 读取模型能力。
3. 将统一情绪和语气转换为模型原生参数。
4. 提交 TTS 任务。

这样更换 LLM 不会影响 TTS Provider。

## 11.3 结构化输出

要求 LLM 返回符合 JSON Schema 的结果。

输出必须经过校验，不能直接将未经验证的 LLM 文本作为模型参数。

---

# 12. STT 与音频处理模块

## 12.1 STT

支持：

- SenseVoice。
- Whisper。
- FunASR。
- 其他兼容模型。

STT 统一接口支持：

- 文件转写。
- 实时转写。
- 语言检测。
- 时间戳。
- 说话人分离（如果底层模型支持）。
- 字幕输出。

## 12.2 音频处理

独立 Audio Processing Service，负责：

- 格式转换。
- 采样率转换。
- 声道转换。
- 音量标准化。
- 静音检测。
- 音频拼接。
- 淡入淡出。
- 静音裁剪。
- 降噪（可选）。
- 音频质量检查。

不能在模型 Provider 中重复实现通用音频处理逻辑。

所有后处理参数必须记录，保证 Benchmark 能够区分原始音频与处理后的音频。

---

# 13. 实时语音交互模块

## 13.1 功能

保留 V2.0 中的全双工语音交互能力。

支持：

- 浏览器麦克风。
- VAD。
- STT。
- LLM。
- TTS。
- 流式播放。
- Barge-in。
- 对话历史。

## 13.2 独立业务流程

实时语音通话与有声书制作共享底层模型服务，但不共享业务状态机。

实时通话需要单独的 Session Manager。

建议状态：

```text
IDLE
LISTENING
RECORDING
TRANSCRIBING
THINKING
SPEAKING
INTERRUPTED
CLOSED
ERROR
```

用户打断时必须能够取消当前生成任务，而不只是停止浏览器播放。

---

# 14. 数据库设计

## 14.1 数据库策略

初期使用 SQLite，适合单用户本地部署。

数据库访问层采用 SQLAlchemy，避免业务代码直接依赖数据库类型。

未来迁移 PostgreSQL 时，不应重写业务逻辑。

音频文件和模型权重不直接存入数据库。

数据库只保存文件 ID、路径、哈希和元数据。

## 14.2 核心数据表

### 模型管理

| 表名 | 主要字段 |
|---|---|
| model_catalog | id、name、provider、task、description |
| model_versions | id、model_id、version、revision、license |
| model_artifacts | id、version_id、path、size、hash |
| model_capabilities | id、model_id、capability、status |
| runtime_instances | id、model_id、status、device、config |
| runtime_events | id、runtime_id、event、timestamp |
| provider_configs | id、model_id、config_json |

### 音色管理

| 表名 | 主要字段 |
|---|---|
| voice_profiles | id、name、description、language |
| voice_assets | id、voice_id、file_id、asset_type |
| voice_model_bindings | id、voice_id、model_id、config_json |
| voice_versions | id、voice_id、version、created_at |

### TTS 生成

| 表名 | 主要字段 |
|---|---|
| generation_jobs | id、model_id、status、created_at |
| generation_inputs | id、job_id、text、voice_id |
| generation_outputs | id、job_id、audio_id、duration |
| generation_configs | id、job_id、config_json |
| generation_metrics | id、job_id、rtf、peak_vram、latency |

### Benchmark

| 表名 | 主要字段 |
|---|---|
| benchmark_suites | id、name、description |
| benchmark_cases | id、suite_id、text、category |
| benchmark_runs | id、name、mode、status |
| benchmark_participants | id、run_id、model_id、version_id |
| benchmark_samples | id、run_id、case_id、participant_id、audio_id |
| benchmark_metrics | id、sample_id、metric、value |
| human_ratings | id、sample_id、rater_id、score_json |
| blind_listen_sessions | id、run_id、blind_mapping_json |
| benchmark_reports | id、run_id、report_path |

### 有声书

| 表名 | 主要字段 |
|---|---|
| audiobook_projects | id、name、description |
| audiobook_chapters | id、project_id、title、order |
| audiobook_segments | id、chapter_id、text、speaker_id |
| audiobook_characters | id、project_id、name、voice_id |
| audiobook_annotations | id、segment_id、emotion_json |
| audiobook_audio_assets | id、segment_id、audio_id |
| audiobook_exports | id、project_id、file_id、format |

### 系统与历史

| 表名 | 主要字段 |
|---|---|
| app_settings | key、value、updated_at |
| operation_logs | id、module、operation、result |
| system_metrics | id、cpu、ram、vram、timestamp |
| audio_files | id、path、hash、format、duration |
| task_events | id、task_id、event、timestamp |

所有表需要统一主键策略、创建时间、更新时间和软删除策略。

对重要实验记录，必须保存不可变的模型版本和配置快照。

---

# 15. REST API 设计

统一前缀：

```text
/api/v1
```

## 15.1 模型管理

```text
GET    /models
GET    /models/{model_id}
GET    /models/{model_id}/capabilities
GET    /models/{model_id}/versions
POST   /models/install
DELETE /models/{model_id}
POST   /models/{model_id}/load
POST   /models/{model_id}/unload
GET    /models/{model_id}/runtime
```

## 15.2 TTS

```text
POST   /tts/synthesize
GET    /tts/tasks/{task_id}
POST   /tts/tasks/{task_id}/cancel
GET    /tts/tasks/{task_id}/audio
GET    /tts/history
```

## 15.3 音色

```text
GET    /voices
POST   /voices
GET    /voices/{voice_id}
PUT    /voices/{voice_id}
DELETE /voices/{voice_id}
POST   /voices/{voice_id}/assets
GET    /voices/{voice_id}/models
```

## 15.4 Benchmark

```text
GET    /benchmark/suites
POST   /benchmark/suites
POST   /benchmark/cases
GET    /benchmark/cases

POST   /benchmark/runs
GET    /benchmark/runs
GET    /benchmark/runs/{run_id}
POST   /benchmark/runs/{run_id}/start
POST   /benchmark/runs/{run_id}/cancel

GET    /benchmark/runs/{run_id}/samples
POST   /benchmark/samples/{sample_id}/rating
GET    /benchmark/runs/{run_id}/report
```

## 15.5 有声书

```text
GET    /audiobooks
POST   /audiobooks
GET    /audiobooks/{project_id}
POST   /audiobooks/{project_id}/import
GET    /audiobooks/{project_id}/chapters
POST   /audiobooks/{project_id}/analyze
POST   /audiobooks/{project_id}/generate
POST   /audiobooks/{project_id}/export
```

## 15.6 LLM

```text
GET    /llm/models
POST   /llm/chat
POST   /llm/analyze-emotion
POST   /llm/analyze-speakers
GET    /llm/prompts
POST   /llm/prompts
```

## 15.7 系统资源

```text
GET    /system/health
GET    /system/telemetry
GET    /system/gpu
GET    /system/runtimes
GET    /system/logs
```

长时间运行的操作必须采用任务 ID 和异步任务状态管理。

前端通过 WebSocket 或轮询获取任务进度。

---

# 16. 前端架构

## 16.1 技术栈

- Vue 3。
- TypeScript。
- Vite。
- Tailwind CSS。
- Pinia。
- Vue Router。
- Axios。
- Web Audio API。

## 16.2 页面布局

```text
VoiceStudio
│
├── Dashboard
│
├── Model Center
│   ├── Model Market
│   ├── Installed Models
│   ├── Model Details
│   └── Runtime Manager
│
├── TTS Lab
│   ├── Playground
│   ├── Batch Generation
│   └── Generation Presets
│
├── Benchmark
│   ├── Test Suites
│   ├── Experiment Builder
│   ├── Running Experiments
│   ├── Blind Listening
│   └── Reports
│
├── Voice Library
│   ├── Voice Profiles
│   ├── Reference Audio
│   └── Model Bindings
│
├── Audiobook Studio
│   ├── Projects
│   ├── Chapters
│   ├── Characters
│   ├── Emotion Editor
│   └── Audio Timeline
│
├── LLM Hub
│
├── STT Lab
│
├── Realtime Call
│
├── History
│
└── Settings
```

## 16.3 前端设计原则

- 统一设计系统。
- 所有模型共用 Playground 框架。
- 参数表单动态生成。
- 任务状态实时显示。
- 音频播放控件统一。
- Benchmark 支持多列对比。
- 盲听时隐藏模型身份。
- 长篇有声书支持章节树和段落列表。
- 运行环境与模型业务界面分离。

---

# 17. 部署架构

## 17.1 推荐部署方式

考虑到 RX 9070 XT 的 AMD ROCm 兼容性，建议将 GPU 推理运行层优先部署在 AMD 官方支持的 Linux 环境中。

Windows 作为日常桌面和浏览器操作环境可以保留。

建议部署结构：

```text
Windows Desktop
│
├── Browser
│
└── VoiceStudio Frontend
        │
        ▼
Linux GPU Host
│
├── FastAPI Core
│
├── Model Registry
│
├── Runtime Manager
│
├── GPU Scheduler
│
├── CPU Workers
│
├── ROCm Workers
│   ├── Qwen3-TTS
│   ├── CosyVoice
│   ├── IndexTTS
│   └── Other Models
│
├── Audio Processing
│
├── SQLite / PostgreSQL
│
└── Storage
```

如果最终采用 Windows 原生部署，必须单独验证每个模型在对应 Windows/PyTorch 环境下的支持情况。

不能把 Linux ROCm 可运行直接等同于 Windows 或 WSL2 可运行。

## 17.2 Docker Compose

推荐将核心服务与模型 Worker 分开。

```text
docker/
├── compose.yml
├── compose.dev.yml
├── backend/
├── frontend/
└── workers/
    ├── qwen3/
    ├── cosyvoice/
    └── indextts/
```

需要注意：

- 不强制所有模型都必须 Docker 化。
- 不兼容容器运行的模型允许使用独立 Python 环境。
- GPU 设备必须正确映射。
- 不同模型的依赖不能共用一个无约束环境。
- 模型权重使用宿主机持久化目录。
- 不把模型权重打包进镜像。

## 17.3 数据目录

```text
VoiceStudioData/
│
├── models/
│   ├── tts/
│   ├── stt/
│   └── llm/
│
├── environments/
│
├── voices/
│
├── datasets/
│
├── generated/
│   ├── tts/
│   ├── benchmark/
│   └── audiobooks/
│
├── reports/
│
├── logs/
│
└── database/
```

所有路径都必须通过配置文件管理，禁止在代码中写死绝对路径。

---

# 18. 推荐项目目录

```text
VoiceStudio/
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── composables/
│   │   ├── stores/
│   │   ├── views/
│   │   ├── router/
│   │   └── types/
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   ├── core/
│   │   │   ├── config/
│   │   │   ├── registry/
│   │   │   ├── runtime/
│   │   │   ├── scheduler/
│   │   │   └── events/
│   │   │
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   ├── voices/
│   │   │   ├── benchmark/
│   │   │   ├── audiobook/
│   │   │   └── generation/
│   │   │
│   │   ├── services/
│   │   │   ├── synthesis/
│   │   │   ├── benchmark/
│   │   │   ├── voice/
│   │   │   ├── audiobook/
│   │   │   ├── llm/
│   │   │   ├── stt/
│   │   │   └── audio/
│   │   │
│   │   ├── repositories/
│   │   ├── schemas/
│   │   └── main.py
│   │
│   └── tests/
│
├── model_plugins/
│   ├── tts/
│   ├── stt/
│   └── llm/
│
├── runtimes/
│   ├── cpu/
│   ├── rocm/
│   └── cloud/
│
├── scripts/
│   ├── install_model.py
│   ├── validate_model.py
│   └── benchmark.py
│
├── migrations/
│
├── docker/
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── plugins/
│   ├── deployment/
│   └── benchmark/
│
└── README.md
```

---

# 19. 安全性与许可证管理

平台完全本地运行不代表没有安全风险。

必须考虑：

1. 模型权重许可证。
2. 模型代码许可证。
3. 用户参考音频授权。
4. 声音克隆使用权限。
5. 上传文件安全检查。
6. 插件代码运行隔离。
7. API 访问控制。
8. 本地服务的跨域限制。
9. 文件路径校验。
10. 外部模型下载来源验证。

模型安装前展示：
- 项目来源。
- 代码许可证。
- 权重许可证。
- 商业用途状态。
- 是否允许修改。
- 是否允许再分发。

插件不应该拥有任意修改核心数据库和任意访问用户文件的权限。

---

# 20. 日志、监控和故障处理

## 20.1 日志分类

```text
logs/
├── application.log
├── api.log
├── model_registry.log
├── runtime.log
├── gpu_scheduler.log
├── tts.log
├── benchmark.log
├── llm.log
├── stt.log
└── audio.log
```

## 20.2 必须监控的指标

| 类别 | 指标 |
|---|---|
| CPU | 使用率、温度（可获取时）、线程负载 |
| GPU | 利用率、显存、温度、功耗（可获取时） |
| RAM | 内存使用量 |
| VRAM | 显存使用量、峰值 |
| Runtime | 启动时间、加载时间、错误次数 |
| TTS | RTF、生成耗时、失败率 |
| API | 请求数量、响应时间、错误率 |
| Storage | 文件数量、磁盘空间 |

不支持的硬件监控字段必须显示为不可用，不能填入伪造数据。

## 20.3 故障处理

模型运行失败时：

1. 保存完整异常信息。
2. 记录模型版本。
3. 记录运行环境。
4. 记录 GPU 状态。
5. 标记当前任务失败。
6. 清理临时资源。
7. 允许用户重新尝试。
8. 不允许异常自动污染其他模型环境。

---

# 21. 测试策略

## 21.1 单元测试

覆盖：
- Provider 参数映射。
- Model Manifest 解析。
- 能力注册。
- 任务状态转换。
- 数据库操作。
- 音色关联。
- Benchmark 评分。
- 文件存储。

## 21.2 集成测试

覆盖：
- 模型安装。
- Runtime 启动。
- 模型加载。
- TTS 生成。
- 音频保存。
- 历史归档。
- GPU 调度。
- 模型切换。
- 任务取消。

## 21.3 兼容性测试

每个模型正式接入前必须完成：

| 检查项 | 要求 |
|---|---|
| 安装 | 环境可安装 |
| 加载 | 模型可以正常加载 |
| 推理 | 能生成有效音频 |
| 输出 | 格式和采样率正确 |
| 参数 | 参数映射有效 |
| 显存 | 峰值可监测 |
| 稳定性 | 重复生成不发生异常 |
| 卸载 | 能正确释放资源 |
| 许可证 | 信息已登记 |

## 21.4 验收标准

V3.0 核心功能验收：

- [ ] 能动态发现模型插件。
- [ ] 能显示模型详细能力。
- [ ] 能安装和管理模型权重。
- [ ] 能使用独立运行环境。
- [ ] 能加载和卸载模型。
- [ ] 能显示模型运行状态。
- [ ] 能监控 GPU 显存。
- [ ] 能执行统一 TTS 请求。
- [ ] 能保留模型专属参数。
- [ ] 能管理人物音色。
- [ ] 能执行多模型 Benchmark。
- [ ] 能执行盲听评分。
- [ ] 能生成评测报告。
- [ ] 能保存可复现实验配置。
- [ ] 能进行长文本分段合成。
- [ ] 能完成小说人物和情绪分析。
- [ ] 能人工修改情绪标注。
- [ ] 能生成章节音频。
- [ ] 能导出最终有声书。

---

# 22. 分阶段开发计划

## Phase 1：核心框架

目标：先建立正确的架构，不追求一次接入所有模型。

任务：
1. 搭建 Vue3 + FastAPI。
2. 建立数据库和迁移机制。
3. 实现 Model Registry。
4. 实现 Manifest。
5. 实现 BaseTTSProvider。
6. 实现 Runtime Manager。
7. 实现 GPU Scheduler。
8. 实现统一任务状态管理。
9. 实现系统资源监控。

验收：平台能发现一个模型、启动模型、执行推理并卸载模型。

## Phase 2：首批模型接入

建议顺序：

1. Piper。
2. Kokoro。
3. F5-TTS。
4. Qwen3-TTS 1.7B。
5. CosyVoice 3。
6. IndexTTS-2.5。

每个模型单独验证运行环境和 AMD GPU 兼容性。

验收：至少三个不同模型可以通过统一 Playground 正常生成音频，且各自专属参数可以使用。

## Phase 3：Benchmark 实验中心

任务：
1. 测试集管理。
2. 实验创建。
3. 多模型任务调度。
4. 性能监控。
5. ASR 评测。
6. 盲听功能。
7. 人工评分。
8. 报告导出。

验收：Qwen3-TTS、CosyVoice 3、IndexTTS-2.5 能够参加同一音色条件的对照实验。

## Phase 4：Voice Library 和 LLM Hub

任务：
1. 人物档案。
2. 参考音频管理。
3. 跨模型音色映射。
4. LLM 模型接入。
5. 人物识别。
6. 情绪分析。
7. 用户修改情绪。
8. 结构化参数转换。

验收：用户可以创建人物、为人物绑定不同模型，并对同一人物进行多模型试听。

## Phase 5：Audiobook Studio

任务：
1. 小说导入。
2. 章节解析。
3. 段落切分。
4. 人物识别。
5. 情绪分析。
6. 用户校正。
7. 批量生成。
8. 音频拼接。
9. 章节导出。

验收：能够完成一部短篇小说从文本到多角色有声书的完整制作。

## Phase 6：模型生态扩展

逐步加入：
- MOSS-TTS。
- VoxCPM2。
- Fish Speech。
- GLM-TTS。
- FireRedTTS-2。
- Spark-TTS。
- 其他后续出现的模型。

验收：新增模型不需要修改平台核心业务逻辑。

---

# 23. 最终架构决策

| 决策项 | 最终方案 |
|---|---|
| 平台定位 | 本地 AI 语音模型实验与制作平台 |
| 前端 | Vue 3 + TypeScript |
| 后端 | FastAPI |
| 数据库 | SQLite 起步，兼容 PostgreSQL |
| 模型管理 | Manifest + Model Registry |
| 模型接入 | Provider 插件体系 |
| 模型运行 | 独立进程 / Worker / Container |
| GPU 调度 | 独立 GPU Scheduler |
| AMD GPU | ROCm 优先，逐模型验证 |
| TTS | 多模型可插拔 |
| STT | 多 Provider |
| LLM | 本地优先，云端可选 |
| 音色 | 独立 Voice Library |
| 实验 | Benchmark + 盲听 + 自动评测 |
| 有声书 | 独立 Audiobook Studio |
| 音频存储 | 文件系统 + 数据库元数据 |
| 部署 | Linux GPU Runtime 优先，Windows 可作为客户端或其他服务宿主 |
| 扩展方式 | 新模型通过插件和 Manifest 接入 |

---

# 24. 给开发 Agent 的执行约束

以下规则属于项目开发约束，不能因为实现方便而随意修改。

1. 必须先阅读本架构文档，再开始开发。
2. 不允许将所有 TTS 模型直接写入一个 Provider 文件。
3. 不允许将所有模型依赖安装在同一个 Python 环境。
4. 不允许让具体模型代码直接依赖 FastAPI 路由。
5. 不允许将模型专属参数全部简化成统一参数。
6. 不允许未经兼容性验证就声明支持 AMD GPU。
7. 不允许在没有实际测试数据的情况下填写性能成绩。
8. 不允许让多个重型模型未经资源检查同时加载到 GPU。
9. 不允许在前端硬编码每个模型的参数表单。
10. 不允许把人物音色和模型音色混为一个对象。
11. 不允许将 LLM 输出未经校验直接传给 TTS。
12. 不允许让有声书业务逻辑和实时通话业务逻辑互相耦合。
13. 不允许破坏已有功能而不提供迁移方案。
14. 所有新增 API 必须有请求和响应 Schema。
15. 所有核心模块必须有单元测试。
16. 所有模型必须保存版本、来源和许可证信息。
17. 所有生成实验必须能够追溯到具体模型版本和参数。
18. 所有运行时异常必须有结构化日志。
19. 所有数据路径必须支持配置，禁止硬编码绝对路径。
20. 不得把计划中的功能描述为已经实现的功能。

## 开发过程要求

每完成一个阶段，必须提交：

- 完成内容。
- 修改文件。
- 新增模块。
- 数据库变更。
- API 变化。
- 测试结果。
- 已知问题。
- 下一阶段计划。

如果发现现有架构与本设计冲突，应先说明冲突、影响范围和建议方案，不得直接删除或修改架构约束。

---

# 25. 项目最终目标

VoiceStudio V3.0 最终应该实现以下使用体验：

用户打开平台后，可以：

1. 查看当前安装的全部语音模型。
2. 浏览可接入的新模型。
3. 安装和管理模型。
4. 查看每个模型的功能和硬件兼容情况。
5. 选择任意模型进行语音合成。
6. 使用模型专属参数调整效果。
7. 使用相同文本和参考音频对比多个模型。
8. 隐藏模型名称进行盲听。
9. 对模型进行评分并生成排名。
10. 保存自己喜欢的音色。
11. 创建小说人物并绑定不同模型。
12. 让 LLM 自动分析人物和情绪。
13. 人工修改情绪、语气和表达方式。
14. 生成长篇多角色有声书。
15. 对生成结果进行试听、修订和重新生成。
16. 查看历史实验、音频和评测报告。

**最终定位：VoiceStudio 不应该只是一个 TTS 播放器，而应该成为一个能够持续接入、运行、测试、比较和使用新一代 AI 语音模型的本地语音实验平台。**