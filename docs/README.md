# VoiceStudio & Realtime Voice Agent (智能语音与大模型管理平台)

基于 **AMD Ryzen 7 9800X3D (CPU)** 与 **AMD Radeon RX 9070 XT (GPU 16GB)** 构建的完全本地化、纯离线、支持多大模型编排与多语音引擎的高性能语音平台。

---

## 零、 VoiceStudio Web 一体化管理平台 (一键启动)

本项目现已配备现代化的 **Web 交互与多模型管理平台**，无需安装复杂前端构建工具，原生单页集成：

```powershell
cd G:\code\backend\clouldflare\voice_agent

# 1. 标准生产/开发启动 (推荐，支持热重载)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. 或快捷启动脚本 (自动转发至 app.main:app)
python main_server.py
# 或 python legacy/main_server.py
```

- 🌐 **Web 控制台地址**：`http://localhost:8000`
- 📖 **Swagger API 接口文档**：`http://localhost:8000/docs`

### 核心功能模块：
1. 🎙️ **调音工坊 (Playground)**：任意文字/预设语料输入，动态调节语速与音调，联动 SenseVoice 自动计算发音准确率（CER）与实时率（RTF）。
2. 👥 **音色矩阵库 (Timbre Matrix)**：集中展示 20 款人声音色（涵盖普通话、东北话、陕西话、台湾腔、粤语及美语），支持秒级免下载在线试听。
3. 🧠 **大模型编排 (LLM Hub)**：管理本地 Qwen2.5-1.5B GPU 模型与 Cloudflare Workers AI 云端无服务器模型，配置角色 Prompt 与在线对话。
4. 📊 **自动化评测大厅 (Benchmark Suite)**：一键执行 8 大标准分类测试用例，生成并在线查阅规范 Markdown 报告。
5. ⚡ **实时语音通话 (Voice Call)**：硬件全双工实时通话状态监控与启动。

---

## 一、 系统架构与异构硬件分配

为了彻底杜绝卡顿并实现超低交互延迟，系统采用了 **CPU + GPU 异构多线程协同流水线**：

```text
[用户通过麦克风说话]
    │ 16kHz 音频流
    ▼
【AMD Ryzen 7 9800X3D (CPU 8核/16线程)】
  • Silero-VAD: 实时人声端点检测 (静音 > 500ms 自动判停，延迟 < 5ms)
  • SenseVoice-Small INT8: 极速语音识别 (~30ms 极速转文字)
    │ 
    ▼ 纯文字送入
【AMD Radeon RX 9070 XT (GPU 16GB 显存)】
  • Qwen2.5-1.5B-Instruct: 大语言模型高速思考 (首字延迟 TTFT 仅 20ms)
    │ 
    ▼ 流式 Token 吐出并按标点智能断句 (。！？，\n)
【AMD Radeon RX 9070 XT (GPU 16GB 显存)】
  • ChatTTS: 拟人化语音生成 (支持欢笑声 [laugh]、呼吸叹气、口语语气停顿)
    │ 
    ▼ 放入无缝播放队列
[默认耳机 / 扬声器顺序播放]

⚡【全双工即时打断 (Barge-in)】：
  当 AI 正在通过扬声器说话时，只要用户对着麦克风插话，VAD 将在 10 毫秒内检测到人声，
  瞬间掐断当前音频播放 (sd.stop())，清空待播队列，并中止当前回复生成，立即重新倾听用户！
```

---

## 二、 规范目录结构与核心文件清单

平台已全面完成工程化结构治理，根目录保持精炼干净，代码与测试资源分类归整：

```text
G:\code\backend\clouldflare\voice_agent/
├── app/                  # ★【平台生产级微内核与服务架构】
│   ├── api/              # RESTful API 路由模块 (voices, tts, llm, call, benchmark, telemetry)
│   ├── core/             # 抽象基类 BaseProvider, BaseTTS, ModelManifest, HistoryManager
│   ├── providers/        # 插件化模型驱动提供商 (TTS, LLM, STT, VAD)
│   └── static/           # Web 前端原生单页应用 (Vue 3 + Tailwind CSS + Lucide)
├── assets/               # ★【静态与音频测试资源库】
│   └── audio/            # 评测与引擎验证音频文件 (*.wav)
├── docs/                 # ★【设计规范与核心文档库】
│   ├── README.md         # 平台快速启动与架构说明全景手册 (本文档)
│   ├── ARCHITECTURE.md   # 系统整体分层架构说明书 (ADD V2.0 生产级就绪规范)
│   ├── TTS_TEST_SPECIFICATION.md # TTS 八大维度自动化基准评测规范
│   └── writing-block*.md # 架构演进与演化过程记录
├── legacy/               # ★【独立 CLI 交互与实验脚本归档】
│   ├── main_server.py    # FastAPI 服务后端入口
│   ├── realtime_call.py  # 纯命令行全双工实时语音通话系统
│   ├── run_tts_benchmark.py # 独立自动化评测套件跑分脚本
│   ├── prosody_studio.py # 独立抑扬顿挫实验室
│   ├── voice_chat.py     # 独立控制台文本交互流式朗读
│   ├── test_stt.py       # 独立语音识别与 CER 评测脚本
│   ├── test_tts.py       # 单次 TTS 测试验证脚本
│   ├── check_mic.py      # 麦克风设备扫描与音量检测工具
│   ├── tts_engine.py     # 独立 TTS 引擎适配包
│   └── config.py         # 历史配置向后兼容层
├── speakers/             # 音色库与参考音频 (含 edge/ 试听音频、克隆参考原声与用户录音 uploads/)
├── models/               # 纯离线 ONNX / GGUF 模型存储目录 (Silero-VAD + SenseVoice-Small)
├── benchmark_results/    # 基准评测报告 (Markdown) 与生成波形归档
├── voice_studio_history.db # 调音实验与多轮通话持久化 SQLite 数据库
└── *.py                  # ★【根目录向后兼容转发存根 (Stub)】 (无缝转发执行至 legacy/)
```

---

## 三、 六大核心语音合成引擎与音色库全景 (TTS Engines)

系统现已支持 **6 种完全不同技术架构的语音生成引擎**，共包含 **31 款高拟真音色**，且全部集成在 Web 控制台与自动化评测套件中：

| 引擎类型 | 技术架构 | 运行环境 | 核心优势 | 声音克隆 (Zero-Shot) |
| :--- | :--- | :--- | :--- | :---: |
| **Piper-TTS** | **轻量 VITS ONNX** | **纯本地 CPU 极速 (0显存)** | **63MB 极致轻量**，RTF 仅 **0.08~0.12**，语速平滑缩放 | 纯离线中英音色 |
| **Kokoro-82M** | **StyleTTS2 极致蒸馏架构** | **纯本地 CPU 极速 (0显存)** | **82M 超轻量**，中英双语极高自然度，RTF 0.16 极速响应 | 内置多语种音色库 |
| **F5-TTS** | **非自回归流匹配扩散** (Flow Matching) | **纯本地 GPU 加速 (RX 9070 XT)** | **零自回归死循环与幻觉**，连贯从容 | **支持 (3~5秒克隆你的声音)** |
| **MeloTTS** | **超轻量离线 VITS** (Sherpa-ONNX) | **纯本地 CPU 极速 (0显存)** | **44.1kHz 高保真**，中英双语，**评测准确率 100.0%** | 预设高拟真音色 |
| **Edge-TTS** | 云端深度神经声学对齐 | 需联网 (0显存) | **99%+ 极高准确率**，音色丰富自然 | 微软14款多方言音色 |
| **ChatTTS** | 自回归离散 Token 架构 | 纯本地 GPU 加速 | 口语欢笑与叹息，潜变量种子探索 | 768维高斯抽卡 |

---

### 3.1 Kokoro-82M 纯本地轻量之王 (100% 离线 CPU 极速运行)

基于 StyleTTS2 结构极致蒸馏的 82M 参数超轻量模型，纯 CPU 运行，零显存开销：
- 采样率：24kHz，单句生成耗时仅 **0.2~0.4 秒** (RTF ~0.16)。
- 预设音色：`kokoro_zh_xiaoxiao` (知性女声)、`kokoro_zh_yunxi` (阳光男声)、`kokoro_en_heart` (美语旗舰)、`kokoro_en_michael` (美语男声)。

---

### 3.2 F5-TTS 纯本地流匹配声音克隆引擎 (100% 离线 GPU 加速)

基于非自回归 Flow Matching Diffusion 架构，彻底消除传统自回归模型的吞字与幻觉问题，并在 Web 平台中完整支持用户**录制或上传 3~5 秒自己声音一键克隆**：

| 预设别名 | 参考声音源 | 性别 | 对应参考音频 | 说明 |
| :--- | :--- | :---: | :---: | :--- |
| **`f5_clone_warm_female`** | 知性女声 | 女声 | `speakers/spk_2222.wav` | 基于知性温润参考音频进行零样本声线克隆 |
| **`f5_clone_young_male`** | 阳光男声 | 男声 | `speakers/spk_7869.wav` | 基于清爽青年参考音频进行零样本声线克隆 |
| **`f5_clone_deep_male`** | 沉稳叔音 | 男声 | `speakers/spk_1111.wav` | 基于成熟低沉参考音频进行零样本声线克隆 |
| **`f5_custom_clone`** | **用户自定义** | 任意 | 用户上传 / 麦克风录音 | **上传 3~5 秒录音，SenseVoice 自动转写并克隆你的声音** |

---

### 3.2 MeloTTS 纯本地极速轻量 VITS 引擎 (纯 CPU 离线运行)

基于 Sherpa-ONNX 纯离线引擎，模型体积仅 **53MB**，完全不占显卡显存：
- 音色 ID：`melo_zh_en`
- 采样率：**44.1kHz 超高保真**
- 客观评测成绩：在日常标准测试集（Daily）中达到 **100.0% 字符识别准确率 (CER=0.0%)**，RTF 仅 0.97
- 特色：中英文混合无缝朗读，纯离线极速运行（0 显存占用，低负载）。

---

### 3.3 Edge-TTS 生产级推荐音色库（云端参考基准）

可通过 `--voice <参数别名>` 直接调用：

| 快捷别名 (`--voice`) | 微软官方 ShortName | 性别/地区 | 试听音频 | 声线质感与典型场景 |
| :--- | :--- | :---: | :---: | :--- |
| **`yunxi` (默认)** | `zh-CN-YunxiNeural` | 青年男声 | [yunxi.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/yunxi.mp3) | **清爽阳光青年男声**：吐字干净、自然充满活力，**全双工通话最佳默认音色** |
| **`xiaoxiao`** | `zh-CN-XiaoxiaoNeural` | 知性女声 | [xiaoxiao.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/xiaoxiao.mp3) | **知性温柔暖心女声**：微软旗舰声线，温润亲和，生活助理、有声读物与陪伴首选 |
| **`yunjian`** | `zh-CN-YunjianNeural` | 成熟男声 | [yunjian.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/yunjian.mp3) | **沉稳磁性叙事叔音**：“注意看这个男人叫小帅”标配声线，影视解说、历史叙事神韵 |
| **`xiaoyi`** | `zh-CN-XiaoyiNeural` | 灵动女声 | [xiaoyi.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/xiaoyi.mp3) | **活泼灵动少女音**：轻快生动、富有元气感染力，动漫陪伴与趣味闲聊 |
| **`yunxia`** | `zh-CN-YunxiaNeural` | 少年男声 | [yunxia.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/yunxia.mp3) | **朝气清脆少年正太**：童真清澈、阳光开朗，青少年问答与少儿伴读 |
| **`yunyang`** | `zh-CN-YunyangNeural` | 播音男声 | [yunyang.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/yunyang.mp3) | **专业新闻播报男声**：成熟稳重、抑扬规范，科技发布、新闻早报、政企汇报 |
| **`xiaobei`** | `zh-CN-liaoning-XiaobeiNeural` | 东北女声 | [xiaobei.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/xiaobei.mp3) | **东北特色方言女声**：“哎呀妈呀”东北腔，诙谐幽默，接地气唠嗑脱口秀 |
| **`xiaoni`** | `zh-CN-shaanxi-XiaoniNeural` | 陕西女声 | [xiaoni.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/xiaoni.mp3) | **陕西方言中原女声**：质朴爽朗、西北风情，“嘹咋咧”亲切生活唠嗑 |
| **`hsiaochen`** | `zh-TW-HsiaoChenNeural` | 台湾女声 | [hsiaochen.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/hsiaochen.mp3) | **台湾软萌甜美女声**：机车甜美腔调，温柔细腻，治愈系慢聊与贴心叮嘱 |
| **`yunjhe`** | `zh-TW-YunJheNeural` | 台湾男声 | [yunjhe.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/yunjhe.mp3) | **台湾清朗从容男声**：轻松日常、自然亲切的台湾腔调小哥 |
| **`hiugaai`** | `zh-HK-HiuGaaiNeural` | 粤语女声 | [hiugaai.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/hiugaai.mp3) | **粤语香港知性女声**：地道粤语发音，优雅大方，粤语区专属交互 |
| **`wanlung`** | `zh-HK-WanLungNeural` | 粤语男声 | [wanlung.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/wanlung.mp3) | **粤语香港成熟男声**：成熟稳健、港风男神声线 |
| **`andrew`** | `en-US-AndrewNeural` | 美语男声 | [andrew.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/andrew.mp3) | **标准美式青年男声**：发音地道纯正，英语口语对话与双语教学 |
| **`ava`** | `en-US-AvaNeural` | 美语女声 | [ava.mp3](file:///g:/code/backend/clouldflare/voice_agent/speakers/edge/ava.mp3) | **标准美式优雅女声**：自然流畅、知性端庄 |

---

### 3.2 ChatTTS 实验性潜变量音色库 (仅在 `--engine chattts` 时生效)

| 参数名称 (`--speaker`) | 种子 (Seed) | 试听音频 | 声线风格与特色 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`warm_female` (默认)** | `2222` | `speakers/spk_2222.wav` | **知性温润女声**：音质温和清亮，吐字亲切自然 | 贴心陪伴、生活助理、知识问答 |
| **`young_male`** | `7869` | `speakers/spk_7869.wav` | **磁性阳光青年男声**：声音清朗利落，充满朝气 | 科技咨询、日常对话、男声助手 |
| **`sweet_female`** | `4099` | `speakers/spk_4099.wav` | **甜美灵动少女音**：语调活泼灵巧，亲和力极佳 | 轻松闲聊、游戏互动、娱乐场景 |
| **`deep_male`** | `1111` | `speakers/spk_1111.wav` | **成熟沉稳低音叔音**：胸腔共鸣浓厚，低沉磁性 | 叙事故事、夜间电台、深情朗读 |
| **`gentle_female`** | `6653` | `speakers/spk_6653.wav` | **温柔从容暖心女声**：轻声细语，治愈感极强 | 睡前夜聊、情绪抚平、慢节奏陪伴 |
| **`vivid_boy`** | `5555` | `speakers/spk_5555.wav` | **活力朝气少年音**：清脆利落，阳光元气满满 | 年轻化场景、朝气交互 |

> [!TIP]
> 也可以通过 `--seed <数字>` 传入任意自定义整数种子，探索高维潜变量空间中成千上万种未被发现的独特声线！

---

## 四、 如何开启通话与切换声音/情感

在终端中执行：

```powershell
cd G:\code\backend\clouldflare\voice_agent

# 1. 默认推荐启动：极速高准确率 Edge-TTS 引擎 (云希 青年男声，准确率 99%+，零吞字零幻觉)
python realtime_call.py

# 2. 切换 Edge-TTS 人声：知性温柔暖心女声 (晓晓)
python realtime_call.py --voice xiaoxiao

# 3. 切换 Edge-TTS 人声：沉稳磁性叙事叔音 (云健)
python realtime_call.py --voice yunjian

# 4. 切换为本地实验性 ChatTTS 引擎：
python realtime_call.py --engine chattts --speaker warm_female --mood warm
```

---

## 五、 停顿与韵律处理机制 (为什么中间不会一口气读完)

- **逗号自然换气停顿 (`[lbreak]`)**：系统在文本清洗流水线中，将每个逗号及短连接符映射为 `[lbreak]` 口语呼吸标记，精准留出约 **190ms 的自然生理换气间隙**；
- **分号与大转折停顿 (`[llbreak]`)**：将分号与破折号映射为 `[llbreak]`，保留约 **350ms 的从容深呼吸时间**；
- **句末余音淡出与呼吸留白**：通过半余弦淡出（45ms）与 280ms 句尾静音延展，确保语段结束舒缓自然，杜绝“戛然而止”。

### 🎛️ 情感语音四重处理引擎架构：
1. **ChatTTS 拟人情感韵律词注入**：
   - 内置 Oral 口语韵律标签 (`[oral_5]` ~ `[oral_7]`)、轻笑特征 (`[laugh_0]` ~ `[laugh_1]`) 与微停顿呼吸标记 (`[break_4]`)，彻底摆脱死板播报。
2. **解锁音高起伏动态（Pitch Contour）**：
   - 固化音色 Seed 杜绝音色跳变的同时，调整解码温度至 `0.28~0.32`，自然呈现汉语问句上扬、感叹句重音、拖音等喜怒哀乐。
3. **大模型语气助词与同理心激发**：
   - 系统级指令引导大模型输出自然的口语助词（“呢、呀、啦、哇、哈、哦、好嘞”），从源头赋予语音生动的旋律感，并彻底清除任何括号动作旁白与 `uv_break` 生硬文字。
4. **数字信号声学润色增强 (Acoustic Enhancement)**：
   - **胸腔温暖共鸣 EQ**：在 250Hz 人声基频微增益，消除数码单薄感；
   - **清澈度与咬字辨析**：在 3500Hz 辅音区提升细节辨析力；
   - **柔和动态限制 (Tanh Soft Limiter)**：提升低音量细节，防止爆音破音；
   - **余音半余弦淡出 + 280ms 换气延展**：句末娓娓道来，从容不仓促。

---

## 六、 语音转文字与正确性校验工具 (test_stt.py)

基于本地 SenseVoice-Small INT8 模型构建，单句转录耗时仅 **30~80ms** (RTF 约 0.015)：

```powershell
cd G:\code\backend\clouldflare\voice_agent

# 1. 单音频转文字
python test_stt.py --audio speakers/spk_2222.wav

# 2. 正确性校验（与期望文本自动逐字核对，计算准确率与字符错误率 CER）
python test_stt.py --audio speakers/spk_2222.wav --text "你好呀！我是你的智能语音助手，今天由我来为你服务，你觉得我的声音怎么样呢？"

# 3. 批量扫描并转录整个目录下的所有音频
python test_stt.py --dir speakers/
python test_stt.py --dir prosody_outputs/

# 4. 实时麦克风收音转文字测试（测试音箱麦克风识别精准度）
python test_stt.py --mic
```

---

## 七、 体验方式与全双工免按键通话

### 7.1 Web 控制台免按键连续对话电台 (推荐)
1. 浏览器打开 `http://localhost:8000`，切换至第 5 个标签页【实时语音交互通话】；
2. 点击「**开始语音通话**」：麦克风保持常开，无需按键；
3. **Web Audio VAD 自动端点检测**：对着麦克风正常说话，说罢停顿约 1.1 秒，系统自动切片提交识别、思考并播放拟人回复；
4. **自动复听接续**：AI 播音结束瞬间自动切回倾听，支持强势插话打断 (Barge-in)；
5. **对话记忆持久化**：所有历史轮次均自动存入 SQLite，刷新页面自动恢复。

### 7.2 终端版全双工原生通话 (CLI 兼容)
1. **无需敲击任何按键**：程序启动后自动锁定麦克风，显示 `🟢 正在倾听...`，直接说话即可；
2. **自动识别与回答**：你说完停顿约 0.45 秒，AI 自动转文字并用高质感拟人声音流式回答；
3. **Barge-in 即时插话**：只要用户插话，VAD 在 10ms 内检测并掐断当前播音，立即重新倾听。
---

## 八、 自动化评测基准套件 (run_tts_benchmark.py)

参考标准语音规范，覆盖日常对话、情绪表达、数字正规化、中英技术混杂、标点停顿、绕口令等 8 大维度，自动生成评测报告：

```powershell
cd G:\code\backend\clouldflare\voice_agent

# 1. 运行高准确率 Edge 引擎日常基准测试 (实测 98.53% 准确率，RTF 0.22)
python run_tts_benchmark.py --engine edge --category daily

# 2. 运行数字正规化发音测试
python run_tts_benchmark.py --engine edge --category numbers

# 3. 运行标点与生理气口测试 (实测 100.0% 完美发音)
python run_tts_benchmark.py --engine edge --category punctuation

# 4. 运行全量 8 大分类评测
python run_tts_benchmark.py --engine edge --all

# 5. 对比运行 ChatTTS 引擎基准：
python run_tts_benchmark.py --engine chattts --category daily
```
