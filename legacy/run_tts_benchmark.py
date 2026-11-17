import os
import sys
import time
import re
import argparse
import numpy as np

# Ensure voice_agent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

import soundfile as sf
import scipy.signal as signal
from test_stt import AudioTranscriber, calculate_cer
from tts_engine import get_tts_engine, EdgeTTSEngine, ChatTTSEngine

BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_results")
REPORTS_DIR = os.path.join(BENCHMARK_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Standard Test Corpus from TTS_TEST_SPECIFICATION.md
BENCHMARK_CORPUS = {
    "daily": [
        ("D-01", "你好，欢迎使用本地智能语音助手，很高兴认识你。"),
        ("D-02", "今天天气真不错，我们要不要一起出去走走？"),
        ("D-03", "请问附近有没有比较好吃的餐厅？最好是人少、环境安静的那种。"),
        ("D-04", "等一下，我马上就回来，你先在这里等我一会儿哈。"),
        ("D-05", "这件事情其实没有你想象的那么复杂，我们可以慢慢来解决。"),
        ("D-06", "如果你已经准备好了，那么我们现在就正式开始吧！"),
    ],
    "emotion": [
        ("E-01", "太棒了！我们终于成功了！这真是一个令人激动的好消息！"),
        ("E-02", "哇！真的假的呀？我简直不敢相信自己的眼睛！"),
        ("E-03", "有些事情，即使过去了很久，想起来还是会觉得有些遗憾。"),
        ("E-04", "我以为我们还有很多时间，没想到这竟然是最后一次见面。"),
        ("E-05", "你到底有没有认真听我说话？我已经解释过很多次了。"),
        ("E-06", "没关系的，不要着急，一切都会慢慢好起来的。"),
        ("E-07", "累了就好好休息一下吧，今天你已经做得很棒了，晚安。"),
        ("E-08", "难道你从一开始，就早已经知道最终的答案了吗？"),
        ("E-09", "什么？你刚才说什么？这件事情真的发生了吗？！"),
        ("E-10", "乌云在天空中翻滚，暴风雨即将降临！我们要勇敢地飞翔！"),
    ],
    "numbers": [
        ("N-01", "我的手机号码是 13800138000。"),
        ("N-02", "今天是 2026 年 11 月 15 日，现在的时间是上午 9 点 30 分。"),
        ("N-03", "这台电脑的价格是 8,599.99 元，折扣后只需要 6,880 元。"),
        ("N-04", "当前室外温度是 -5℃，最高温度预计达到 18.5℃。"),
        ("N-05", "圆周率前几位约等于 3.1415926。"),
        ("N-06", "服务器的局域网 IP 地址是 192.168.1.100，开放端口为 8080。"),
        ("N-07", "本季商品一律 75 折，每满 300 减 50 元。"),
        ("N-08", "请在 10:30 前完成任务，预计耗时 2 小时 15 分钟。"),
    ],
    "multilingual": [
        ("M-01", "Hello everyone，欢迎来到今天的 AI Voice Testing 实验室！"),
        ("M-02", "我正在使用 Python、JavaScript 和 TypeScript 开发全新的 Web Application。"),
        ("M-03", "请访问 https://www.google.com 查看官方的 API Documentation。"),
        ("M-04", "当前主机的 CPU 使用率达到了 95%，但是 RAM 依然有足够的剩余空间。"),
        ("M-05", "ChatGPT 和 Qwen 是两款非常强大的人工智能大语言模型。"),
        ("M-06", "请使用 HTTP POST 方法向服务接口发送 JSON 格式的数据包。"),
        ("M-07", "ESP32-S3 芯片同时支持 Wi-Fi 和 Bluetooth Low Energy，也就是我们常说的 BLE。"),
        ("M-08", "The quick brown fox jumps over the lazy dog."),
    ],
    "punctuation": [
        ("P-01", "第一步，打开电源；第二步，连接网络；第三步，启动系统。"),
        ("P-02", "他犹豫了很久……最终，还是决定迈出那一步。"),
        ("P-03", "这……这是真的吗？我不是在做梦吧？"),
        ("P-04", "生活就像一盒巧克力——你永远不知道下一颗是什么味道。"),
        ("P-05", "难道你真的以为，只要放弃了，一切就会自然结束吗？"),
        ("P-06", "春天的花，夏天的风，秋天的落叶，冬天的雪，四季各有各的美好。"),
        ("P-07", "注意！前方有危险！请立即减速慢行！"),
    ],
    "articulation": [
        ("A-01", "四是四，十是十，十四是十四，四十是四十。"),
        ("A-02", "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。"),
        ("A-03", "扁担长，板凳宽，板凳没有扁担长，扁担没有板凳宽。"),
        ("A-04", "红鲤鱼与绿鲤鱼与驴。"),
        ("A-05", "刘奶奶找牛奶奶买牛奶，牛奶奶给刘奶奶拿牛奶。"),
    ],
    "longtext": [
        ("L-01", "在苍茫的大海上，狂风卷集着乌云。在乌云和大海之间，海燕像黑色的闪电，在高傲地飞翔。一会儿翅膀碰着波浪，一会儿箭一般地直冲向乌云，它叫喊着，——就在这鸟儿勇敢的叫喊声里，乌云听出了欢乐。在这叫喊声里——充满着对暴风雨的渴望！在这叫喊声里，乌云听出了愤怒的力量、热情的火焰和胜利的信心。"),
    ],
    "robustness": [
        ("R-01", "测试系统对于特殊标点的过滤：【提示】@用户#话题$50%&and*星号~波浪线。"),
        ("R-02", "连续空白符与换行测试：\n\n\n你好，      请问你听得到吗？\n\n"),
        ("R-03", "生僻字与多音字发音测试：行行出状元，银行的行业标准非常严格。"),
        ("R-04", "表情符号测试：今天真的超级开心😊，庆祝一下🎉，加油💪！"),
    ]
}

class TTSBenchmarkRunner:
    def __init__(self, engine_type="edge", voice="yunxi", seed=7869):
        print("\n" + "="*70)
        print("   TTS 规范自动化基准测试套件 (TTS Benchmark Suite)")
        print("="*70)
        self.engine_type = engine_type
        self.voice = voice
        self.seed = seed

        print(f"初始化 TTS 引擎: [{engine_type.upper()}] (Voice/Seed: {voice if engine_type in ['edge', 'f5tts'] else seed})...")
        if engine_type == "edge":
            from app.providers.tts.edge_provider import EdgeTTSProvider
            self.engine = EdgeTTSProvider()
            self._voice_arg = {"voice_id": voice}
        elif engine_type == "melotts":
            from app.providers.tts.melotts_provider import MeloTTSProvider
            self.engine = MeloTTSProvider()
            self._voice_arg = {"voice_id": "melo_zh_en"}
        elif engine_type == "f5tts":
            from app.providers.tts.f5tts_provider import F5TTSProvider
            self.engine = F5TTSProvider()
            self._voice_arg = {"voice_id": voice or "f5_clone_warm_female"}
        elif engine_type == "kokoro":
            from app.providers.tts.kokoro_provider import KokoroTTSProvider
            self.engine = KokoroTTSProvider()
            self._voice_arg = {"voice_id": voice or "kokoro_zh_xiaoxiao"}
        else:
            from app.providers.tts.chattts_provider import ChatTTSProvider
            self.engine = ChatTTSProvider()
            self._voice_arg = {"voice_id": voice or "spk_young_male"}

        print("正在加载 SenseVoice-Small 识别校验模型 (CPU)...")
        self.transcriber = AudioTranscriber()
        print("测试环境初始化完成！\n")

    def run_case(self, case_id, text, category):
        cat_dir = os.path.join(BENCHMARK_DIR, f"{self.engine_type}_{category}")
        os.makedirs(cat_dir, exist_ok=True)
        out_wav = os.path.join(cat_dir, f"{case_id}.wav")

        res_synth = self.engine.synthesize(text, output_path=out_wav, **self._voice_arg)
        cost_time = res_synth["cost_sec"]
        duration = res_synth["duration"]
        rtf = cost_time / duration if duration > 0 else 0.0

        # STT Verification
        trans_text, _, stt_cost, _ = self.transcriber.transcribe(out_wav)
        cer, accuracy, (ref_norm, hyp_norm) = calculate_cer(text, trans_text)

        return {
            "case_id": case_id,
            "original_text": text,
            "transcription": trans_text,
            "accuracy": accuracy,
            "cer": cer,
            "duration": duration,
            "cost_time": cost_time,
            "rtf": rtf,
            "wav_path": out_wav
        }

    def run_category(self, category):
        if category not in BENCHMARK_CORPUS:
            print(f"❌ 未知分类: {category}")
            return []
            
        items = BENCHMARK_CORPUS[category]
        print(f"\n🚀 开始执行测试集: 【{category.upper()}】 (共 {len(items)} 条语料, 引擎: {self.engine_type})")
        print("-" * 75)
        print(f"{'编号':<6} | {'耗时':<6} | {'时长':<6} | {'RTF':<6} | {'准确率':<7} | {'评定'}")
        print("-" * 75)
        
        results = []
        for case_id, text in items:
            res = self.run_case(case_id, text, category)
            results.append(res)
            acc = res["accuracy"]
            status = "S 完美" if acc >= 95 else ("A 良好" if acc >= 80 else "B 瑕疵")
            print(f"{case_id:<6} | {res['cost_time']:4.2f}s | {res['duration']:4.2f}s | {res['rtf']:4.2f} | {acc:5.1f}%  | {status}")
        return results

    def generate_report(self, all_results):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(REPORTS_DIR, f"benchmark_{self.engine_type}_{timestamp}.md")
        
        total_cases = 0
        total_duration = 0.0
        total_cost = 0.0
        acc_list = []

        report_md = f"# TTS 规范自动化基准评测报告 ({self.engine_type.upper()})\n\n"
        report_md += f"- **测试时间：** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_md += f"- **TTS 引擎：** `{self.engine_type}` ({self.voice if self.engine_type == 'edge' else f'Seed {self.seed}'})\n"
        report_md += f"- **测试环境：** AMD Ryzen 7 9800X3D + AMD Radeon RX 9070 XT\n"
        report_md += f"- **评测工具：** SenseVoice-Small INT8 (Offline ASR Evaluator)\n\n"
        report_md += "---\n\n"
        report_md += "## 一、 各测试分类汇总表\n\n"
        report_md += "| 评测集 | 测试用例数 | 平均时长 | 平均RTF | 平均准确率 | 评级 |\n"
        report_md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"

        for cat, items in all_results.items():
            if not items:
                continue
            cat_dur = sum(x["duration"] for x in items) / len(items)
            cat_rtf = sum(x["rtf"] for x in items) / len(items)
            cat_acc = sum(x["accuracy"] for x in items) / len(items)
            grade = "S 卓越" if cat_acc >= 95 else ("A 良好" if cat_acc >= 80 else "B 合格")
            report_md += f"| **{cat}** | {len(items)} | {cat_dur:.2f}s | {cat_rtf:.3f} | **{cat_acc:.1f}%** | {grade} |\n"
            
            total_cases += len(items)
            total_duration += sum(x["duration"] for x in items)
            total_cost += sum(x["cost_time"] for x in items)
            acc_list.extend([x["accuracy"] for x in items])

        overall_acc = sum(acc_list) / max(len(acc_list), 1)
        overall_rtf = total_cost / max(total_duration, 1e-4)

        report_md += "\n---\n\n"
        report_md += "## 二、 总体指标总览\n\n"
        report_md += f"- **执行总用例数：** {total_cases} 例\n"
        report_md += f"- **平均发音准确率：** **{overall_acc:.2f}%**\n"
        report_md += f"- **端到端平均实时率 (RTF)：** **{overall_rtf:.3f}** (生成耗时 / 音频时长)\n"
        report_md += f"- **综合评估结论：** **{'S级 (商业商用级)' if overall_acc >= 95 else ('A级 (生产可用级)' if overall_acc >= 85 else 'B级 (需优化)')}**\n\n"
        report_md += "---\n\n"
        report_md += "## 三、 详细逐条测试明细\n\n"
        report_md += "| 编号 | 原始输入文本 | ASR 识别结果 | 准确率 | 时长 | RTF |\n"
        report_md += "| :--- | :--- | :--- | :---: | :---: | :---: |\n"

        for cat, items in all_results.items():
            for x in items:
                report_md += f"| `{x['case_id']}` | {x['original_text']} | {x['transcription']} | {x['accuracy']:.1f}% | {x['duration']:.1f}s | {x['rtf']:.2f} |\n"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)

        print("\n" + "="*70)
        print(f"🎉 评测报告生成完成！")
        print(f" • 总体平均准确率: {overall_acc:.1f}% | 平均 RTF: {overall_rtf:.3f}")
        print(f" • 报告已保存至: {report_file}")
        print("="*70 + "\n")
        return report_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Automated Benchmark Runner")
    parser.add_argument("--engine", type=str, default="edge", choices=["edge", "chattts", "melotts", "f5tts", "kokoro"], help="TTS Engine: edge, chattts, melotts, f5tts, or kokoro (default: edge)")
    parser.add_argument("--voice", type=str, default="yunxi", help="Voice for Edge-TTS (yunxi, xiaoxiao, yunjian, xiaoyi, etc.)")
    parser.add_argument("--seed", type=int, default=7869, help="Speaker timbre seed for ChatTTS (default: 7869)")
    parser.add_argument("--category", type=str, default=None, choices=list(BENCHMARK_CORPUS.keys()), help="Category to benchmark")
    parser.add_argument("--all", action="store_true", help="Run full benchmark across all 8 categories")
    args = parser.parse_args()

    runner = TTSBenchmarkRunner(engine_type=args.engine, voice=args.voice, seed=args.seed)

    all_res = {}
    if args.all:
        for cat in BENCHMARK_CORPUS.keys():
            all_res[cat] = runner.run_category(cat)
        runner.generate_report(all_res)
    elif args.category:
        res = runner.run_category(args.category)
        runner.generate_report({args.category: res})
    else:
        print("未指定分类，正在默认运行【日常对话 daily】测试集：")
        res = runner.run_category("daily")
        runner.generate_report({"daily": res})
        print("💡 提示:")
        print(" 运行全部 8 大测试集: python run_tts_benchmark.py --engine edge --all")
        print(" 运行数字读法测试: python run_tts_benchmark.py --engine edge --category numbers")
