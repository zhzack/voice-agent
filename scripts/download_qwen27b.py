"""
ModelScope / HuggingFace Fast Download Script for Qwen3.8-27B GGUF Models.
Supports Q3_K_M (12.2GB) and Q4_K_M (15.5GB).
"""
import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "q3": {
        "name": "Qwen3.8-27B (Q3_K_M 极速纯显存版)",
        "file": "qwen3.8-27b-instruct-q3_k_m.gguf",
        "approx_size_gb": 12.2,
        "model_id": "Qwen/Qwen3.8-27B-GGUF"
    },
    "q4": {
        "name": "Qwen3.8-27B (Q4_K_M 推荐混合分层版)",
        "file": "qwen3.8-27b-instruct-q4_k_m.gguf",
        "approx_size_gb": 15.5,
        "model_id": "Qwen/Qwen3.8-27B-GGUF"
    }
}

def download_model(variant="q3"):
    cfg = MODEL_CONFIGS.get(variant, MODEL_CONFIGS["q3"])
    target_path = MODELS_DIR / cfg["file"]
    print("=" * 65)
    print(f" 📥 准备下载: {cfg['name']}")
    print(f" 💾 文件路径: {target_path}")
    print(f" 📦 预计体积: ~{cfg['approx_size_gb']} GB")
    print("=" * 65)

    if target_path.exists() and target_path.stat().st_size > 10 * 1024 * 1024 * 1024:
        print(f"✅ 文件已存在且完整 ({target_path.stat().st_size / (1024**3):.2f} GB)，无需重新下载！")
        return str(target_path)

    try:
        from modelscope.hub.file_download import model_file_download
        print("\n🚀 正在通过 ModelScope (魔搭社区) 高速通道下载权重...")
        downloaded = model_file_download(
            model_id=cfg["model_id"],
            file_path=cfg["file"],
            local_dir=str(MODELS_DIR)
        )
        print(f"🎉 下载成功: {downloaded}")
        return downloaded
    except Exception as e:
        print(f"\n⚠️ ModelScope 下载遇到问题: {e}")
        print("\n您也可以直接使用镜像加速或手动下载放入 models/ 目录：")
        print(f"👉 目标文件名: {target_path}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 Qwen3.8-27B GGUF 权重")
    parser.add_argument("--variant", choices=["q3", "q4"], default="q3", help="选择规格: q3 (12.2G) 或 q4 (15.5G)")
    args = parser.parse_args()
    download_model(args.variant)

