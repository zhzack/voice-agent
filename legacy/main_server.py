"""
Legacy entry point for VoiceStudio Web Platform.
Forwards to standard application factory in `app.main:app`.
"""
import os
import sys

# Ensure voice_agent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app, create_app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print(" 🚀 VoiceStudio Web 控制台启动中 (via legacy/main_server.py)...")
    print(" 🌐 本地访问地址: http://localhost:8000")
    print(" 📖 API 文档地址: http://localhost:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
