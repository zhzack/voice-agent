import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure voice_agent directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.api.routes_voices import router as voices_router
from app.api.routes_tts import router as tts_router
from app.api.routes_llm import router as llm_router
from app.api.routes_benchmark import router as benchmark_router
from app.api.routes_telemetry import router as telemetry_router
from app.api.routes_call import router as call_router
from app.api.routes_stt import router as stt_router

def create_app() -> FastAPI:
    """Application factory for VoiceStudio API and SPA."""
    application = FastAPI(
        title="VoiceStudio - 智能语音与大模型管理平台",
        description="支持多大模型编排、多TTS语音引擎、全音色矩阵库、自动化ASR客观评测与实时通话的一体化管理工作台",
        version="1.0.0"
    )

    # CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    application.include_router(voices_router)
    application.include_router(tts_router)
    application.include_router(llm_router)
    application.include_router(benchmark_router)
    application.include_router(telemetry_router)
    application.include_router(call_router)
    application.include_router(stt_router)

    # Ensure and mount static directories
    static_dir = os.path.join(BASE_DIR, "app", "static")
    uploads_dir = os.path.join(BASE_DIR, "speakers", "uploads")
    generated_dir = os.path.join(static_dir, "generated")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    application.mount("/static/uploads", StaticFiles(directory=uploads_dir), name="uploads")
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/health")
    def health_check():
        return {"status": "ok", "app": "VoiceStudio", "version": "1.0.0"}

    @application.get("/")
    def serve_index():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "VoiceStudio Backend API is running. index.html not found."}

    return application

app = create_app()

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print(" 🚀 VoiceStudio Web 控制台启动中...")
    print(" 🌐 本地访问地址: http://localhost:8000")
    print(" 📖 API 文档地址: http://localhost:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
