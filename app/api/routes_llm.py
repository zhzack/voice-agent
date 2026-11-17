import os
from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from app.core.base import ChatMessage
from app.providers.llm.local_llm_provider import LocalQwenLLMProvider
from app.providers.llm.cloudflare_llm_provider import CloudflareWorkersAILLMProvider
from app.providers.llm.ollama_provider import OllamaLLMProvider

router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])

local_llm = None
cloudflare_llm = CloudflareWorkersAILLMProvider()
ollama_llm = OllamaLLMProvider()

def get_local_llm():
    global local_llm
    if local_llm is None:
        local_llm = LocalQwenLLMProvider()
    return local_llm

def get_ollama_llm(model_name: Optional[str] = None):
    return OllamaLLMProvider(model_name=model_name or "qwen3.8:27b")

class ChatRequest(BaseModel):
    provider: str = "local"  # "local" | "cloudflare" | "ollama"
    model_name: Optional[str] = "qwen3.8:27b"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 150
    cf_account_id: Optional[str] = None
    cf_api_token: Optional[str] = None
    cf_model: Optional[str] = "@cf/meta/llama-3-8b-instruct"

@router.get("/providers")
def list_providers():
    ollama_ready = ollama_llm.is_available()
    ollama_models = ollama_llm.list_local_models() if ollama_ready else []

    return [
        {
            "id": "local",
            "name": "本地 Qwen2.5-1.5B-Instruct (GPU加速)",
            "device": "AMD Radeon RX 9070 XT",
            "status": "ready",
            "desc": "完全离线运行、首字延迟 20ms、无网络依赖"
        },
        {
            "id": "ollama",
            "name": "本地 Qwen3.8-27B (ROCm / 混合卸载加速)",
            "device": "AMD Radeon RX 9070 XT + 9800X3D",
            "status": "ready" if ollama_ready else "not_started",
            "desc": "支持 27B 大模型：Q3_K_M (100% 显存) 与 Q4_K_M (显存+内存混合)",
            "available_models": ollama_models
        },
        {
            "id": "cloudflare",
            "name": "Cloudflare Workers AI (云端无服务器模型)",
            "device": "Cloudflare Global Edge",
            "status": "configurable",
            "desc": "支持 Llama 3、Qwen1.5-14B 等云端大模型，免显存消耗"
        }
    ]

@router.post("/chat")
def chat_completion(req: ChatRequest):
    if req.provider == "local":
        provider = get_local_llm()
        reply = provider.generate(
            messages=req.messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens
        )
    elif req.provider == "ollama":
        provider = OllamaLLMProvider(model_name=req.model_name or "qwen3.8:27b")
        reply = provider.generate(
            messages=req.messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens
        )
    elif req.provider == "cloudflare":
        cf = CloudflareWorkersAILLMProvider(
            account_id=req.cf_account_id or "",
            api_token=req.cf_api_token or "",
            model_name=req.cf_model or "@cf/meta/llama-3-8b-instruct"
        )
        reply = cf.generate(
            messages=req.messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens
        )
    else:
        raise HTTPException(status_code=400, detail=f"未知提供商: {req.provider}")

    return {"reply": reply}
