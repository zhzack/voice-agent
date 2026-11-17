import os
import json
import httpx
from typing import List, Iterator

from app.core.base import BaseLLMProvider, ChatMessage

class OllamaLLMProvider(BaseLLMProvider):
    """
    Local Ollama / OpenAI-compatible High-Performance LLM Provider.
    Supports Qwen3.8-27B (Q3_K_M / Q4_K_M), Qwen2.5-32B, DeepSeek, etc.
    Seamlessly integrates with AMD ROCm hardware acceleration.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen3.8:27b"
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def is_available(self) -> bool:
        """Checks if the local Ollama / llama.cpp server is reachable."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_local_models(self) -> List[str]:
        """Lists all locally installed models in Ollama."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 250
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": m.role if hasattr(m, 'role') else m["role"],
                 "content": m.content if hasattr(m, 'content') else m["content"]}
                for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json().get("message", {}).get("content", "").strip()
                return f"【Ollama 响应错误 ({res.status_code})】: {res.text}"
        except httpx.ConnectError:
            return "【提示】：无法连接到本地 Ollama 服务，请确认 Ollama 已启动 (http://localhost:11434)。"
        except Exception as e:
            return f"【Ollama 异常】: {str(e)}"

    def stream_generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 250
    ) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": m.role if hasattr(m, 'role') else m["role"],
                 "content": m.content if hasattr(m, 'content') else m["content"]}
                for m in messages
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        yield f"【Ollama 错误 {response.status_code}】"
                        return
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                msg = chunk.get("message", {})
                                content = msg.get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                pass
        except Exception as e:
            yield f"【连接失败: {e}】"
