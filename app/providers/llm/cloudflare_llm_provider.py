import os
import json
import httpx
from typing import List, Iterator

from app.core.base import BaseLLMProvider, ChatMessage

class CloudflareWorkersAILLMProvider(BaseLLMProvider):
    """
    Cloudflare Workers AI / OpenAI Compatible LLM Provider.
    Supports @cf/meta/llama-3-8b-instruct, @cf/qwen/qwen1.5-14b-chat, etc.
    """
    def __init__(
        self,
        account_id: str = "",
        api_token: str = "",
        model_name: str = "@cf/meta/llama-3-8b-instruct",
        base_url: str = ""
    ):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        self.model_name = model_name
        self.base_url = base_url

    def _get_endpoint(self) -> str:
        if self.base_url:
            return self.base_url
        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"

    def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 150
    ) -> str:
        if not self.api_token or not self.account_id:
            return "【提示】：未配置 Cloudflare Account ID 或 API Token，请在平台设置中配置。"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self._get_endpoint(), headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("result", {}).get("response", str(data))
        except Exception as e:
            return f"Cloudflare API 请求异常: {str(e)}"

    def stream_generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 150
    ) -> Iterator[str]:
        # Simple fallback for single turn
        full = self.generate(messages, temperature, top_p, max_tokens)
        for char in full:
            yield char
