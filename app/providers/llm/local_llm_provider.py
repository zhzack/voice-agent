import os
import time
from typing import List, Iterator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

from app.core.base import BaseLLMProvider, ChatMessage

DEFAULT_LLM_CACHE = os.path.expanduser(
    r"~/.cache/modelscope/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
)
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", DEFAULT_LLM_CACHE)

class LocalQwenLLMProvider(BaseLLMProvider):
    def __init__(self, model_path: str = LLM_MODEL_PATH):
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    print(f"正在加载本地 Qwen2.5 模型 ({self.model_path}) 至 GPU...")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map=device
                    )

    def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 150
    ) -> str:
        self._ensure_loaded()
        msg_dicts = [
            {"role": m.role if hasattr(m, 'role') else m["role"], "content": m.content if hasattr(m, 'content') else m["content"]}
            for m in messages
        ]
        text_input = self._tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self._tokenizer([text_input], return_tensors="pt").to(device)

        with self._lock:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
        # Strip input tokens
        output_tokens = outputs[0][len(inputs.input_ids[0]):]
        return self._tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    def stream_generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 150
    ) -> Iterator[str]:
        self._ensure_loaded()
        msg_dicts = [
            {"role": m.role if hasattr(m, 'role') else m["role"], "content": m.content if hasattr(m, 'content') else m["content"]}
            for m in messages
        ]
        text_input = self._tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text_input], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        thread = threading.Thread(target=self._model.generate, kwargs=kwargs)
        thread.start()

        for token in streamer:
            yield token
