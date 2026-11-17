from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from pydantic import BaseModel, Field

class VoiceMetadata(BaseModel):
    id: str
    name: str
    engine: str
    gender: str  # 'male' | 'female'
    locale: str
    desc: str
    sample_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_default: bool = False

class SynthesisResult(BaseModel):
    audio_url: str
    duration: float
    cost_sec: float
    rtf: float
    sample_rate: int
    engine: str
    voice_id: str
    accuracy_report: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class BaseTTSProvider(ABC):
    @abstractmethod
    def list_voices(self) -> List[VoiceMetadata]:
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: str,
        rate: float = 1.0,
        pitch: float = 0.0,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 150
    ) -> str:
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 150
    ) -> Iterator[str]:
        pass

class BaseSTTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def verify(self, audio_path: str, expected_text: str) -> Dict[str, Any]:
        pass
