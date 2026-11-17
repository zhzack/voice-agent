import os
import time
import re
import json
import uuid
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.core.base import ChatMessage
from app.core.history_manager import history_manager
from app.api.routes_tts import (
    get_stt, get_chattts, get_melotts, get_f5tts, get_kokoro, get_piper, edge_provider,
    GENERATED_DIR
)
from app.api.routes_llm import get_local_llm, get_ollama_llm, CloudflareWorkersAILLMProvider

router = APIRouter(prefix="/api/v1/call", tags=["VoiceCall"])

def _convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """使用 imageio-ffmpeg 提供的二进制工具将任意输入音频转为 16kHz/24kHz 单声道标准 WAV"""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-ac", "1", "-ar", str(sample_rate),
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100
    except Exception as e:
        print(f"[Audio Convert] FFmpeg 转换错误: {e}")
        return False

class TextCallRequest(BaseModel):
    text: str
    llm_provider: str = "local"  # "local" | "cloudflare" | "ollama"
    llm_model: Optional[str] = None
    system_prompt: str = "你是一个善解人意的实时智能语音伴侣，语言简明亲切，回答控制在1-2句话内。"
    tts_engine: str = "edge"  # "edge" | "kokoro" | "chattts" | "f5tts" | "melotts" | "piper"
    tts_voice: str = "yunxi"
    tts_rate: Optional[float] = None
    rate: Optional[float] = None
    tts_pitch: float = 0.0
    tts_volume: Optional[float] = None
    volume: Optional[float] = None
    temperature: float = 0.7
    chat_history: Optional[List[Any]] = None
    history: Optional[List[Any]] = None
    cf_account_id: Optional[str] = None
    cf_api_token: Optional[str] = None

def _run_tts(engine: str, voice_id: str, text: str, rate: float, pitch: float, volume: float, file_id: str) -> Dict[str, Any]:
    engine = engine.lower()
    clean_text = text

    if engine == "chattts":
        clean_text = clean_text.replace("[break_500ms]", " [uv_break] [uv_break] ")
    elif engine == "edge":
        clean_text = clean_text.replace("[break_500ms]", "…… ").replace("[uv_break]", "…… ").replace("[v_break]", "…… ")
        clean_text = clean_text.replace("[laugh]", " ").replace("[sigh]", " ")
    else:
        clean_text = clean_text.replace("[break_500ms]", "…… ").replace("[uv_break]", "…… ").replace("[v_break]", "…… ")
        clean_text = clean_text.replace("[laugh]", " ").replace("[sigh]", " ")

    if engine == "edge":
        filename = f"{file_id}.mp3"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = edge_provider.synthesize(text=clean_text, voice_id=voice_id, rate=rate, pitch=pitch, output_path=out_path)
    elif engine == "kokoro":
        filename = f"{file_id}.wav"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = get_kokoro().synthesize(text=clean_text, voice_id=voice_id, rate=rate, output_path=out_path)
    elif engine == "chattts":
        filename = f"{file_id}.wav"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = get_chattts().synthesize(text=clean_text, voice_id=voice_id, rate=rate, output_path=out_path)
    elif engine == "f5tts":
        filename = f"{file_id}.wav"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = get_f5tts().synthesize(text=clean_text, voice_id=voice_id, rate=rate, output_path=out_path)
    elif engine == "piper":
        filename = f"{file_id}.wav"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = get_piper().synthesize(text=clean_text, voice_id=voice_id, rate=rate, output_path=out_path)
    elif engine == "melotts":
        filename = f"{file_id}.wav"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = get_melotts().synthesize(text=clean_text, voice_id=voice_id, rate=rate, output_path=out_path)
    else:
        filename = f"{file_id}.mp3"
        out_path = os.path.join(GENERATED_DIR, filename)
        res = edge_provider.synthesize(text=clean_text, voice_id=voice_id, rate=rate, pitch=pitch, output_path=out_path)

    # Volume scaling
    if abs(volume - 1.0) > 0.02 and os.path.exists(out_path):
        try:
            aud, sr = sf.read(out_path)
            aud = np.clip(aud * volume, -1.0, 1.0)
            sf.write(out_path, aud, sr)
        except Exception:
            pass

    return {
        "filename": filename,
        "out_path": out_path,
        "duration": res["duration"],
        "cost_sec": res["cost_sec"],
        "rtf": res["rtf"],
        "sample_rate": res["sample_rate"]
    }

@router.post("/text_turn")
def text_call_turn(req: TextCallRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="对话内容不能为空")

    effective_rate = req.tts_rate if req.tts_rate is not None else (req.rate if req.rate is not None else 1.0)
    effective_volume = req.tts_volume if req.tts_volume is not None else (req.volume if req.volume is not None else 1.0)
    raw_history = req.chat_history or req.history or []

    # 1. LLM Generation
    messages = [{"role": "system", "content": req.system_prompt}]
    for m in raw_history[-10:]:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": req.text})

    if req.llm_provider == "local":
        provider = get_local_llm()
        reply = provider.generate(messages=messages, temperature=req.temperature, max_tokens=150)
    elif req.llm_provider == "ollama":
        provider = get_ollama_llm(req.llm_model)
        reply = provider.generate(messages=messages, temperature=req.temperature, max_tokens=150)
    else:
        cf = CloudflareWorkersAILLMProvider(
            account_id=req.cf_account_id or "",
            api_token=req.cf_api_token or ""
        )
        reply = cf.generate(messages=messages, temperature=req.temperature, max_tokens=150)

    # 2. TTS Synthesis
    file_id = f"call_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    synth = _run_tts(
        engine=req.tts_engine,
        voice_id=req.tts_voice,
        text=reply,
        rate=effective_rate,
        pitch=req.tts_pitch,
        volume=effective_volume,
        file_id=file_id
    )
    audio_url = f"/static/generated/{synth['filename']}"

    # Auto archive to history & dialogue stream
    history_manager.add_record(
        record_id=file_id,
        engine=req.tts_engine,
        voice_id=req.tts_voice,
        voice_name=req.tts_voice,
        text=reply,
        audio_url=audio_url,
        duration=round(synth["duration"], 2),
        cost_sec=round(synth["cost_sec"], 3),
        rtf=round(synth["rtf"], 3),
        sample_rate=synth["sample_rate"],
        params={"rate": effective_rate, "pitch": req.tts_pitch, "volume": effective_volume, "mode": "call_text", "user_text": req.text}
    )
    history_manager.add_dialogue_turn(
        turn_id=file_id,
        user_text=req.text,
        assistant_reply=reply,
        audio_url=audio_url,
        engine=req.tts_engine,
        voice_id=req.tts_voice,
        duration=round(synth["duration"], 2),
        cost_sec=round(synth["cost_sec"], 3),
        rtf=round(synth["rtf"], 3),
        mode="text"
    )

    return {
        "user_text": req.text,
        "assistant_reply": reply,
        "ai_text": reply,
        "audio_url": audio_url,
        "duration": round(synth["duration"], 2),
        "cost_sec": round(synth["cost_sec"], 3),
        "rtf": round(synth["rtf"], 3),
        "engine": req.tts_engine,
        "voice_id": req.tts_voice
    }

@router.post("/stream_turn")
async def stream_call_turn(req: TextCallRequest):
    """
    流式低延迟语音对话端点 (Server-Sent Events / SSE):
    大模型逐 Token 输出，遇标点智能切句立即送入 TTS 合成，首句延迟 < 300ms
    """
    effective_rate = req.tts_rate if req.tts_rate is not None else (req.rate if req.rate is not None else 1.0)
    effective_volume = req.tts_volume if req.tts_volume is not None else (req.volume if req.volume is not None else 1.0)

    history = req.chat_history or req.history or []
    messages = [{"role": "system", "content": req.system_prompt}]
    for m in history[-10:]:
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        content = m.get("content", "") if isinstance(m, dict) else ""
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": req.text})

    def event_generator():
        if req.llm_provider == "local":
            provider = get_local_llm()
            token_iter = provider.stream_generate(messages=messages, temperature=req.temperature, max_tokens=150)
        elif req.llm_provider == "ollama":
            provider = get_ollama_llm(req.llm_model)
            token_iter = provider.stream_generate(messages=messages, temperature=req.temperature, max_tokens=150)
        else:
            cf = CloudflareWorkersAILLMProvider(
                account_id=req.cf_account_id or "",
                api_token=req.cf_api_token or ""
            )
            token_iter = iter([cf.generate(messages=messages, temperature=req.temperature, max_tokens=150)])

        buffer = ""
        full_reply = ""
        chunk_idx = 0
        split_pattern = re.compile(r'([。！？!?；;\n]+)')

        for token in token_iter:
            buffer += token
            full_reply += token
            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            # Check if punctuation boundary reached
            parts = split_pattern.split(buffer)
            if len(parts) > 1:
                sentence = parts[0] + parts[1]
                buffer = "".join(parts[2:])
                if sentence.strip():
                    file_id = f"stream_{int(time.time())}_{chunk_idx}_{uuid.uuid4().hex[:4]}"
                    synth = _run_tts(
                        engine=req.tts_engine,
                        voice_id=req.tts_voice,
                        text=sentence,
                        rate=effective_rate,
                        pitch=req.tts_pitch,
                        volume=effective_volume,
                        file_id=file_id
                    )
                    audio_url = f"/static/generated/{synth['filename']}"
                    chunk_payload = {
                        "type": "audio_chunk",
                        "index": chunk_idx,
                        "text": sentence,
                        "audio_url": audio_url,
                        "duration": round(synth["duration"], 2)
                    }
                    chunk_idx += 1
                    yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"

        # Tail buffer if any
        if buffer.strip():
            file_id = f"stream_{int(time.time())}_{chunk_idx}_{uuid.uuid4().hex[:4]}"
            synth = _run_tts(
                engine=req.tts_engine,
                voice_id=req.tts_voice,
                text=buffer,
                rate=effective_rate,
                pitch=req.tts_pitch,
                volume=effective_volume,
                file_id=file_id
            )
            audio_url = f"/static/generated/{synth['filename']}"
            chunk_payload = {
                "type": "audio_chunk",
                "index": chunk_idx,
                "text": buffer,
                "audio_url": audio_url,
                "duration": round(synth["duration"], 2)
            }
            yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"

        done_payload = {
            "type": "done",
            "full_reply": full_reply,
            "total_chunks": chunk_idx + (1 if buffer.strip() else 0)
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/voice_turn")
async def voice_call_turn(
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    llm_provider: str = Form("local"),
    system_prompt: str = Form("你是一个善解人意的实时智能语音伴侣，语言简明亲切，回答控制在1-2句话内。"),
    tts_engine: str = Form("edge"),
    tts_voice: str = Form("yunxi"),
    tts_rate: Optional[float] = Form(None),
    rate: Optional[float] = Form(None),
    tts_pitch: float = Form(0.0),
    tts_volume: Optional[float] = Form(None),
    volume: Optional[float] = Form(None),
    temperature: float = Form(0.7),
    chat_history_json: Optional[str] = Form(None),
    history_json: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    cf_account_id: Optional[str] = Form(None),
    cf_api_token: Optional[str] = Form(None)
):
    upload_file = file or audio
    if not upload_file:
        raise HTTPException(status_code=400, detail="未收到音频文件，请确认麦克风录音权限并重试")

    effective_rate = tts_rate if tts_rate is not None else (rate if rate is not None else 1.0)
    effective_volume = tts_volume if tts_volume is not None else (volume if volume is not None else 1.0)
    raw_history = chat_history_json or history_json or "[]"

    # 1. Save uploaded microphone audio
    ext = os.path.splitext(upload_file.filename or "")[1].lower() or ".webm"
    fd, temp_audio = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    content = await upload_file.read()
    with open(temp_audio, "wb") as f:
        f.write(content)

    # 2. Convert to 16kHz mono WAV for SenseVoice ASR
    fd_wav, wav_audio = tempfile.mkstemp(suffix=".wav")
    os.close(fd_wav)
    if not _convert_to_wav(temp_audio, wav_audio, sample_rate=16000):
        # Fallback if ffmpeg skipped or already wav
        wav_audio = temp_audio

    # 3. ASR Transcription via SenseVoice
    user_text = ""
    try:
        stt = get_stt()
        asr_res = stt.transcribe(wav_audio)
        user_text = asr_res.get("text", "").strip()
    except Exception as e:
        print(f"[ASR Call Error] {e}")
        user_text = ""
    finally:
        for p in [temp_audio, wav_audio]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # If no speech detected, return vocal prompt
    if not user_text:
        fallback_msg = "抱歉，刚刚没有听清您说的话，可以再说一遍吗？"
        file_id = f"call_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        synth = _run_tts(
            engine=tts_engine,
            voice_id=tts_voice,
            text=fallback_msg,
            rate=effective_rate,
            pitch=tts_pitch,
            volume=effective_volume,
            file_id=file_id
        )
        audio_url = f"/static/generated/{synth['filename']}"
        return {
            "user_text": "（未检测到清晰语音）",
            "assistant_reply": fallback_msg,
            "ai_text": fallback_msg,
            "audio_url": audio_url,
            "duration": round(synth["duration"], 2),
            "cost_sec": round(synth["cost_sec"], 3),
            "rtf": round(synth["rtf"], 3),
            "engine": tts_engine,
            "voice_id": tts_voice
        }

    # 4. LLM Generation
    try:
        history = json.loads(raw_history)
    except Exception:
        history = []

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-10:]:
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        content = m.get("content", "") if isinstance(m, dict) else ""
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    if llm_provider == "local":
        provider = get_local_llm()
        reply = provider.generate(messages=messages, temperature=temperature, max_tokens=150)
    elif llm_provider == "ollama":
        provider = get_ollama_llm(llm_model)
        reply = provider.generate(messages=messages, temperature=temperature, max_tokens=150)
    else:
        cf = CloudflareWorkersAILLMProvider(
            account_id=cf_account_id or "",
            api_token=cf_api_token or ""
        )
        reply = cf.generate(messages=messages, temperature=temperature, max_tokens=150)

    # 5. TTS Synthesis
    file_id = f"call_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    synth = _run_tts(
        engine=tts_engine,
        voice_id=tts_voice,
        text=reply,
        rate=effective_rate,
        pitch=tts_pitch,
        volume=effective_volume,
        file_id=file_id
    )
    audio_url = f"/static/generated/{synth['filename']}"

    # Auto archive to history & dialogue stream
    history_manager.add_record(
        record_id=file_id,
        engine=tts_engine,
        voice_id=tts_voice,
        voice_name=tts_voice,
        text=reply,
        audio_url=audio_url,
        duration=round(synth["duration"], 2),
        cost_sec=round(synth["cost_sec"], 3),
        rtf=round(synth["rtf"], 3),
        sample_rate=synth["sample_rate"],
        params={"rate": effective_rate, "pitch": tts_pitch, "volume": effective_volume, "mode": "call_voice", "user_text": user_text}
    )
    history_manager.add_dialogue_turn(
        turn_id=file_id,
        user_text=user_text,
        assistant_reply=reply,
        audio_url=audio_url,
        engine=tts_engine,
        voice_id=tts_voice,
        duration=round(synth["duration"], 2),
        cost_sec=round(synth["cost_sec"], 3),
        rtf=round(synth["rtf"], 3),
        mode="voice"
    )

    return {
        "user_text": user_text,
        "assistant_reply": reply,
        "ai_text": reply,
        "audio_url": audio_url,
        "duration": round(synth["duration"], 2),
        "cost_sec": round(synth["cost_sec"], 3),
        "rtf": round(synth["rtf"], 3),
        "engine": tts_engine,
        "voice_id": tts_voice
    }

@router.get("/history")
def get_call_dialogue_history():
    """获取全双工实时通话多轮对话持久化记录流"""
    return history_manager.list_dialogue_history(limit=100)

@router.delete("/history")
def clear_call_dialogue_history():
    """清空通话多轮对话历史"""
    history_manager.clear_dialogue_history()
    return {"status": "success"}
