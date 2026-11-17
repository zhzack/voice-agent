import os
import time
import uuid
import numpy as np
import soundfile as sf
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Optional, List, Dict, Any

from app.core.history_manager import history_manager
from app.providers.tts.edge_provider import EdgeTTSProvider
from app.providers.tts.chattts_provider import ChatTTSProvider
from app.providers.tts.melotts_provider import MeloTTSProvider
from app.providers.tts.f5tts_provider import F5TTSProvider
from app.providers.stt.sensevoice_provider import SenseVoiceSTTProvider

router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GENERATED_DIR = os.path.join(BASE_DIR, "app", "static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# Lazy singletons
edge_provider = EdgeTTSProvider()
chattts_provider = None
melotts_provider = None
f5tts_provider = None
kokoro_provider = None
piper_provider = None
stt_provider = None

def get_piper():
    global piper_provider
    if piper_provider is None:
        from app.providers.tts.piper_provider import PiperTTSProvider
        piper_provider = PiperTTSProvider()
    return piper_provider

def get_chattts():
    global chattts_provider
    if chattts_provider is None:
        chattts_provider = ChatTTSProvider()
    return chattts_provider

def get_melotts():
    global melotts_provider
    if melotts_provider is None:
        melotts_provider = MeloTTSProvider()
    return melotts_provider

def get_f5tts():
    global f5tts_provider
    if f5tts_provider is None:
        f5tts_provider = F5TTSProvider()
    return f5tts_provider

def get_kokoro():
    global kokoro_provider
    if kokoro_provider is None:
        from app.providers.tts.kokoro_provider import KokoroTTSProvider
        kokoro_provider = KokoroTTSProvider()
    return kokoro_provider

def get_stt():
    global stt_provider
    if stt_provider is None:
        stt_provider = SenseVoiceSTTProvider()
    return stt_provider

class SynthesisRequest(BaseModel):
    text: str
    engine: str = "edge"  # "edge" | "chattts" | "melotts" | "f5tts" | "kokoro"
    voice_id: str = "yunxi"
    rate: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    verify_asr: bool = True
    # Advanced F5-TTS
    nfe_step: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    # Advanced ChatTTS
    temperature: float = 0.3
    top_p: float = 0.75
    oral: int = 2
    laugh: int = 0
    break_level: int = 0
    # Custom clone
    custom_ref_audio: Optional[str] = None
    custom_ref_text: Optional[str] = None

@router.post("/synthesize")
def synthesize_audio(req: SynthesisRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空")

    engine_type = req.engine.lower()
    file_id = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    print(f"\n[TTS API] ⚡ 收到合成请求 | 引擎: {engine_type.upper()} | 音色: {req.voice_id} | 文本({len(req.text)}字): '{req.text}'")

    # 针对不同模型的语气/停顿标记进行针对性适配
    clean_text = req.text
    if engine_type == "chattts":
        clean_text = clean_text.replace("[break_500ms]", " [uv_break] [uv_break] ")
    elif engine_type == "edge":
        clean_text = clean_text.replace("[break_500ms]", "…… ").replace("[uv_break]", "…… ").replace("[v_break]", "…… ")
        clean_text = clean_text.replace("[laugh]", " ").replace("[sigh]", " ")
    else:
        # f5tts, melotts, kokoro
        clean_text = clean_text.replace("[break_500ms]", "…… ").replace("[uv_break]", "…… ").replace("[v_break]", "…… ")
        clean_text = clean_text.replace("[laugh]", " ").replace("[sigh]", " ")

    try:
        if engine_type == "edge":
            filename = f"{file_id}.mp3"
            out_path = os.path.join(GENERATED_DIR, filename)
            synth_res = edge_provider.synthesize(
                text=clean_text,
                voice_id=req.voice_id,
                rate=req.rate,
                pitch=req.pitch,
                output_path=out_path
            )
        elif engine_type == "chattts":
            filename = f"{file_id}.wav"
            out_path = os.path.join(GENERATED_DIR, filename)
            provider = get_chattts()
            synth_res = provider.synthesize(
                text=clean_text,
                voice_id=req.voice_id,
                temperature=req.temperature,
                top_p=req.top_p,
                oral=req.oral,
                laugh=req.laugh,
                bk=req.break_level,
                output_path=out_path
            )
        elif engine_type == "melotts":
            filename = f"{file_id}.wav"
            out_path = os.path.join(GENERATED_DIR, filename)
            provider = get_melotts()
            synth_res = provider.synthesize(
                text=clean_text,
                voice_id=req.voice_id,
                rate=req.rate,
                output_path=out_path
            )
        elif engine_type == "f5tts":
            filename = f"{file_id}.wav"
            out_path = os.path.join(GENERATED_DIR, filename)
            provider = get_f5tts()
            synth_res = provider.synthesize(
                text=clean_text,
                voice_id=req.voice_id,
                rate=req.rate,
                nfe_step=req.nfe_step,
                cfg_strength=req.cfg_strength,
                sway_sampling_coef=req.sway_sampling_coef,
                custom_ref_audio=req.custom_ref_audio,
                custom_ref_text=req.custom_ref_text,
                output_path=out_path
            )
        elif engine_type == "kokoro":
            filename = f"{file_id}.wav"
            out_path = os.path.join(GENERATED_DIR, filename)
            provider = get_kokoro()
            synth_res = provider.synthesize(
                text=clean_text,
                voice_id=req.voice_id,
                rate=req.rate,
                output_path=out_path
            )
        elif engine_type == "piper":
            filename = f"{file_id}.wav"
            out_path = os.path.join(GENERATED_DIR, filename)
            provider = get_piper()
            synth_res = provider.synthesize(
                text=clean_text,
                voice_id=req.voice_id,
                rate=req.rate,
                output_path=out_path
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的引擎: {engine_type}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"【{engine_type.upper()}】合成推理异常: {str(e)}")

    # 声音增益调整 (Volume)
    if abs(req.volume - 1.0) > 0.02 and os.path.exists(out_path):
        try:
            aud_data, aud_sr = sf.read(out_path)
            aud_data = np.clip(aud_data * req.volume, -1.0, 1.0)
            sf.write(out_path, aud_data, aud_sr)
        except Exception as ve:
            print(f"[TTS Volume] 增益调节跳过: {ve}")

    audio_url = f"/static/generated/{filename}"
    accuracy_report = None

    if req.verify_asr:
        try:
            stt = get_stt()
            accuracy_report = stt.verify(out_path, req.text)
        except Exception as e:
            accuracy_report = {"error": f"ASR 校验失败: {str(e)}"}

    print(f"[TTS API] ✅ 合成完成 | 耗时: {synth_res['cost_sec']:.3f}s | 音频时长: {synth_res['duration']:.2f}s | 静态文件: {filename}")

    # 自动归档至历史数据库
    history_manager.add_record(
        record_id=file_id,
        engine=engine_type,
        voice_id=req.voice_id,
        voice_name=req.voice_id,
        text=req.text,
        audio_url=audio_url,
        duration=round(synth_res["duration"], 2),
        cost_sec=round(synth_res["cost_sec"], 3),
        rtf=round(synth_res["rtf"], 3),
        sample_rate=synth_res["sample_rate"],
        params={
            "rate": req.rate,
            "pitch": req.pitch,
            "volume": req.volume,
            "nfe_step": req.nfe_step,
            "cfg_strength": req.cfg_strength,
            "sway_sampling_coef": req.sway_sampling_coef,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "oral": req.oral,
            "laugh": req.laugh,
            "break_level": req.break_level
        },
        accuracy_report=accuracy_report
    )

    return {
        "id": file_id,
        "audio_url": audio_url,
        "duration": round(synth_res["duration"], 2),
        "cost_sec": round(synth_res["cost_sec"], 3),
        "rtf": round(synth_res["rtf"], 3),
        "sample_rate": synth_res["sample_rate"],
        "engine": engine_type,
        "voice_id": req.voice_id,
        "accuracy_report": accuracy_report
    }

class TagUpdateRequest(BaseModel):
    is_favorite: Optional[bool] = None
    user_tags: Optional[List[str]] = None
    user_notes: Optional[str] = None

@router.get("/history")
def get_synthesis_history(limit: int = 50, favorite_only: bool = False):
    return history_manager.list_records(limit=limit, favorite_only=favorite_only)

@router.post("/history/{record_id}/tag")
def update_history_tag(record_id: str, req: TagUpdateRequest):
    success = history_manager.update_tag_and_note(
        record_id=record_id,
        is_favorite=req.is_favorite,
        user_tags=req.user_tags,
        user_notes=req.user_notes
    )
    if not success:
        raise HTTPException(status_code=404, detail="记录未找到")
    return {"status": "success", "id": record_id}

@router.delete("/history/{record_id}")
def delete_history_item(record_id: str):
    success = history_manager.delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="记录未找到")
    return {"status": "success", "id": record_id}

UPLOAD_DIR = os.path.join(BASE_DIR, "speakers", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload_ref_audio")
async def upload_reference_audio(file: UploadFile = File(...)):
    """
    上传用户自定义参考人声音频（用于 F5-TTS 零样本声音克隆），
    自动调用 SenseVoice 进行语音识别获得参考文本 ref_text。
    """
    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    temp_filename = f"user_ref_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    target_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 规范化转换为 24kHz/16kHz 单声道 WAV 确保 F5-TTS 和 SenseVoice 兼容
    wav_target_path = os.path.splitext(target_path)[0] + ".wav"
    try:
        import soundfile as sf
        import librosa
        audio_data, sr = sf.read(target_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        sf.write(wav_target_path, audio_data, sr)
        audio_duration = len(audio_data) / sr
    except Exception as e:
        wav_target_path = target_path
        audio_duration = 0.0

    # 自动调用 SenseVoice 进行 ASR 识别参考音频中的文字
    recognized_text = ""
    try:
        stt = get_stt()
        asr_res = stt.transcribe(wav_target_path)
        recognized_text = asr_res.get("text", "")
    except Exception as e:
        recognized_text = ""

    return {
        "ref_audio_path": wav_target_path,
        "ref_audio_url": f"/static/uploads/{os.path.basename(wav_target_path)}",
        "recognized_text": recognized_text,
        "duration": round(audio_duration, 2),
        "filename": os.path.basename(wav_target_path)
    }

def cleanup_generated_audio(max_age_seconds: int = 86400, max_files: int = 200) -> Dict[str, int]:
    """Clean up old generated audio files to conserve disk space."""
    if not os.path.exists(GENERATED_DIR):
        return {"deleted": 0, "remaining": 0}

    files = []
    now = time.time()
    for fname in os.listdir(GENERATED_DIR):
        if fname.startswith("gen_") or fname.startswith("call_") or fname.endswith((".wav", ".mp3")):
            fpath = os.path.join(GENERATED_DIR, fname)
            if os.path.isfile(fpath):
                files.append((fpath, os.path.getmtime(fpath)))

    deleted_count = 0
    # 1. Delete files older than max_age_seconds
    for fpath, mtime in files:
        if now - mtime > max_age_seconds:
            try:
                os.remove(fpath)
                deleted_count += 1
            except Exception:
                pass

    # 2. If still more than max_files, delete oldest
    remaining = [f for f in files if os.path.exists(f[0])]
    if len(remaining) > max_files:
        remaining.sort(key=lambda x: x[1])
        to_delete = len(remaining) - max_files
        for fpath, _ in remaining[:to_delete]:
            try:
                os.remove(fpath)
                deleted_count += 1
            except Exception:
                pass

    remaining_count = len([f for f in os.listdir(GENERATED_DIR) if os.path.isfile(os.path.join(GENERATED_DIR, f))])
    return {"deleted": deleted_count, "remaining": remaining_count}

@router.delete("/cache")
def clear_audio_cache(max_age_hours: int = 1):
    """清理历史生成的语音缓存文件"""
    res = cleanup_generated_audio(max_age_seconds=max_age_hours * 3600, max_files=50)
    return {"status": "success", "cleaned": res["deleted"], "remaining": res["remaining"]}

