import os
import sys
import time
import json
import asyncio
import tempfile
import threading
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.providers.stt.sensevoice_provider import (
    SenseVoiceSTTProvider,
    LiveMicRecognizer,
    query_microphones,
    find_preferred_microphone
)
from app.api.routes_call import _convert_to_wav

router = APIRouter(prefix="/api/v1/stt", tags=["STT"])

_live_mic_recognizer: Optional[LiveMicRecognizer] = None
_stt_provider: Optional[SenseVoiceSTTProvider] = None

def get_live_mic_recognizer() -> LiveMicRecognizer:
    global _live_mic_recognizer
    if _live_mic_recognizer is None:
        _live_mic_recognizer = LiveMicRecognizer()
    return _live_mic_recognizer

def get_stt_provider() -> SenseVoiceSTTProvider:
    global _stt_provider
    if _stt_provider is None:
        _stt_provider = SenseVoiceSTTProvider()
    return _stt_provider

class MicRecordRequest(BaseModel):
    duration: float = 5.0
    device_id: Optional[int] = None

@router.get("/devices")
def list_microphones():
    """获取本机所有可用音频输入麦克风设备列表及推荐默认项"""
    devices = query_microphones()
    pref_id, pref_name = find_preferred_microphone()
    return {
        "devices": devices,
        "preferred_device": {
            "id": pref_id,
            "name": pref_name
        }
    }

@router.get("/mic/stream")
async def stream_mic_recognition(
    request: Request,
    device_id: Optional[int] = None,
    max_duration: Optional[float] = None,
    min_silence: float = 0.45,
    min_speech: float = 0.25
):
    """
    调起本机麦克风，边录音边识别 (Server-Sent Events / SSE 流式接口)
    用户每说完一句话，VAD 自动切句并由 SenseVoice 解码，实时推送结果。
    """
    recognizer = get_live_mic_recognizer()
    async_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def mic_worker():
        try:
            for item in recognizer.listen(
                device_id=device_id,
                min_silence_duration=min_silence,
                min_speech_duration=min_speech,
                max_duration=max_duration,
                stop_event=stop_event
            ):
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(async_queue.put_nowait, item)
        except Exception as e:
            err_payload = {"type": "error", "message": str(e)}
            loop.call_soon_threadsafe(async_queue.put_nowait, err_payload)
        finally:
            loop.call_soon_threadsafe(async_queue.put_nowait, None)

    worker_thread = threading.Thread(target=mic_worker, daemon=True)
    worker_thread.start()

    async def sse_generator():
        # 发送起始连接握手帧
        dev_id, dev_name = (device_id, f"Device-{device_id}") if device_id is not None else find_preferred_microphone()
        init_info = {
            "type": "ready",
            "message": "麦克风已开启，正在监听语音输入...",
            "device_id": dev_id,
            "device_name": dev_name
        }
        yield f"data: {json.dumps(init_info, ensure_ascii=False)}\n\n"

        try:
            while not stop_event.is_set():
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(async_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 心跳保持连接
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    break

                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@router.post("/mic/record")
def record_and_transcribe(req: MicRecordRequest):
    """
    调起麦克风录制指定秒数（默认 5s）并自动执行 ASR 转文字，直接返回识别文本
    """
    if req.duration <= 0 or req.duration > 60:
        raise HTTPException(status_code=400, detail="录音时长必须在 1 到 60 秒之间")

    recognizer = get_live_mic_recognizer()
    try:
        res = recognizer.record_and_transcribe(
            duration=req.duration,
            device_id=req.device_id
        )
        return {
            "status": "success",
            "text": res["text"],
            "duration": res["duration"],
            "cost_sec": res["cost_sec"],
            "record_time_sec": res["record_time_sec"],
            "rtf": res["rtf"],
            "device_id": res["device_id"],
            "device_name": res["device_name"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"麦克风录音识别失败: {str(e)}")

@router.post("/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    标准文件 ASR 识别接口：上传任意常见音频文件，返回 SenseVoice 转写文字
    """
    stt = get_stt_provider()
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    fd_in, temp_input = tempfile.mkstemp(suffix=ext)
    os.close(fd_in)

    content = await file.read()
    with open(temp_input, "wb") as f:
        f.write(content)

    fd_wav, wav_audio = tempfile.mkstemp(suffix=".wav")
    os.close(fd_wav)

    try:
        if not _convert_to_wav(temp_input, wav_audio, sample_rate=16000):
            wav_audio = temp_input

        res = stt.transcribe(wav_audio)
        return {
            "status": "success",
            "text": res["text"],
            "duration": round(res["duration"], 2),
            "cost_sec": round(res["cost_sec"], 3),
            "rtf": round(res["rtf"], 3),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频识别失败: {str(e)}")
    finally:
        for p in [temp_input, wav_audio]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

@router.websocket("/ws")
async def websocket_stt_endpoint(websocket: WebSocket):
    """
    WebSocket 实时双向流式转录接口
    支持客户端控制服务端麦克风监听，或客户端向服务端推流 PCM 音频分块识别
    """
    await websocket.accept()
    recognizer = get_live_mic_recognizer()
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()
    async_queue: asyncio.Queue = asyncio.Queue()

    async def forward_queue_to_ws():
        while not stop_event.is_set():
            item = await async_queue.get()
            if item is None:
                break
            try:
                await websocket.send_json(item)
            except Exception:
                break

    forward_task = asyncio.create_task(forward_queue_to_ws())

    def mic_worker(device_id: Optional[int] = None):
        try:
            for item in recognizer.listen(device_id=device_id, stop_event=stop_event):
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(async_queue.put_nowait, item)
        except Exception as e:
            loop.call_soon_threadsafe(async_queue.put_nowait, {"type": "error", "message": str(e)})

    mic_thread: Optional[threading.Thread] = None

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await websocket.send_json({"type": "error", "message": "消息格式错误，需为 JSON"})
                continue

            action = data.get("action")
            if action == "start_mic":
                dev_id = data.get("device_id")
                if mic_thread is None or not mic_thread.is_alive():
                    stop_event.clear()
                    mic_thread = threading.Thread(target=mic_worker, args=(dev_id,), daemon=True)
                    mic_thread.start()
                    await websocket.send_json({"type": "ready", "message": "服务端本地麦克风监听已启动"})
            elif action == "stop_mic":
                stop_event.set()
                await websocket.send_json({"type": "stopped", "message": "麦克风监听已停止"})
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        await async_queue.put(None)
        if forward_task:
            forward_task.cancel()
