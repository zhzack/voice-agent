import os
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "voice_studio_history.db")

class HistoryManager:
    """
    Manages persistent voice synthesis history, experiment parameters,
    SenseVoice objective evaluation results, and user tags/annotations.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS synthesis_history (
                    id TEXT PRIMARY KEY,
                    created_at REAL,
                    time_str TEXT,
                    engine TEXT,
                    voice_id TEXT,
                    voice_name TEXT,
                    text TEXT,
                    audio_url TEXT,
                    duration REAL,
                    cost_sec REAL,
                    rtf REAL,
                    sample_rate INTEGER,
                    params_json TEXT,
                    accuracy REAL,
                    cer REAL,
                    evaluation_status TEXT,
                    is_favorite INTEGER DEFAULT 0,
                    user_tags TEXT DEFAULT '',
                    user_notes TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_time ON synthesis_history(created_at DESC)")

            # 全双工多轮通话持久化对话记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS call_dialogue_history (
                    id TEXT PRIMARY KEY,
                    created_at REAL,
                    time_str TEXT,
                    user_text TEXT,
                    assistant_reply TEXT,
                    audio_url TEXT,
                    engine TEXT,
                    voice_id TEXT,
                    duration REAL,
                    cost_sec REAL,
                    rtf REAL,
                    mode TEXT DEFAULT 'voice'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_time ON call_dialogue_history(created_at ASC)")

    def add_record(
        self,
        record_id: str,
        engine: str,
        voice_id: str,
        voice_name: str,
        text: str,
        audio_url: str,
        duration: float,
        cost_sec: float,
        rtf: float,
        sample_rate: int,
        params: Optional[Dict[str, Any]] = None,
        accuracy_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        now = time.time()
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        
        acc = accuracy_report.get("accuracy", None) if accuracy_report else None
        cer = accuracy_report.get("cer", None) if accuracy_report else None
        status = accuracy_report.get("status", "") if accuracy_report else ""

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO synthesis_history (
                    id, created_at, time_str, engine, voice_id, voice_name, text,
                    audio_url, duration, cost_sec, rtf, sample_rate, params_json,
                    accuracy, cer, evaluation_status, is_favorite, user_tags, user_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', '')
            """, (
                record_id, now, time_str, engine, voice_id, voice_name, text,
                audio_url, duration, cost_sec, rtf, sample_rate, json.dumps(params or {}, ensure_ascii=False),
                acc, cer, status
            ))

        return {
            "id": record_id,
            "created_at": now,
            "time_str": time_str,
            "engine": engine,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "text": text,
            "audio_url": audio_url,
            "duration": duration,
            "cost_sec": cost_sec,
            "rtf": rtf,
            "params": params,
            "accuracy": acc,
            "cer": cer,
            "evaluation_status": status,
            "is_favorite": False,
            "user_tags": [],
            "user_notes": ""
        }

    def list_records(self, limit: int = 50, favorite_only: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM synthesis_history"
        args = []
        if favorite_only:
            query += " WHERE is_favorite = 1"
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, args).fetchall()
            results = []
            for r in rows:
                params = json.loads(r["params_json"]) if r["params_json"] else {}
                tags = [t.strip() for t in r["user_tags"].split(",") if t.strip()]
                results.append({
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "time_str": r["time_str"],
                    "engine": r["engine"],
                    "voice_id": r["voice_id"],
                    "voice_name": r["voice_name"],
                    "text": r["text"],
                    "audio_url": r["audio_url"],
                    "duration": r["duration"],
                    "cost_sec": r["cost_sec"],
                    "rtf": r["rtf"],
                    "sample_rate": r["sample_rate"],
                    "params": params,
                    "accuracy": r["accuracy"],
                    "cer": r["cer"],
                    "evaluation_status": r["evaluation_status"],
                    "is_favorite": bool(r["is_favorite"]),
                    "user_tags": tags,
                    "user_notes": r["user_notes"] or ""
                })
            return results

    def update_tag_and_note(
        self,
        record_id: str,
        is_favorite: Optional[bool] = None,
        user_tags: Optional[List[str]] = None,
        user_notes: Optional[str] = None
    ) -> bool:
        updates = []
        args = []
        if is_favorite is not None:
            updates.append("is_favorite = ?")
            args.append(1 if is_favorite else 0)
        if user_tags is not None:
            updates.append("user_tags = ?")
            args.append(",".join([t.strip() for t in user_tags if t.strip()]))
        if user_notes is not None:
            updates.append("user_notes = ?")
            args.append(user_notes)

        if not updates:
            return False

        args.append(record_id)
        sql = f"UPDATE synthesis_history SET {', '.join(updates)} WHERE id = ?"
        with self._get_conn() as conn:
            cursor = conn.execute(sql, args)
            return cursor.rowcount > 0

    def delete_record(self, record_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM synthesis_history WHERE id = ?", (record_id,))
            return cursor.rowcount > 0

    def add_dialogue_turn(
        self,
        turn_id: str,
        user_text: str,
        assistant_reply: str,
        audio_url: str,
        engine: str,
        voice_id: str,
        duration: float,
        cost_sec: float,
        rtf: float,
        mode: str = "voice"
    ) -> Dict[str, Any]:
        now = time.time()
        time_str = time.strftime("%H:%M:%S", time.localtime(now))
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO call_dialogue_history (
                    id, created_at, time_str, user_text, assistant_reply,
                    audio_url, engine, voice_id, duration, cost_sec, rtf, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                turn_id, now, time_str, user_text, assistant_reply,
                audio_url, engine, voice_id, duration, cost_sec, rtf, mode
            ))
        return {
            "id": turn_id,
            "created_at": now,
            "time_str": time_str,
            "user_text": user_text,
            "assistant_reply": assistant_reply,
            "audio_url": audio_url,
            "engine": engine,
            "voice_id": voice_id,
            "duration": duration,
            "cost_sec": cost_sec,
            "rtf": rtf,
            "mode": mode
        }

    def list_dialogue_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM call_dialogue_history ORDER BY created_at ASC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_dialogue_history(self) -> bool:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM call_dialogue_history")
            return True

history_manager = HistoryManager()
