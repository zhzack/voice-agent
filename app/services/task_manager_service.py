# app/services/task_manager_service.py
"""Task Manager Service – handles asynchronous tasks such as TTS synthesis,
LLM generation, and real‑time calls.

Each task is persisted in a SQLite ``tasks`` table (via SQLAlchemy) so that
the API can query its status, cancel it, or retrieve the result later.
"""

import uuid
import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import (Column, String, DateTime, JSON, Enum as SAEnum,
                        create_engine, select, update, delete)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------- SQLAlchemy setup ----------
Base = declarative_base()
engine = create_engine("sqlite:///voice_agent.db", echo=False, future=True)
Session = sessionmaker(bind=engine)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskORM(Base):
    __tablename__ = "tasks"
    task_id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)   # e.g. "chattts"
    voice_id = Column(String, nullable=True)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    parameters = Column(JSON, nullable=True)    # raw request parameters
    result = Column(JSON, nullable=True)        # e.g. {"audio_path": "...", "metrics": {...}}

# Ensure table exists
Base.metadata.create_all(engine)


class TaskManagerService:
    """Service layer that creates, updates and queries tasks.

    The service is deliberately lightweight – heavy‑weight work (actual TTS / LLM)
    should be executed in background coroutines or external processes.  The
    ``TaskManagerService`` only records the lifecycle.
    """

    def __init__(self):
        self.session = Session()
        # Simple in‑memory lock to guard creation (optional for single‑process)
        self._lock = asyncio.Lock()

    # ---------- Creation ----------
    async def create_task(self, model_id: str, voice_id: Optional[str] = None,
                          parameters: Optional[Dict] = None) -> str:
        """Create a new task record and return its UUID string."""
        async with self._lock:
            task_id = uuid.uuid4().hex
            task = TaskORM(
                task_id=task_id,
                model_id=model_id,
                voice_id=voice_id,
                status=TaskStatus.PENDING,
                parameters=parameters or {},
            )
            self.session.add(task)
            self.session.commit()
            return task_id

    # ---------- Status updates ----------
    def set_status(self, task_id: str, status: TaskStatus, result: Optional[Dict] = None) -> None:
        stmt = (
            update(TaskORM)
            .where(TaskORM.task_id == task_id)
            .values(status=status, result=result)
        )
        self.session.execute(stmt)
        self.session.commit()

    # ---------- Query ----------
    def get_task(self, task_id: str) -> Dict[str, Any]:
        task = self.session.get(TaskORM, task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        return {
            "task_id": task.task_id,
            "model_id": task.model_id,
            "voice_id": task.voice_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "parameters": task.parameters,
            "result": task.result,
        }

    def list_tasks(self, status: Optional[TaskStatus] = None) -> list[Dict[str, Any]]:
        stmt = select(TaskORM)
        if status:
            stmt = stmt.where(TaskORM.status == status)
        rows = self.session.execute(stmt).scalars().all()
        return [
            {
                "task_id": r.task_id,
                "model_id": r.model_id,
                "voice_id": r.voice_id,
                "status": r.status.value,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
                "parameters": r.parameters,
                "result": r.result,
            }
            for r in rows
        ]

    # ---------- Cancellation ----------
    def cancel_task(self, task_id: str) -> None:
        # In a real system we would signal the running worker; here we just mark.
        self.set_status(task_id, TaskStatus.CANCELLED)

    # ---------- Helper for background workers ----------
    async def run_with_task(self, model_id: str, voice_id: Optional[str],
                            parameters: Dict, coro):
        """Utility wrapper used by background workers.

        1. Create a task record (PENDING).
        2. Mark RUNNING, await the coroutine, capture result.
        3. On success set SUCCESS + result, on exception set FAILED.
        """
        task_id = await self.create_task(model_id, voice_id, parameters)
        self.set_status(task_id, TaskStatus.RUNNING)
        try:
            result = await coro
            self.set_status(task_id, TaskStatus.SUCCESS, result=result)
        except Exception as exc:
            self.set_status(task_id, TaskStatus.FAILED, result={"error": str(exc)})
        return task_id

# End of file
