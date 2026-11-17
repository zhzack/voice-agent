import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_results")
REPORTS_DIR = os.path.join(BENCHMARK_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Import corpus from benchmark runner
try:
    from legacy.run_tts_benchmark import BENCHMARK_CORPUS, TTSBenchmarkRunner
except ImportError:
    from run_tts_benchmark import BENCHMARK_CORPUS, TTSBenchmarkRunner

router = APIRouter(prefix="/api/v1/benchmark", tags=["Benchmark"])

# Background state for monitoring
BENCHMARK_STATE = {
    "is_running": False,
    "last_run_time": None,
    "current_category": None,
    "latest_summary": None
}

class BenchmarkRunRequest(BaseModel):
    engine: str = "edge"
    voice: str = "yunxi"
    seed: int = 7869
    category: Optional[str] = "daily"  # None or specific category
    run_all: bool = False

@router.get("/status")
def get_benchmark_status():
    """获取当前或最近一次自动化评测的运行状态"""
    return BENCHMARK_STATE

@router.get("/categories")
def get_categories():
    result = {}
    for cat, cases in BENCHMARK_CORPUS.items():
        result[cat] = [
            {"id": c[0], "text": c[1]}
            for c in cases
        ]
    return result

def _execute_benchmark_sync(engine: str, voice: str, seed: int, category: Optional[str], run_all: bool):
    global BENCHMARK_STATE
    BENCHMARK_STATE["is_running"] = True
    BENCHMARK_STATE["last_run_time"] = time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        runner = TTSBenchmarkRunner(engine_type=engine, voice=voice, seed=seed)
        all_res = {}
        if run_all or category is None:
            for cat in BENCHMARK_CORPUS.keys():
                BENCHMARK_STATE["current_category"] = cat
                all_res[cat] = runner.run_category(cat)
        else:
            BENCHMARK_STATE["current_category"] = category
            all_res[category] = runner.run_category(category)

        report_path = runner.generate_report(all_res)
        filename = os.path.basename(report_path)

        # Calculate summary
        total_acc = 0.0
        total_rtf = 0.0
        count = 0
        for cat_items in all_res.values():
            for x in cat_items:
                total_acc += x["accuracy"]
                total_rtf += x["rtf"]
                count += 1

        summary = {
            "status": "success",
            "report_filename": filename,
            "report_url": f"/api/v1/benchmark/reports/{filename}",
            "total_cases": count,
            "avg_accuracy": round(total_acc / max(count, 1), 2),
            "avg_rtf": round(total_rtf / max(count, 1), 3),
            "details": all_res
        }
        BENCHMARK_STATE["latest_summary"] = summary
        return summary
    finally:
        BENCHMARK_STATE["is_running"] = False
        BENCHMARK_STATE["current_category"] = None

@router.post("/run")
async def trigger_benchmark(req: BenchmarkRunRequest):
    if req.category and req.category not in BENCHMARK_CORPUS and not req.run_all:
        raise HTTPException(status_code=400, detail=f"未知分类: {req.category}")

    if BENCHMARK_STATE["is_running"]:
        raise HTTPException(status_code=429, detail="已有正在执行的评测任务，请稍后刷新重试")

    # Run in threadpool so FastAPI event loop remains unblocked
    summary = await asyncio.to_thread(
        _execute_benchmark_sync,
        engine=req.engine,
        voice=req.voice,
        seed=req.seed,
        category=req.category,
        run_all=req.run_all
    )
    return summary

@router.get("/reports")
def list_reports():
    if not os.path.exists(REPORTS_DIR):
        return []
    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".md")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(REPORTS_DIR, x)), reverse=True)

    reports = []
    for f in files:
        f_path = os.path.join(REPORTS_DIR, f)
        reports.append({
            "filename": f,
            "created_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f_path))),
            "size_bytes": os.path.getsize(f_path)
        })
    return reports

@router.get("/reports/{filename}")
def read_report(filename: str):
    f_path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(f_path):
        raise HTTPException(status_code=404, detail="报告不存在")
    with open(f_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "markdown": content}
