"""The API contract (section 6 of the guide): /analyze, /ingest, /health, /trace/{run_id}."""

import json
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app import agent, config, db, llm
from app.ingest import ingest

app = FastAPI(title="Factory Energy Cost Agent")


class AnalyzeRequest(BaseModel):
    question: str
    period: str | None = None       # e.g. "2026-Q3"
    compare_to: str | None = None   # e.g. "2026-Q2"


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return agent.answer(req.question, req.period, req.compare_to)


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), meta: str = Form(...)):
    try:
        table = json.loads(meta)["table"]
    except (ValueError, KeyError):
        raise HTTPException(400, 'meta must be JSON like {"table": "meter_15min"}')
    if table not in config.REQUIRED:
        raise HTTPException(400, f"table must be one of {list(config.REQUIRED)}")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
    return ingest(tmp.name, table, file.filename)


@app.get("/health")
def health():
    try:
        store = {"status": "ok", "data_version": db.data_version(), "meter_data": config.SLOTS_PATH.exists()}
    except Exception as e:  # noqa: BLE001 - health must answer even if the store is broken
        store = {"status": "error", "detail": str(e)}
    return {"api": "ok", "data_store": store, "llm": llm.status()}


@app.get("/trace/{run_id}")
def trace(run_id: str):
    t = agent.get_trace(run_id)
    if t is None:
        raise HTTPException(404, f"run {run_id} not found")
    return t
