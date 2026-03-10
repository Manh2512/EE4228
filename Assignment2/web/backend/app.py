"""
app.py — FastAPI web server for the EE4228 face recognition demo.

Run from Assignment2/ root:
    uvicorn web.backend.app:app --host 0.0.0.0 --port 5000 --reload

Interactive docs: http://localhost:5000/docs
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .pipeline_bridge import bgr_to_base64_jpeg, get_pipeline, process_frame

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="EE4228 Face Recognition Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  {job_id: {"status": str, "progress": int}}
_jobs: dict[str, dict] = {}

_TMP = Path("/tmp/ee4228")
_TMP.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Startup: warm up the pipeline
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    await asyncio.to_thread(get_pipeline)


# ---------------------------------------------------------------------------
# Image endpoint
# ---------------------------------------------------------------------------

@app.post("/api/process/image")
async def api_process_image(image: UploadFile = File(...)):
    data = await image.read()
    buf  = np.frombuffer(data, np.uint8)
    bgr  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image file")

    annotated, faces = await asyncio.to_thread(process_frame, bgr)
    b64 = bgr_to_base64_jpeg(annotated)

    return {"processed_image": b64, "faces": faces, "face_count": len(faces)}


# ---------------------------------------------------------------------------
# Video endpoints
# ---------------------------------------------------------------------------

@app.post("/api/process/video")
async def api_process_video(video: UploadFile = File(...)):
    job_id   = str(uuid.uuid4())[:8]
    raw_path = _TMP / f"{job_id}_raw.mp4"

    with open(raw_path, "wb") as f:
        f.write(await video.read())

    _jobs[job_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(_video_task(job_id, str(raw_path)))
    return {"job_id": job_id}


@app.get("/api/process/video/status/{job_id}")
async def api_video_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **_jobs[job_id]}


@app.get("/api/process/video/result/{job_id}")
async def api_video_result(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    out_path = _TMP / f"{job_id}_processed.mp4"
    return FileResponse(str(out_path), media_type="video/mp4", filename="processed.mp4")


async def _video_task(job_id: str, raw_path: str):
    _jobs[job_id]["status"] = "processing"
    out_path = str(_TMP / f"{job_id}_processed.mp4")
    try:
        await asyncio.to_thread(_process_video_sync, job_id, raw_path, out_path)
        _jobs[job_id] = {"status": "done", "progress": 100}
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "progress": 0, "detail": str(exc)}


def _process_video_sync(job_id: str, raw_path: str, out_path: str):
    cap   = cv2.VideoCapture(raw_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _ = process_frame(frame)
        out.write(annotated)
        i += 1
        _jobs[job_id]["progress"] = int(i / total * 100)

    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# Webcam WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/webcam")
async def ws_webcam(websocket: WebSocket):
    await websocket.accept()
    paused = False

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ---- JSON control message (text) ----
            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                    action = ctrl.get("action", "")
                    if action == "pause":
                        paused = True
                    elif action == "resume":
                        paused = False
                except json.JSONDecodeError:
                    pass
                continue

            # ---- Binary JPEG frame ----
            if paused or "bytes" not in message:
                continue

            raw_bytes = message["bytes"]
            buf   = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            annotated, faces = await asyncio.to_thread(process_frame, frame)
            b64 = bgr_to_base64_jpeg(annotated)

            await websocket.send_text(json.dumps({
                "annotated_frame": b64,
                "faces":           faces,
                "face_count":      len(faces),
            }))

    except WebSocketDisconnect:
        pass
