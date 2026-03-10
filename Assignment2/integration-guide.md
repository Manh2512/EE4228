# Integration Guide — EE4228 GUI Web Demo

This guide explains how to connect the web frontend to the AI pipeline backend using **FastAPI**.

---

## Prerequisites

- AI pipeline already set up and functional (models downloaded, database built)
- Python 3.8+ environment with `AI/requirements.txt` installed
- Node.js 18+ (if using a JS build tool like Vite)

---

## 1. Backend Setup

### Install web dependencies

```bash
pip install fastapi uvicorn python-multipart websockets
```

### Bridge the AI pipeline

Create `web/backend/pipeline_bridge.py` to load the pipeline once at startup:

```python
import sys
sys.path.insert(0, "AI")

from pipeline import FacePipeline

# Initialised once on startup — reused across all requests
pipeline = FacePipeline(
    detector_weights="AI/models/yolov7-face.pt",
    yolov7_dir="AI/models/yolov7-face",
    recognizer_weights="AI/models/arcface_r100.onnx",
    db_path="AI/database/embeddings.npz",
    match_threshold=0.40,
)

def process_image(bgr_image):
    """Returns (annotated_bgr, list_of_results)."""
    return pipeline.run_image(bgr_image)

def process_frame(bgr_frame):
    """Returns (annotated_bgr, list_of_results) for a single frame."""
    return pipeline.run_image(bgr_frame)
```

---

## 2. Backend Routes (FastAPI)

### `web/backend/app.py`

```python
import cv2, numpy as np, base64, json, asyncio, uuid
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pipeline_bridge import process_image, process_frame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (replace with Redis/Celery for production)
jobs = {}
```

### Image endpoint

```python
@app.post("/api/process/image")
async def api_process_image(image: UploadFile = File(...)):
    buf = np.frombuffer(await image.read(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image file")

    annotated, faces = process_image(bgr)

    _, jpeg = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(jpeg).decode()

    return {"processed_image": b64, "faces": faces, "face_count": len(faces)}
```

### Video endpoints

```python
@app.post("/api/process/video")
async def api_process_video_upload(video: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0}

    # Save upload
    raw_path = f"/tmp/{job_id}_raw.mp4"
    with open(raw_path, "wb") as f:
        f.write(await video.read())

    asyncio.create_task(_process_video_task(job_id, raw_path))
    return {"job_id": job_id}


@app.get("/api/process/video/status/{job_id}")
async def api_video_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **jobs[job_id]}


@app.get("/api/process/video/result/{job_id}")
async def api_video_result(job_id: str):
    if jobs.get(job_id, {}).get("status") != "done":
        raise HTTPException(404, "Result not ready")
    return FileResponse(f"/tmp/{job_id}_processed.mp4", media_type="video/mp4")


async def _process_video_task(job_id: str, raw_path: str):
    jobs[job_id]["status"] = "processing"
    out_path = f"/tmp/{job_id}_processed.mp4"

    cap = cv2.VideoCapture(raw_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _ = process_frame(frame)
        out.write(annotated)
        jobs[job_id]["progress"] = int((i + 1) / total * 100)
        await asyncio.sleep(0)   # yield to event loop

    cap.release()
    out.release()
    jobs[job_id] = {"status": "done", "progress": 100}
```

### Webcam WebSocket endpoint

```python
@app.websocket("/ws/webcam")
async def ws_webcam(websocket: WebSocket):
    await websocket.accept()
    paused = False

    while True:
        data = await websocket.receive()

        # Control message (JSON text)
        if "text" in data:
            msg = json.loads(data["text"])
            paused = (msg.get("action") == "pause")
            continue

        if paused:
            continue

        # Binary frame (JPEG bytes)
        buf   = np.frombuffer(data["bytes"], np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        annotated, faces = process_frame(frame)
        _, jpeg = cv2.imencode(".jpg", annotated)
        b64     = base64.b64encode(jpeg).decode()

        await websocket.send_text(json.dumps({
            "annotated_frame": b64,
            "faces":           faces,
            "face_count":      len(faces),
        }))
```

### Start the server

```bash
uvicorn web.backend.app:app --host 0.0.0.0 --port 5000 --reload
# Server runs at http://localhost:5000
# Interactive API docs at http://localhost:5000/docs
```

---

## 3. Frontend Integration

### Image page (`image.js`)

```javascript
document.getElementById("upload-btn").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Show raw image immediately
  document.getElementById("raw-img").src = URL.createObjectURL(file);

  const form = new FormData();
  form.append("image", file);

  const res  = await fetch("http://localhost:5000/api/process/image",
                           { method: "POST", body: form });
  const data = await res.json();

  document.getElementById("processed-img").src =
    `data:image/jpeg;base64,${data.processed_image}`;
});
```

### Video page (`video.js`)

```javascript
async function uploadVideo(file) {
  document.getElementById("raw-video").src = URL.createObjectURL(file);

  const form = new FormData();
  form.append("video", file);

  const res        = await fetch("http://localhost:5000/api/process/video",
                                 { method: "POST", body: form });
  const { job_id } = await res.json();

  await pollUntilDone(job_id);
}

async function pollUntilDone(jobId) {
  while (true) {
    const res    = await fetch(
      `http://localhost:5000/api/process/video/status/${jobId}`);
    const status = await res.json();

    if (status.status === "done") {
      document.getElementById("processed-video").src =
        `http://localhost:5000/api/process/video/result/${jobId}`;
      break;
    }
    if (status.status === "error") break;

    await new Promise(r => setTimeout(r, 1000));
  }
}
```

### Webcam page (`webcam.js`)

```javascript
const ws     = new WebSocket("ws://localhost:5000/ws/webcam");
const video  = document.getElementById("webcam-feed");
const canvas = document.createElement("canvas");
let   paused = false;

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
  video.play();
  requestAnimationFrame(captureLoop);
});

function captureLoop() {
  if (!paused) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(blob => blob && ws.send(blob), "image/jpeg", 0.8);
  }
  requestAnimationFrame(captureLoop);
}

ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  document.getElementById("processed-feed").src =
    `data:image/jpeg;base64,${data.annotated_frame}`;
};

document.getElementById("pause-btn").addEventListener("click", () => {
  paused = !paused;
  ws.send(JSON.stringify({ action: paused ? "pause" : "resume" }));
  document.getElementById("pause-btn").textContent =
    paused ? "Continue" : "Pause/Continue";
});
```

---

## 4. Quick-Start Checklist

- [ ] AI models downloaded, gallery database built
- [ ] Web dependencies installed: `pip install fastapi uvicorn python-multipart websockets`
- [ ] Backend running: `uvicorn web.backend.app:app --port 5000 --reload`
- [ ] Frontend served (open `index.html` or `npm run dev`)
- [ ] Three mode cards visible on main page
- [ ] Image mode: upload photo → processed result displayed alongside
- [ ] Video mode: upload clip → processed video plays when ready
- [ ] Webcam mode: live feed shows annotated faces; Pause/Continue works
