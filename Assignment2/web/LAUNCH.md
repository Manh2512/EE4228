# Web Demo — Launch Guide

## Prerequisites

- Python environment with all AI dependencies installed (see `AI/LAUNCH.md`)
- Node.js ≥ 18
- Model weights and database already built:
  - `AI/models/yolov7-face/`
  - `AI/models/yolov7-face.pt`
  - `AI/models/arcface_r100.onnx`
  - `AI/database/embeddings.npz`

---

## 1. Start the Backend

Run from the **`Assignment2/`** root directory:

```bash
uvicorn web.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

> The pipeline (detector + recognizer + gallery) loads on startup. Wait for `[bridge] Pipeline ready.` before sending requests.

---

## 2. Start the Frontend

```bash
cd web/frontend
npm install       # first time only
npm run dev
```

The app is served at `http://localhost:5173`.
Vite proxies `/api` → `http://localhost:8000` and `/ws` → `ws://localhost:8000` automatically.

---

## Pages

| Route | Description |
|-------|-------------|
| `/` | Main page — choose input mode |
| `/image` | Upload a still image for detection & recognition |
| `/video` | Upload a video file; processed asynchronously |
| `/webcam` | Live webcam stream over WebSocket |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/process/image` | Process a single image |
| `POST` | `/api/process/video` | Submit a video job, returns `job_id` |
| `GET` | `/api/process/video/status/{job_id}` | Poll job progress |
| `GET` | `/api/process/video/result/{job_id}` | Download processed video |
| `WS` | `/ws/webcam` | WebSocket for real-time webcam frames |

---

## Stopping

- Backend: `Ctrl+C` in the uvicorn terminal
- Frontend: `Ctrl+C` in the Vite terminal
