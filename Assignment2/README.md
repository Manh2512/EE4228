# Online Face Detection and Recognition System Based on Deep Neural Network

A face detection and recognition system with a browser-based demo supporting static images, uploaded videos, and real-time webcam streaming.

---

## Architecture Overview

```
Input Frame (BGR)
    │
    ├─→ Detector (YOLOv7-Face)          → bounding boxes + 5 landmarks
    │
    ├─→ Aligner (5-point affine)        → 112×112 aligned crop
    │
    ├─→ Recognizer (ArcFace ResNet-100) → 512-D L2-normalized embedding
    │
    └─→ Matcher (cosine / euclidean)    → identity + similarity score
```

### Models

| Model | File | Size | Purpose |
|---|---|---|---|
| YOLOv7-Face | `AI/models/yolov7-face.pt` | 73.6 MB | Face detection + landmark regression |
| YOLOv7-Face Tiny | `AI/models/yolov7-tiny-face.pt` | 12.4 MB | Lightweight detector variant |
| ArcFace ResNet-100 | `AI/models/arcface_r100.onnx` | 260.7 MB | Face embedding extraction |

### Gallery Database

Pre-computed embeddings for 7 identities (`0001`–`0007`) are stored in `AI/database/embeddings.npz`.
Each identity embedding is the L2-renormalized average of all per-image embeddings.

---

## Project Structure

```
Assignment2/
├── AI/                          # Core AI pipeline
│   ├── pipeline.py              # FaceRecognitionPipeline orchestrator
│   ├── build_database.py        # Build gallery embeddings from data/faces/
│   ├── modules/
│   │   ├── detector.py          # YOLOv7-Face wrapper
│   │   ├── aligner.py           # 5-point similarity transform → 112×112 crop
│   │   ├── recognizer.py        # ArcFace ONNX inference
│   │   └── matcher.py           # Cosine / Euclidean similarity matching
│   ├── models/                  # Model weight files
│   ├── database/
│   │   └── embeddings.npz       # Pre-computed gallery
│   ├── data/
│   │   └── faces/               # Gallery images: <PersonID>/<image>.jpg
│   ├── tests/                   # Unit tests
│   └── requirements.txt
│
└── web/                         # Web demo
    ├── backend/
    │   ├── app.py               # FastAPI server
    │   ├── pipeline_bridge.py   # Pipeline singleton + frame encoding
    │   └── requirements.txt
    └── frontend/                # React 18 + TypeScript + Vite SPA
        ├── src/
        │   ├── App.tsx
        │   └── pages/
        │       ├── MainPage.tsx
        │       ├── ImagePage.tsx
        │       ├── VideoPage.tsx
        │       └── WebcamPage.tsx
        ├── vite.config.ts       # Proxies /api and /ws → localhost:8000
        └── package.json
```

---

## Setup

### 1. Install AI and backend dependencies

From the `Assignment2/` root:

```bash
pip install -r AI/requirements.txt
pip install -r web/backend/requirements.txt
```

### 2. (Optional) Rebuild the gallery database

Only needed if you change the images in `AI/data/faces/`:

```bash
cd AI
python build_database.py
```

### 3. Start the backend

From the `Assignment2/` root:

```bash
uvicorn web.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start the frontend

```bash
cd web/frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Usage

The landing page offers three modes:

| Mode | Description |
|---|---|
| **Image** | Upload a JPEG/PNG; receive an annotated image with bounding boxes and identity labels |
| **Video** | Upload an MP4; processing runs asynchronously; download the annotated result when done |
| **Webcam** | Real-time WebSocket streaming; annotated frames are displayed live with pause/resume control |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/process/image` | Process a single image |
| `POST` | `/api/process/video` | Submit a video job; returns `job_id` |
| `GET` | `/api/process/video/status/{job_id}` | Poll job progress (`0`–`100`) |
| `GET` | `/api/process/video/result/{job_id}` | Download processed video |
| `WS` | `/ws/webcam` | Real-time frame exchange |

### Image response schema

```json
{
  "processed_image": "<base64 JPEG>",
  "face_count": 1,
  "faces": [
    {
      "identity": "0001",
      "score": 0.52,
      "metric": "cosine",
      "result": "MATCH",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## Matching Metrics

| Metric | Threshold | Match condition |
|---|---|---|
| Cosine similarity | 0.40 | `score >= threshold` |
| Euclidean distance | 1.10 | `distance <= threshold` |

Default: **cosine**. Change via `_METRIC` and `_THRESHOLD` in `web/backend/pipeline_bridge.py`.

---

## Running the AI Pipeline Standalone

```bash
cd AI

# Single image
python pipeline.py --mode image --input photo.jpg --output result.jpg

# Webcam (device 0)
python pipeline.py --mode video --input 0

# Folder of images
python pipeline.py --mode folder --input faces/ --output output_dir/
```

Optional flags: `--metric cosine|euclidean`, `--threshold <float>`, `--model yolov7-face.pt|yolov7-tiny-face.pt`
