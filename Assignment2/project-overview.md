# Project Overview — EE4228 GUI Web Demo

A browser-based GUI for the face detection and recognition pipeline. Users can run inference on static images, uploaded videos, or a live webcam feed — all from a single-page web app.

---

## Application Flow

```
Main Page
  ├── Image Page   — upload image → side-by-side raw / processed view
  ├── Video Page   — upload video → side-by-side raw / processed view
  └── Webcam Page  — live feed   → real-time annotated stream
```

---

## Pages & UI Components

### Main Page (`/`)

Three clickable mode cards on a dark background:

| Card | Icon | Description shown |
|---|---|---|
| Image | Camera | Detect and recognize faces within an image |
| Video | Film strip | Detect and recognize faces within a video |
| Webcam | Webcam | Real-time detect and recognize faces with webcam |

Each card navigates to its respective page.

---

### Image Page (`/image`)

| Element | Behaviour |
|---|---|
| `← Back` link | Returns to Main Page |
| `Upload an Image` button | Opens file picker (JPEG, PNG) |
| **Raw Image** panel | Displays the original uploaded image |
| **Processed Image** panel | Displays the annotated result returned by the backend |

Flow: user uploads → raw image shown immediately → POST to `/api/process/image` → processed image replaces placeholder.

---

### Video Page (`/video`)

| Element | Behaviour |
|---|---|
| `← Back` link | Returns to Main Page |
| `Upload a Video` button | Opens file picker (MP4, AVI, MOV) |
| **Raw Video** panel | Plays the original uploaded video |
| **Processed Video** panel | Plays the annotated result once processing is complete |

Flow: user uploads → raw video shown immediately → POST to `/api/process/video` → poll `/api/process/video/status/<id>` → fetch and play result when done.

---

### Webcam Page (`/webcam`)

| Element | Behaviour |
|---|---|
| `← Back` link | Returns to Main Page |
| **Webcam** panel | Full-width live feed with face annotations overlaid |
| `Pause/Continue` button | Freezes or resumes the processed stream |

Flow: page opens → WebSocket `/ws/webcam` established → browser captures frames → sends each as JPEG bytes → renders annotated frames returned by server.

---

## Recommended Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Frontend | React + Vite (or plain HTML/JS) | Dark theme, minimal UI |
| Backend | Python Flask or FastAPI | Wraps the AI pipeline |
| WebSocket | `flask-sock` or FastAPI WebSocket | Webcam real-time stream |
| Styling | Tailwind CSS or plain CSS | Dark background `#1a1a1a`, card borders |
| Video async | Celery + Redis (or in-process threading) | Offloads video processing |

---

## Suggested File Structure

```
web/
├── backend/
│   ├── app.py              # Flask/FastAPI server, route definitions
│   ├── routes/
│   │   ├── image.py        # POST /api/process/image
│   │   ├── video.py        # POST /api/process/video + status/result
│   │   └── webcam.py       # WebSocket /ws/webcam
│   └── pipeline_bridge.py  # Thin wrapper calling AI/pipeline.py
│
└── frontend/
    ├── index.html           # Main page — mode selection cards
    ├── image.html           # Image page
    ├── video.html           # Video page
    ├── webcam.html          # Webcam page
    ├── styles/
    │   └── main.css
    └── scripts/
        ├── image.js         # Upload → POST → display result
        ├── video.js         # Upload → POST → poll → play result
        └── webcam.js        # WebSocket capture-and-render loop
```

---

## Design Tokens

| Property | Value |
|---|---|
| Background | `#1a1a1a` |
| Card background | `#2a2a2a` |
| Card border | `#444` |
| Primary text | `#ffffff` |
| Secondary text | `#aaaaaa` |
| Button background | `#333` |
| Font | System sans-serif |
