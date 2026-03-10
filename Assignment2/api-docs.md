# API Documentation — EE4228 GUI Web Demo

All endpoints are served by the backend web server wrapping the AI pipeline.
Base URL: `http://localhost:5000`

---

## Image Processing

### `POST /api/process/image`

Upload an image and receive an annotated version with detected and recognized faces.

**Request**

```
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | yes | Image file (JPEG, PNG) |
| `match_threshold` | float | no | Cosine similarity threshold (default `0.40`) |

**Response `200 OK`**

```json
{
  "processed_image": "<base64-encoded JPEG>",
  "faces": [
    {
      "identity":   "Alice",
      "score":      0.72,
      "angle_deg":  43.9,
      "result":     "MATCH",
      "bbox":       [120, 45, 280, 230]
    }
  ],
  "face_count": 1
}
```

**Response `400 Bad Request`**

```json
{ "error": "No image file provided" }
```

---

## Video Processing

### `POST /api/process/video`

Upload a video file and receive an annotated version. Processing is asynchronous.

**Request**

```
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | yes | Video file (MP4, AVI, MOV) |
| `match_threshold` | float | no | Cosine similarity threshold (default `0.40`) |

**Response `202 Accepted`**

```json
{ "job_id": "f3a2c1d8" }
```

### `GET /api/process/video/status/<job_id>`

Poll processing progress.

**Response `200 OK`**

```json
{
  "job_id":   "f3a2c1d8",
  "status":   "processing",   // "queued" | "processing" | "done" | "error"
  "progress": 64              // percentage 0–100
}
```

### `GET /api/process/video/result/<job_id>`

Retrieve the processed video once `status` is `"done"`.

**Response `200 OK`**

```
Content-Type: video/mp4
Body: binary video stream
```

---

## Webcam Stream

### `WebSocket /ws/webcam`

Bidirectional WebSocket for real-time webcam processing.

**Client → Server (binary)**

Raw video frame as JPEG bytes, sent on each capture tick.

**Server → Client (JSON)**

```json
{
  "annotated_frame": "<base64-encoded JPEG>",
  "faces": [
    {
      "identity":   "Bob",
      "score":      0.65,
      "angle_deg":  49.4,
      "result":     "MATCH",
      "bbox":       [50, 30, 200, 210]
    }
  ],
  "face_count": 1
}
```

**Client control messages (JSON)**

```json
{ "action": "pause" }
{ "action": "resume" }
```

---

## Response Field Reference

| Field | Type | Description |
|---|---|---|
| `identity` | string | Recognized name, or `"Unknown"` |
| `score` | float | Cosine similarity ∈ [−1, 1] |
| `angle_deg` | float | Geodesic angle in degrees ∈ [0°, 180°] |
| `result` | string | `"MATCH"` or `"NO MATCH"` |
| `bbox` | int[4] | `[x1, y1, x2, y2]` in original image pixels |
| `face_count` | int | Total faces detected in the frame |

---

## HTTP Status Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `202` | Accepted (async job started) |
| `400` | Bad request (missing or invalid input) |
| `404` | Job ID not found |
| `500` | Internal server error |
