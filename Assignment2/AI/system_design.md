# System Overview

## 1. Architecture

### 1.1 Module Breakdown

| Module | File | Responsibility |
|---|---|---|
| **Detector** | `modules/detector.py` | Detect faces with YOLOv7-Face; return bounding boxes + 5 landmarks |
| **Aligner** | `modules/aligner.py` | Crop and warp each face to a canonical 112×112 ArcFace-compatible crop |
| **Recognizer** | `modules/recognizer.py` | Extract L2-normalized 512-D ArcFace embedding per aligned face |
| **Matcher** | `modules/matcher.py` | Compare query embedding against database using ArcFace angular similarity; return identity + score |
| **DB Builder** | `build_database.py` | Pre-compute and cache embeddings for all gallery identities |
| **Pipeline** | `pipeline.py` | Orchestrate all modules end-to-end; supports image and video stream |

---

### 1.2 Data Flow

```
BGR Image (H×W×3, uint8)
  │
  ▼ [Detector]
List of {bbox:[x1,y1,x2,y2], conf:float, landmarks:[[x,y]×5]}
  │
  ▼ [Aligner]  — per detection
Aligned face (112×112×3, uint8, BGR)
  │
  ▼ [Recognizer]
512-D float32 embedding  (L2-normalized, ‖e‖=1)
  │
  ▼ [Matcher]
(identity: str, score: float, result: "MATCH"|"NO MATCH")
```

---

### 1.3 Pipeline Sequence

1. Load model weights (detector, recognizer) on startup.
2. Load embedding database from `database/embeddings.npz`.
3. For each input frame:
   a. Run YOLOv7-Face → list of face detections.
   b. For each detection, run the aligner → 112×112 crop.
   c. Run ArcFace extractor → 512-D L2-normalized embedding.
   d. Compare embedding against every gallery entry using ArcFace angular similarity (cos θ).
   e. If `max_score ≥ threshold` → report identity; else → "Unknown".
4. Draw bounding boxes, landmarks, and identity labels; output annotated frame.

---

## 2. Implementation Plan

### 2.1 Dependencies

Install:
```bash
pip install -r requirements.txt
```

---

### 2.2 Folder Structure

```
Assignment2/
├── pipeline.py               # End-to-end inference entry point
├── build_database.py         # Gallery database builder
├── requirements.txt
├── system_design.md
│
├── modules/
│   ├── __init__.py
│   ├── detector.py           # YOLOv7-Face wrapper
│   ├── aligner.py            # 5-point face alignment
│   ├── recognizer.py         # ArcFace ONNX wrapper
│   └── matcher.py            # ArcFace angular similarity matching
│
├── models/                   # Downloaded weight files (not committed)
│   ├── yolov7-face.pt        # YOLOv7-Face PyTorch weights
│   └── arcface_r100.onnx     # ArcFace ONNX weights (buffalo_l)
│
├── data/
│   └── faces/
│       └── <PersonName>/     # One sub-folder per identity
│           └── *.jpg
│
├── database/
│   └── embeddings.npz        # Pre-computed gallery embeddings
│
├── scripts/
│   └── run_tests.sh
│
└── tests/
    ├── test_aligner.py
    ├── test_matcher.py
    ├── test_recognizer.py
    ├── test_detector.py
    └── test_pipeline.py
```

---

### 2.3 Detection Module

- **Model**: YOLOv7-Face (`yolov7-face.pt`)
  - Supports two loading modes: PyTorch (`.pt`, requires `yolov7-face` repo on `sys.path`) and ONNX (`.onnx`, standalone).
- **Preprocessing** (`letterbox`):
  - Resize image with padding to `640×640` keeping aspect ratio.
  - Normalize pixels to `[0, 1]`, convert BGR→RGB, add batch dim.
- **Inference**:
  - `model(img)[0]` → raw predictions tensor `[batch, anchors, 16]`
  - 16 = 4 bbox (cx,cy,w,h) + 1 objectness + 1 class + 10 landmarks (5×xy)
- **Postprocessing** (`non_max_suppression_face`):
  - Multiply objectness × class confidence.
  - Convert `(cx,cy,w,h)` → `(x1,y1,x2,y2)`.
  - Apply `torchvision.ops.nms` with configurable `iou_threshold`.
  - Scale coordinates back to original image size.
- **Output**: `List[dict]` — `{'bbox', 'confidence', 'landmarks'}`

---

### 2.4 Alignment Module

Reference landmarks (112×112, ArcFace standard):

```
Left eye:         [38.29, 51.70]
Right eye:        [73.53, 51.50]
Nose tip:         [56.03, 71.74]
Left mouth:       [41.55, 92.37]
Right mouth:      [70.73, 92.20]
```

Steps:
1. Receive 5 detected landmarks `src` (5×2 float array).
2. Compute `skimage.transform.SimilarityTransform` mapping `src → dst`.
3. Extract 2×3 affine matrix `M = tform.params[:2]`.
4. Apply `cv2.warpAffine(image, M, (112, 112))`.
5. Return aligned BGR crop (uint8, 112×112×3).

---

### 2.5 Recognition Module

- **Model**: ArcFace ONNX (`arcface_r100.onnx`, InsightFace buffalo_l `w600k_r50.onnx`).
- **Input**: 112×112 BGR face crop.
- **Preprocessing**:
  - Convert BGR → RGB.
  - Normalize: `(pixel - 127.5) / 127.5` → range `[-1, 1]`.
  - Reshape: HWC → CHW, add batch dim → `[1, 3, 112, 112]` float32.
- **Inference**: `onnxruntime.InferenceSession.run(...)` → `[1, 512]`.
- **Postprocessing**: L2-normalize: `e = e / ‖e‖₂`.
- **Output**: `np.ndarray` shape `(512,)`, L2-normalized.

---

### 2.6 Matching Module

**Metric: ArcFace Additive Angular Margin Similarity** (Deng et al., CVPR 2019)

```python
# Both embeddings are L2-normalized (‖e‖₂ = 1).
# The dot product equals the cosine of the angle between them:
cos_theta = np.dot(query_emb, db_emb)           # score ∈ [-1, 1]
theta_deg = np.degrees(np.arccos(cos_theta))     # geodesic angle in degrees
```

- **Decision rule**: `cos(θ) ≥ threshold` → MATCH.
- Default threshold: `0.40` (θ ≤ 66°) — appropriate operating point for ArcFace R100.
  - Raise toward `1.0` (smaller angle) for higher precision.
  - Lower toward `0.0` (larger angle) for higher recall.
- Returns `MatchResult(identity, score=cos(θ), angle_deg=θ, "MATCH" | "NO MATCH")`.
- If no gallery entry passes threshold → `("Unknown", max_score, "NO MATCH")`.

---

## 3. Embedding Database Design

- **Storage format**: NumPy `.npz` file at `database/embeddings.npz`.
- **Schema**:
  ```python
  {
    "names":      np.array([str, ...]),       # shape (N,)
    "embeddings": np.ndarray,                 # shape (N, 512), float32, L2-normalized
  }
  ```
- **Building**: `build_database.py` scans `data/faces/<Name>/<*.jpg>`,
  aligns each image, extracts its embedding, and averages all embeddings per
  identity before L2-renormalizing. This improves robustness to pose variation.
- **Loading**: A single `np.load("database/embeddings.npz")` call at runtime.
- **Adding a new identity**: Re-run `build_database.py`; it rebuilds the full
  `.npz`. No schema migration required.

---

## 4. Matching Strategy

### ArcFace Angular Margin Similarity (only metric)

Based on the additive angular margin loss from **ArcFace** (Deng et al., CVPR 2019).  The training loss penalizes pairs by adding a margin `m` to the angle between the embedding and its class centre on the unit hypersphere, forcing the inter-class angles to be larger and the intra-class angles to be smaller.  At inference, this tighter angular clustering means a simple cosine threshold in angle space is highly discriminative.

- **Formula**: `score = cos(θ) = e_q · e_db` (both embeddings L2-normalized, so dot product = cosine of angle).
- **Angle**: `θ = arccos(score)` — geodesic distance on the unit hypersphere.
- **Range**: `score ∈ [-1, 1]`, `θ ∈ [0°, 180°]`; higher score / smaller angle = more similar.
- **Threshold** `τ = 0.40` (θ ≤ 66°): default operating point for ArcFace R100.
  - Raise (e.g. `0.60`) for higher precision, fewer false positives.
  - Lower (e.g. `0.20`) for higher recall, fewer false negatives.
- **Vectorized** over gallery: `scores = embeddings_matrix @ query_emb`.
- **Best match**: `idx = argmax(scores)`.

### Unknown Face Handling

- If `max(scores) < threshold` → identity = `"Unknown"`.
- Reported score is still the best `cos(θ)` found (along with the corresponding angle in degrees).

### Multi-Face Handling

- Detector returns all faces independently.
- Each face goes through the full Align → Embed → Match pipeline.
- Results are a list; one entry per detected face, in detection-confidence order.

---

## 5. Failure Cases & Edge Handling

| Failure | Cause | Handling |
|---|---|---|
| No face detected | Low resolution, extreme pose, occlusion | Return empty list; skip matching |
| Detection below conf threshold | Low-quality frame | Tunable `conf_thres` (default 0.25) |
| Alignment fails (degenerate landmarks) | Detector returns near-collinear points | Catch `cv2.error`; skip face |
| Embedding is all-zero | Blank/solid-color crop | `norm + ε` in L2 normalization prevents divide-by-zero |
| Empty gallery | Database not built | Raise `RuntimeError` with setup instructions |
| GPU OOM | Large batch or resolution | Fall back to CPU automatically via `torch.cuda.is_available()` |
| Multiple images per identity | Same person, different photos | Average embeddings per identity in `build_database.py`, then re-normalize |
| Near-duplicate identities in gallery | Very similar people | Lower threshold may cause mis-identification; report top-2 scores |
