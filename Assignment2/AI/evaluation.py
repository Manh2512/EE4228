"""
Evaluate the face recognition model
python pipeline.py
    --input <folder_of_an_identity_not_include_database_images> \
    --detector-mode pytorch \
    --detector-weights models/yolov7-tiny-face.pt \
    --yolov7-dir models/yolov7-face \
    --recognizer-weights models/arcface_r100.onnx
"""
import argparse
import sys
from pathlib import Path
import cv2
from dataclasses import asdict
from pipeline import build_pipeline


def folder_inference(args):
    """
    Run face recognizer inference with a folder of an identity
    Return accuracy
    """
    folder = Path(args.input)
    if not folder.is_dir():
        sys.exit(f"Not a directory: {args.input}")
        
    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    )
    
    if not image_paths:
        sys.exit(f"No images found in: {args.input}")

    real_identity = folder.name
    score = 0 # number of images correctly classified
        
    pipeline = build_pipeline(args)

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue
        
        detections, results = pipeline.run(frame) # list of dicts, list of MatchResult
        faces = []
        
        for det, res in zip(detections, results):
            res_dict = asdict(res)
            faces.append(det | res_dict)
            
        # process largest face in image
        largest = max(
            faces,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )
        
        if largest["identity"] == real_identity:
            score += 1
    
    return real_identity, score/len(image_paths)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face Detection and Recognition Pipeline")
    p.add_argument("--input",                required=True, help="Path to input folder")
    p.add_argument("--detector-weights",     required=True, help=".pt or .onnx YOLOv7-Face weights")
    p.add_argument("--recognizer-weights",   required=True, help="ArcFace ONNX model path")
    p.add_argument("--database",             default="database/embeddings.npz")
    p.add_argument("--detector-mode",        default=None,  choices=["pytorch", "onnx"])
    p.add_argument("--yolov7-dir",           default=None,  help="yolov7-face repo root (pytorch mode only)")
    p.add_argument("--img-size",             type=int,   default=640)
    p.add_argument("--conf-thres",           type=float, default=0.25)
    p.add_argument("--iou-thres",            type=float, default=0.45)
    p.add_argument("--threshold",            type=float, default=None,
                                             help="Match decision threshold. Defaults: 0.40 for cosine (cos θ ≥ threshold), "
                                                  "1.10 for euclidean (L2 dist ≤ threshold).")
    p.add_argument("--metric",               default="cosine", choices=["cosine", "euclidean"],
                                             help="Similarity metric: 'cosine' (default) or 'euclidean'.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    identity, acc = folder_inference(args)
    print(f"Identity: {identity} | ACC: {(acc*100):.2f}%")