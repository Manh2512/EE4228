Deliverables                                                                             
                                                                       | File | Purpose |
                                                                       |-----|----------|
                                                                       | Purpose | Full architecture doc (all 5 required sections) |
                                                                       | modules/detector.py | YOLOv7-Face wrapper — PyTorch .pt + ONNX .onnx support -> inline letterbox/NMS utilities |
                                                                       | modules/aligner.py | 5-point similarity-transform alignment → 112×112 ArcFace -> crop |    
                                                                       | modules/matcher.py | Cosine (default) + Euclidean matching with configurable |
                                                                       | modules/recognizer.py | ArcFace ONNX inference, [-1,1] normalization, L2-normalized -> 512-D output |
                                                                       | build_database.py | Scans data/faces/<Name>/, averages embeddings per identity -> saves database/embeddings.npz |

  To run after downloading model weights:
  # Set-up
  1. Input images to `/data/faces` folder.
  2. Download feature extraction model for face recognition [ArcFace](https://drive.google.com/file/d/1Hc5zUfBATaXUgcU2haUNa7dcaZSw95h2/view?usp=sharing). Place the checkpoint in `models/`.
  3. Download yolov7-face [checkpoint](https://drive.google.com/file/d/1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo/view?usp=sharing). Place the checkpoint in `models/`.
  4. Create Conda environment and install dependencies.
  5. Build gallery database.
  6. Run inference on an image.

  # 1. Build gallery database
  python build_database.py \
    --detector-weights models/yolov7-face.pt \
    --detector-weights models/yolov7-face.pt \
    --yolov7-dir models/yolov7-face \
    --recognizer-weights models/arcface_r100.onnx

  # 2. Run inference on an image
  python pipeline.py --mode image --input photo.jpg \
    -- detection-mode pytorch \
    --detector-weights models/yolov7-face.pt \
    --yolov7-dir models/yolov7-face \
    --recognizer-weights models/arcface_r100.onnx