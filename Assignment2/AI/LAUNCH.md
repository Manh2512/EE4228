Deliverables                                                                             
                                                                       | File | Purpose |
                                                                       |-----|----------|
                                                                       | Purpose | Full architecture doc (all 5 required sections) |
                                                                       | modules/detector.py | YOLOv7-Face wrapper — PyTorch .pt + ONNX .onnx support -> inline letterbox/NMS utilities |
                                                                       | modules/aligner.py | 5-point similarity-transform alignment → 112×112 ArcFace -> crop |    
                                                                       | modules/matcher.py | Cosine (default) + Euclidean matching with configurable |
                                                                       | modules/recognizer.py | ArcFace ONNX inference, [-1,1] normalization, L2-normalized -> 512-D output |
                                                                       | build_database.py | Scans data/faces/<Name>/, averages embeddings per identity -> saves database/embeddings.npz |
                                                                       | tests/ |  51 unit + integration tests (6 auto-skipped when model weights absent) |
                                                                       | scripts/run_tests.sh | Test runner with VERBOSE and SKIP_INTEGRATION flags |

  To run after downloading model weights:
  # 1. Build gallery database
  python build_database.py \
    --detector-weights models/yolov7-face.pt \
    --recognizer-weights models/arcface_r100.onnx

  # 2. Run inference on an image
  python pipeline.py --mode image --input photo.jpg \
    --detector-weights models/yolov7-face.pt \
    --recognizer-weights models/arcface_r100.onnx