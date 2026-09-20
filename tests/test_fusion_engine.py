from defense.localization.fusion_engine import FusionEngine

def test_soft_fusion_scoring():
    """Score should be in [0, 1] range."""
    engine = FusionEngine(mode="soft")
    candidate = (100, 100, 40, 30)
    yolo_dets = [{"center_x": 120, "center_y": 115, "width": 40, "height": 30, "confidence": 0.85, "polygon": None}]
    score = engine.score_candidate(candidate, yolo_dets, prev_centroids=None, image_shape=(720, 1280))
    assert 0.0 <= score <= 1.0

def test_yolo_mode_passthrough():
    """In 'yolo' mode, fusion should return YOLO detections unchanged."""
    engine = FusionEngine(mode="yolo")
    yolo_dets = [{"center_x": 100, "center_y": 100, "width": 20, "height": 20, "confidence": 0.9, "polygon": None}]
    candidates = [(200, 200, 30, 30)]
    fused = engine.fuse(candidates, yolo_dets)
    assert fused == yolo_dets

def test_seg_mode_passthrough():
    """In 'seg' mode, fusion should return segmentation candidates."""
    engine = FusionEngine(mode="seg")
    candidates = [(100, 100, 20, 20)]
    yolo_dets = [{"center_x": 300, "center_y": 300, "width": 30, "height": 30, "confidence": 0.9, "polygon": None}]
    fused = engine.fuse(candidates, yolo_dets)
    assert len(fused) == 1
    assert fused[0]["center_x"] == 110
    assert fused[0]["center_y"] == 110
