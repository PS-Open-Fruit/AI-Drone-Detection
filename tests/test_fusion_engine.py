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

def test_fusion_deduplication_single_drone():
    """Ensure that overlapping YOLO box and dilated morphological candidate for the same drone are merged into 1 detection."""
    engine = FusionEngine(mode="adaptive")
    # Tight YOLO detection at (580, 450), 40x25
    yolo_dets = [{"center_x": 580, "center_y": 450, "width": 40, "height": 25, "confidence": 0.88, "polygon": None}]
    # Dilated SEG candidate covering the same drone at (540, 410), 80x80 (centroid: 580, 450)
    seg_candidates = [(540, 410, 80, 80)]
    fused = engine.fuse(seg_candidates, yolo_dets, image_shape=(720, 1280))
    assert len(fused) == 1
    assert fused[0]["center_x"] == 580
    assert fused[0]["center_y"] == 450
    # Confidence should be boosted from multi-modal agreement
    assert fused[0]["confidence"] > 0.88

def test_duplicate_box_detection():
    """Verify _is_duplicate correctly identifies centroid proximity and center inclusion."""
    engine = FusionEngine()
    box1 = (540, 410, 80, 80) # centroid (580, 450)
    box2 = {"center_x": 585, "center_y": 452, "width": 30, "height": 20}
    assert engine._is_duplicate(box1, box2)

    # Distant box should not be duplicate
    box3 = {"center_x": 100, "center_y": 100, "width": 30, "height": 20}
    assert not engine._is_duplicate(box1, box3)

