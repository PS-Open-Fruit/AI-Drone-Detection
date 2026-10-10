import numpy as np
from defense.detection.image_processor import ImageProcessor
from defense.detection.drone_detector import DroneDetector
from defense.localization.fusion_engine import FusionEngine

PROC_CFG = {
    "threshold_min": 60, "dilation_kernel_size": 15, "erosion_kernel_size": 5, "close_kernel_size": 3,
    "max_area_ratio": 0.01, "suppress_timestamp": True, "timestamp_y_max": 120, "timestamp_x_ratio": 0.60,
    "suppress_foliage": True, "foliage_y_ratio": 0.88,
}


def _frame():
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    img[400:420, 900:920] = 30          # drone in the sky
    img[50:70, 1500:1600] = 30          # OSD timestamp digits
    return img


def _det(cx, cy, conf, w=20, h=20):
    return {"center_x": cx, "center_y": cy, "width": w, "height": h, "confidence": conf, "polygon": None}


class FakeYolo:
    """Returns fixed full-frame detections (frame mode) and records the inputs it saw."""

    def __init__(self, dets):
        self.dets = dets
        self.calls = []

    def detect(self, image, imgsz=None):
        self.calls.append((image.shape, imgsz))
        return [dict(d) for d in self.dets]


def test_confirmation_zones_and_standalone_threshold():
    yolo = FakeYolo([_det(910, 410, 0.80),      # on the morphological candidate -> confirmed
                     _det(300, 700, 0.65),      # static scene object, no candidate -> dropped
                     _det(1550, 60, 0.90)])     # inside the OSD banner zone -> dropped
    det = DroneDetector(ImageProcessor(PROC_CFG), yolo, {"yolo_mode": "frame", "yolo_imgsz": 1280,
                                                         "confidence": 0.4})
    res = det.detect(_frame())
    assert [(d["center_x"], d["source"]) for d in res["yolo"]] == [(910, "yolo+seg")]
    assert yolo.calls[0][1] == 1280

    det.standalone_conf = 0.6
    sources = sorted(d["source"] for d in det.detect(_frame())["yolo"])
    assert sources == ["yolo", "yolo+seg"]


def test_tight_box_is_dark_pixel_extent():
    det = DroneDetector(ImageProcessor(PROC_CFG), None, {})
    res = det.detect(_frame())
    assert len(res["seg_candidates"]) == 1
    x, y, w, h = res["seg_candidates"][0]
    assert w > 20 and h > 20                      # dilated blob is larger than the drone
    assert res["seg_boxes"][0] == (900, 400, 20, 20)


def test_crop_mode_maps_detections_back_to_frame():
    class CropYolo:
        def __init__(self):
            self.shapes = []

        def detect(self, image, imgsz=None):
            self.shapes.append(image.shape[:2])
            h, w = image.shape[:2]
            return [_det(w / 2, h / 2, 0.8)]       # drone at the crop centre

    yolo = CropYolo()
    det = DroneDetector(ImageProcessor(PROC_CFG), yolo, {"yolo_mode": "crops", "crop_size": 320})
    res = det.detect(_frame())
    assert yolo.shapes == [(320, 320)]
    (d,) = res["yolo"]
    assert abs(d["center_x"] - 910) <= 2 and abs(d["center_y"] - 410) <= 2
    assert d["source"] == "yolo+seg"


def test_best_box_priority():
    fusion = FusionEngine(mode="adaptive")
    det = DroneDetector(ImageProcessor(PROC_CFG), FakeYolo([_det(910, 410, 0.7, w=24, h=18)]),
                        {"yolo_mode": "frame"})
    box, source, _ = det.best_box(det.detect(_frame()), fusion, (1080, 1920))
    assert source == "yolo+seg" and box == (898, 401, 24, 18)

    det_no_yolo = DroneDetector(ImageProcessor(PROC_CFG), None, {})
    box, source, _ = det_no_yolo.best_box(det_no_yolo.detect(_frame()), fusion, (1080, 1920))
    assert source == "seg" and box == (900, 400, 20, 20)

    empty = DroneDetector(ImageProcessor(PROC_CFG), None, {})
    sky = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    assert empty.best_box(empty.detect(sky), fusion, (1080, 1920))[0] is None
