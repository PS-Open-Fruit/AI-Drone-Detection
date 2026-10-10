from pathlib import Path
import numpy as np
from defense.utils.logger import log


def build_drone_detector(det_cfg: dict, yolo_path: str = None, device: str = None):
    """Build the shared detection path from the `detection` config section.

    Without YOLO weights it degrades to morphology-only candidates (logged)."""
    from defense.detection.image_processor import ImageProcessor
    from defense.detection.yolo_detector import YoloDetector

    path = yolo_path or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
    yolo = None
    if path and Path(path).exists():
        # YOLO's own threshold equals the confirmation threshold so no confirmable box is dropped early
        yolo = YoloDetector(model_path=path, confidence=float(det_cfg.get("confidence", 0.40)),
                            device=device or det_cfg.get("device", "cpu"))
    else:
        log("DroneDet", f"⚠️ YOLO weights not found at {path}. Proceeding with morphological candidates only.")
    return DroneDetector(ImageProcessor(det_cfg), yolo, det_cfg)


class DroneDetector:
    """Single detection path shared by image detection, localization training/inference and video tracking.

    1. Morphological candidates from ImageProcessor (cheap, high recall, but boxes inflated by dilation).
    2. YOLO on the RAW frame: either the full frame at `imgsz`, or native-resolution crops around each
       candidate (`yolo_mode: crops`, much cheaper on CPU when candidates are few).
    3. YOLO boxes inside the OSD / foliage exclusion zones are dropped.
    4. A YOLO box is "confirmed" when it lies on a morphological candidate. Unconfirmed YOLO boxes are kept
       only above `standalone_conf` (static scene objects otherwise score 0.5-0.7 every frame).
    5. Candidates without a confirmed YOLO box fall back to a tight box: the extent of the dark (binarized)
       pixels inside the dilated candidate, so both box sources describe the drone at the same scale.
    """

    def __init__(self, image_processor, yolo=None, config: dict = None):
        cfg = config or {}
        self.image_processor = image_processor
        self.yolo = yolo
        self.yolo_mode = str(cfg.get("yolo_mode", "crops"))
        self.imgsz = int(cfg.get("yolo_imgsz", 640))
        self.crop_size = int(cfg.get("crop_size", 480))
        self.confirm_conf = float(cfg.get("confidence", 0.40))
        self.confirm_margin = int(cfg.get("confirm_margin", 30))
        sc = cfg.get("standalone_conf", None)
        self.standalone_conf = None if sc in (None, "none", "None") else float(sc)
        self.tight_boxes = bool(cfg.get("tight_boxes", True))
        log("DroneDet", f"Initialized: yolo_mode={self.yolo_mode}, imgsz={self.imgsz}, crop={self.crop_size}, "
                        f"confirm_conf={self.confirm_conf}, standalone_conf={self.standalone_conf}")

    def detect(self, image: np.ndarray) -> dict:
        """
        Returns a dict with:
            seg_candidates: dilated morphological boxes (x, y, w, h)
            seg_boxes:      the same candidates as tight boxes (x, y, w, h)
            yolo_raw:       YOLO detections before zone filtering
            yolo:           kept YOLO detections (confirmed, or standalone above standalone_conf),
                            each with "source" = "yolo+seg" or "yolo"
            binary:         binarized frame (dark pixels = 255), reused for tight boxes
        """
        ip = self.image_processor
        binary = ip._binarize(ip._to_grayscale(image))
        mask, seg = ip.process(image, binary=binary)
        seg_boxes = [self._tight_box(binary, c) for c in seg] if self.tight_boxes else list(seg)

        yolo_raw = self._run_yolo(image, seg) if self.yolo is not None else []
        ih, iw = image.shape[:2]
        kept = []
        for d in yolo_raw:
            if ip.in_exclusion_zone(d["center_x"], d["center_y"], (ih, iw)) or d["confidence"] < self.confirm_conf:
                continue
            if any(self._near(d, c) for c in seg):
                kept.append(dict(d, source="yolo+seg"))
            elif self.standalone_conf is not None and d["confidence"] >= self.standalone_conf:
                kept.append(dict(d, source="yolo"))
        return {"seg_candidates": seg, "seg_boxes": seg_boxes, "yolo_raw": yolo_raw, "yolo": kept,
                "binary": binary, "mask": mask}

    def best_box(self, result: dict, fusion_engine, image_shape: tuple):
        """Single most likely drone box for a still image (localization): a confirmed YOLO box first, then a
        standalone YOLO box, then the morphological candidate with the best shape score (as a tight box)."""
        for source in ("yolo+seg", "yolo"):
            dets = [d for d in result["yolo"] if d["source"] == source]
            if dets:
                d = max(dets, key=lambda d: d["confidence"])
                return self._xywh(d), source, float(d["confidence"])
        seg = result["seg_candidates"]
        if not seg:
            return None, "none", 0.0
        scores = [fusion_engine.score_candidate(c, [], None, image_shape) for c in seg]
        i = int(np.argmax(scores))
        return tuple(result["seg_boxes"][i]), "seg", float(scores[i])

    def _run_yolo(self, image: np.ndarray, seg: list) -> list[dict]:
        if self.yolo_mode == "frame":
            return self.yolo.detect(image, imgsz=self.imgsz)
        ih, iw = image.shape[:2]
        s = min(self.crop_size, ih, iw)
        dets = []
        for (x, y, w, h) in seg:
            cx, cy = x + w / 2.0, y + h / 2.0
            x0 = int(np.clip(cx - s / 2.0, 0, iw - s))
            y0 = int(np.clip(cy - s / 2.0, 0, ih - s))
            for d in self.yolo.detect(image[y0:y0 + s, x0:x0 + s], imgsz=s):
                d = dict(d, center_x=d["center_x"] + x0, center_y=d["center_y"] + y0)
                if d.get("polygon") is not None:
                    d["polygon"] = d["polygon"] + np.array([x0, y0], dtype=d["polygon"].dtype)
                # overlapping crops can see the same drone twice
                if not any(abs(d["center_x"] - e["center_x"]) < 10 and abs(d["center_y"] - e["center_y"]) < 10
                           for e in dets):
                    dets.append(d)
        return dets

    def _near(self, det: dict, cand: tuple) -> bool:
        x, y, w, h = cand
        m = self.confirm_margin
        return x - m <= det["center_x"] <= x + w + m and y - m <= det["center_y"] <= y + h + m

    @staticmethod
    def _tight_box(binary: np.ndarray, cand: tuple) -> tuple:
        x, y, w, h = cand
        ys, xs = np.nonzero(binary[y:y + h, x:x + w])
        if len(xs) == 0:
            return (int(x), int(y), int(w), int(h))
        return (int(x + xs.min()), int(y + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

    @staticmethod
    def _xywh(d: dict) -> tuple:
        return (int(round(d["center_x"] - d["width"] / 2.0)), int(round(d["center_y"] - d["height"] / 2.0)),
                int(round(d["width"])), int(round(d["height"])))
