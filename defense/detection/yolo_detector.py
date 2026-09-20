from pathlib import Path
import numpy as np
from ultralytics import YOLO
from defense.utils.logger import log

class YoloDetector:
    """YOLO OBB / BBox inference wrapper for drone detection."""

    def __init__(self, model_path: str, confidence: float = 0.75, device: str = "cpu"):
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"To obtain this model:\n"
                f"  1. Train with: see notebooks/train_yolo_obb.ipynb\n"
                f"  2. Or download from the project's model registry\n"
                f"  3. Place the .pt file at: {model_path}"
            )
        log("YOLO", f"Loading model: {model_path}")
        log("YOLO", f"Device: {device}, Confidence: {confidence}")
        self.model = YOLO(str(p))
        self.confidence = float(confidence)
        self.device = device

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Run inference on a single image (BGR numpy array).
        Returns list of detections:
            {
                "center_x": int,
                "center_y": int,
                "width": int,
                "height": int,
                "confidence": float,
                "polygon": np.ndarray or None
            }
        """
        # Note: do NOT pass verbose=False to keep progress visible
        results = self.model.predict(image, conf=self.confidence, device=self.device)
        detections = []

        if not results:
            return detections

        res = results[0]
        # Check OBB format
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            for det in res.obb:
                poly = det.xyxyxyxy[0].cpu().numpy()  # shape (4, 2)
                conf = float(det.conf[0].cpu().item())
                cx, cy, w, h = self.obb_polygon_to_bbox(poly)
                detections.append({
                    "center_x": cx,
                    "center_y": cy,
                    "width": w,
                    "height": h,
                    "confidence": conf,
                    "polygon": poly,
                })
        elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            for det in res.boxes:
                xywh = det.xywh[0].cpu().numpy()
                conf = float(det.conf[0].cpu().item())
                cx = int(round(float(xywh[0])))
                cy = int(round(float(xywh[1])))
                w = int(round(float(xywh[2])))
                h = int(round(float(xywh[3])))
                detections.append({
                    "center_x": cx,
                    "center_y": cy,
                    "width": w,
                    "height": h,
                    "confidence": conf,
                    "polygon": None,
                })

        return detections

    @staticmethod
    def obb_polygon_to_bbox(polygon: np.ndarray) -> tuple[int, int, int, int]:
        """
        Convert OBB polygon (4 corner points) to axis-aligned (center_x, center_y, width, height).
        polygon: np.ndarray of shape (4, 2)
        """
        x_min = float(np.min(polygon[:, 0]))
        x_max = float(np.max(polygon[:, 0]))
        y_min = float(np.min(polygon[:, 1]))
        y_max = float(np.max(polygon[:, 1]))
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2.0
        center_y = y_min + height / 2.0
        return int(round(center_x)), int(round(center_y)), int(round(width)), int(round(height))
