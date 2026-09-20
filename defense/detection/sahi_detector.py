from pathlib import Path
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from defense.detection.yolo_detector import YoloDetector
from defense.utils.logger import log

class SahiDetector:
    """SAHI (Slicing Aided Hyper Inference) detector for high-resolution images."""

    def __init__(self, model_path: str, confidence: float = 0.75,
                 device: str = "cpu", slice_size: int = 640,
                 overlap_ratio: float = 0.25, nms_threshold: float = 0.60):
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"To obtain this model:\n"
                f"  1. Train with: see notebooks/train_yolo_obb.ipynb\n"
                f"  2. Or download from the project's model registry\n"
                f"  3. Place the .pt file at: {model_path}"
            )
        self.model_path = str(p)
        self.confidence = float(confidence)
        self.device = device
        self.slice_size = int(slice_size)
        self.overlap_ratio = float(overlap_ratio)
        self.nms_threshold = float(nms_threshold)

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=self.model_path,
            confidence_threshold=self.confidence,
            device=self.device,
        )
        log("SAHI", f"Initialized: model={model_path}, slice={slice_size}x{slice_size}, "
                     f"overlap={overlap_ratio}, nms={nms_threshold}")

    def detect(self, image_input) -> list[dict]:
        """
        Run SAHI sliced prediction on an image path (str) or numpy array.
        Returns list of detections matching YoloDetector.detect() format.
        """
        log("SAHI", f"Running sliced inference on: {image_input if isinstance(image_input, str) else 'numpy array'}")
        sliced_result = get_sliced_prediction(
            image=image_input,
            detection_model=self.detection_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=self.nms_threshold,
            verbose=1,
        )

        detections = []
        for obj in sliced_result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()  # [minx, miny, maxx, maxy]
            minx, miny, maxx, maxy = bbox
            w = maxx - minx
            h = maxy - miny
            cx = minx + w / 2.0
            cy = miny + h / 2.0
            conf = float(obj.score.value)

            poly = None
            if hasattr(obj, "mask") and obj.mask is not None and hasattr(obj.mask, "segmentation"):
                seg = obj.mask.segmentation
                if seg and len(seg) >= 4:
                    poly = np.array(seg).reshape(-1, 2)
                    cx, cy, w, h = YoloDetector.obb_polygon_to_bbox(poly)

            detections.append({
                "center_x": int(round(cx)),
                "center_y": int(round(cy)),
                "width": int(round(w)),
                "height": int(round(h)),
                "confidence": conf,
                "polygon": poly,
            })

        log("SAHI", f"  → {len(detections)} detections after NMS merge")
        return detections
