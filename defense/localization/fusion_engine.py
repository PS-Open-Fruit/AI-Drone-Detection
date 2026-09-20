import math
from defense.utils.logger import log

class FusionEngine:
    """Multi-modal fusion engine merging YOLO deep learning detections with classical morphological candidates."""

    def __init__(self, mode="adaptive", iou_weight=0.60, shape_weight=0.30,
                 motion_weight=0.10, soft_threshold=0.25, **kwargs):
        self.mode = mode
        self.iou_weight = float(iou_weight)
        self.shape_weight = float(shape_weight)
        self.motion_weight = float(motion_weight)
        self.soft_threshold = float(soft_threshold)

        self.area_ratio_min = float(kwargs.get("area_ratio_min", 0.00005))
        self.area_ratio_max = float(kwargs.get("area_ratio_max", 0.003))
        self.aspect_ratio_min = float(kwargs.get("aspect_ratio_min", 0.4))
        self.aspect_ratio_max = float(kwargs.get("aspect_ratio_max", 3.0))

        log("Fusion", f"Initialized: mode={mode}, weights=(IoU={iou_weight}, "
                       f"Shape={shape_weight}, Motion={motion_weight}), threshold={soft_threshold}")

    def score_candidate(self, candidate_box, yolo_detections, prev_centroids, image_shape) -> float:
        """
        Compute fusion score for a single candidate box.
        S(b) = w_iou * T_iou + w_shape * T_shape + w_motion * T_motion
        """
        t_iou = max((self._compute_iou(candidate_box, det) for det in yolo_detections), default=0.0) if yolo_detections else 0.0
        t_shape = self._shape_score(candidate_box, image_shape)
        t_motion = self._motion_score(candidate_box, prev_centroids, image_shape)

        score = self.iou_weight * t_iou + self.shape_weight * t_shape + self.motion_weight * t_motion
        return float(score)

    def fuse(self, seg_candidates: list, yolo_detections: list, active_tracks: list = None,
             prev_centroids: list = None, image_shape: tuple = (720, 1280)) -> list[dict]:
        """
        Apply the configured fusion mode to merge candidates.
        Returns a list of accepted detection dicts.
        """
        accepted = []

        if self.mode == "yolo":
            accepted = list(yolo_detections)
        elif self.mode == "seg":
            accepted = [self._to_det_dict(c) for c in seg_candidates]
        elif self.mode == "soft":
            accepted = list(yolo_detections)
            for c in seg_candidates:
                score = self.score_candidate(c, yolo_detections, prev_centroids, image_shape)
                if score >= self.soft_threshold:
                    # check overlap with existing accepted
                    if not any(self._compute_iou(c, acc) > 0.40 for acc in accepted):
                        accepted.append(self._to_det_dict(c, conf=score))
        elif self.mode == "hard":
            accepted = list(yolo_detections)
            for c in seg_candidates:
                max_iou = max((self._compute_iou(c, det) for det in yolo_detections), default=0.0)
                if max_iou >= 0.20:
                    if not any(self._compute_iou(c, acc) > 0.40 for acc in accepted):
                        accepted.append(self._to_det_dict(c, conf=0.75))
        elif self.mode == "filter":
            accepted = list(yolo_detections)
            ih, iw = image_shape[:2]
            diag = math.sqrt(iw * iw + ih * ih)
            for c in seg_candidates:
                max_iou = max((self._compute_iou(c, det) for det in yolo_detections), default=0.0)
                is_near_track = False
                if active_tracks:
                    cx, cy = self._get_centroid(c)
                    for t in active_tracks:
                        tx, ty = t.centroid if hasattr(t, "centroid") else (t["center_x"], t["center_y"])
                        if math.sqrt((cx - tx)**2 + (cy - ty)**2) <= 0.10 * diag:
                            is_near_track = True
                            break
                if max_iou >= 0.20 or is_near_track:
                    if not any(self._compute_iou(c, acc) > 0.40 for acc in accepted):
                        accepted.append(self._to_det_dict(c, conf=0.70))
        elif self.mode == "adaptive":
            accepted = list(yolo_detections)
            ih, iw = image_shape[:2]
            diag = math.sqrt(iw * iw + ih * ih)

            for c in seg_candidates:
                base_score = self.score_candidate(c, yolo_detections, prev_centroids, image_shape)
                boost = 0.0
                if active_tracks:
                    cx, cy = self._get_centroid(c)
                    for t in active_tracks:
                        confirmed = getattr(t, "confirmed", True)
                        if confirmed:
                            tx, ty = t.centroid if hasattr(t, "centroid") else (t["center_x"], t["center_y"])
                            if math.sqrt((cx - tx)**2 + (cy - ty)**2) <= 0.10 * diag:
                                boost = 0.50
                                break
                final_score = min(1.0, base_score + boost)
                if final_score >= self.soft_threshold:
                    # Avoid duplicate bounding box with existing YOLO detection
                    if not any(self._compute_iou(c, acc) > 0.40 for acc in accepted):
                        accepted.append(self._to_det_dict(c, conf=final_score))
        else:
            accepted = list(yolo_detections)

        log("Fusion", f"  {self.mode}: {len(seg_candidates)} seg + {len(yolo_detections)} yolo → {len(accepted)} accepted")
        return accepted

    def _to_xywh(self, box) -> tuple[int, int, int, int]:
        if isinstance(box, dict):
            cx = box["center_x"]
            cy = box["center_y"]
            w = box["width"]
            h = box["height"]
            return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h))
        return int(box[0]), int(box[1]), int(box[2]), int(box[3])

    def _to_det_dict(self, box, conf: float = 0.5) -> dict:
        if isinstance(box, dict):
            return box
        x, y, w, h = box
        return {
            "center_x": int(round(x + w / 2.0)),
            "center_y": int(round(y + h / 2.0)),
            "width": int(round(w)),
            "height": int(round(h)),
            "confidence": float(conf),
            "polygon": None,
        }

    def _get_centroid(self, box) -> tuple[float, float]:
        if isinstance(box, dict):
            return float(box["center_x"]), float(box["center_y"])
        x, y, w, h = box
        return float(x + w / 2.0), float(y + h / 2.0)

    def _compute_iou(self, box1, box2) -> float:
        x1, y1, w1, h1 = self._to_xywh(box1)
        x2, y2, w2, h2 = self._to_xywh(box2)

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        inter = max(0, xb - xa) * max(0, yb - ya)
        union = (w1 * h1) + (w2 * h2) - inter
        return float(inter / union) if union > 0 else 0.0

    def _shape_score(self, box, image_shape: tuple) -> float:
        ih, iw = image_shape[:2]
        x, y, w, h = self._to_xywh(box)
        area_ratio = (w * h) / float(iw * ih)
        ar = w / max(1e-6, float(h))

        score = 1.0
        if area_ratio < self.area_ratio_min or area_ratio > self.area_ratio_max:
            score *= 0.4
        if ar < self.aspect_ratio_min or ar > self.aspect_ratio_max:
            score *= 0.6
        return score

    def _motion_score(self, box, prev_centroids: list, image_shape: tuple) -> float:
        if not prev_centroids:
            return 1.0
        ih, iw = image_shape[:2]
        diag = math.sqrt(iw * iw + ih * ih)
        cx, cy = self._get_centroid(box)
        min_d = min(math.sqrt((cx - px)**2 + (cy - py)**2) for px, py in prev_centroids)
        return 1.0 if min_d <= 0.10 * diag else 0.7
