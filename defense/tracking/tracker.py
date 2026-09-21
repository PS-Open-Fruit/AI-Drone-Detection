import math
import numpy as np
from defense.tracking.drone_track import DroneTrack
from defense.localization.feature_encoder import encode_features
from defense.utils.logger import log

class VideoTracker:
    """Multi-object video tracker for drones with stateful trajectory management and localization."""

    def __init__(self, detector, localizer, fusion_engine, config: dict, image_processor=None):
        """
        Args:
            detector: YoloDetector or SahiDetector (or None)
            localizer: GBDTRegressor (or None)
            fusion_engine: FusionEngine
            config: tracking section from YAML config
            image_processor: ImageProcessor (optional)
        """
        self.detector = detector
        self.localizer = localizer
        self.fusion = fusion_engine
        self.image_processor = image_processor

        cfg = config or {}
        self.max_distance = float(cfg.get("max_distance_threshold", 100))
        self.min_confirm = int(cfg.get("min_track_confidence", 3))
        self.track_memory = int(cfg.get("track_memory", 30))

        self.tracks: list[DroneTrack] = []
        self.next_id: int = 1
        self.frame_count: int = 0

        # Debug intermediate state cache
        self.last_binary_mask = None
        self.last_seg_candidates = []
        self.last_yolo_detections = []
        self.last_fused_detections = []

        log("Tracker", f"Initialized: max_dist={self.max_distance}px, "
                       f"confirm={self.min_confirm} frames, memory={self.track_memory} frames")

    def process_frame(self, frame: np.ndarray) -> list[DroneTrack]:
        """
        Process a single frame. Returns list of confirmed active tracks.
        """
        self.frame_count += 1

        # 1. Detection
        seg_candidates = []
        binary_mask = None
        clean_frame = frame
        if self.image_processor is not None:
            binary_mask, seg_candidates = self.image_processor.process(frame)
            clean_frame = self.image_processor.clean_image(frame, seg_candidates)

        yolo_detections = []
        if self.detector is not None:
            yolo_detections = self.detector.detect(clean_frame)
            ih, iw = frame.shape[:2]
            yolo_detections = [
                d for d in yolo_detections
                if not (d["center_y"] < 120 and d["center_x"] > 0.60 * iw)
                and not (d["center_y"] > 0.88 * ih)
            ]

        # 2. Candidate fusion
        prev_centroids = [t.centroid for t in self.tracks if not t.is_dead]
        detections = self.fusion.fuse(
            seg_candidates=seg_candidates,
            yolo_detections=yolo_detections,
            active_tracks=self.tracks,
            prev_centroids=prev_centroids,
            image_shape=frame.shape[:2]
        )

        # Cache debug state
        self.last_binary_mask = binary_mask
        self.last_seg_candidates = seg_candidates
        self.last_yolo_detections = yolo_detections
        self.last_fused_detections = detections

        # 3. Association
        matches, unmatched_detections, unmatched_tracks = self._associate(detections, self.tracks)

        # 4. Update matched tracks
        for t_idx, d_idx in matches.items():
            det = detections[d_idx]
            cx = det["center_x"]
            cy = det["center_y"]
            w = det["width"]
            h = det["height"]
            bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))
            track = self.tracks[t_idx]
            was_confirmed = track.confirmed
            track.update(bbox, det.get("confidence", 0.8))
            if not was_confirmed and track.confirmed:
                log("Tracker", f"✅ Track ID={track.track_id} confirmed ({track.min_confirm_frames} consecutive frames)")

        # 5. Create new tracks for unmatched detections (with proximity gating to prevent duplicates)
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            cx = det["center_x"]
            cy = det["center_y"]
            w = det["width"]
            h = det["height"]
            bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))
            conf = det.get("confidence", 0.8)

            # Suppress track creation if too close to any existing track or newly added track
            too_close = False
            for t in self.tracks:
                if t.is_dead:
                    continue
                tx, ty = t.centroid
                dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                iou = self._compute_bbox_iou(bbox, t.bbox)
                max_dim = max(w, h, t.bbox[2], t.bbox[3])
                if iou >= 0.10 or dist <= max(50.0, max_dim * 0.70):
                    too_close = True
                    log("Tracker", f"🚫 Suppressed duplicate track creation at ({cx}, {cy}) near track ID={t.track_id} (dist={dist:.1f}px)")
                    break

            if too_close:
                continue

            new_track = DroneTrack(
                track_id=self.next_id,
                bbox=bbox,
                confidence=conf,
                min_confirm_frames=self.min_confirm,
                max_lost_frames=self.track_memory
            )
            self.next_id += 1
            self.tracks.append(new_track)
            log("Tracker", f"🆕 New track created: ID={new_track.track_id} at ({cx}, {cy}) conf={conf:.2f}")

        # 6. Mark unmatched tracks as lost
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.mark_lost()
            log("Tracker", f"⚠️  Track ID={track.track_id} lost (last seen frame {self.frame_count - track.lost_count})")

        # 7. Reap dead tracks
        alive_tracks = []
        for track in self.tracks:
            if track.is_dead:
                log("Tracker", f"💀 Track ID={track.track_id} removed (lost for {track.lost_count} frames)")
            else:
                alive_tracks.append(track)
        self.tracks = alive_tracks

        # 8. Merge co-located tracks in the same drone area
        self._merge_colocated_tracks()

        # 9. Localization on confirmed tracks
        for track in self.tracks:
            if track.confirmed and self.localizer is not None:
                features = encode_features(track.bbox, frame.shape[:2])
                pred = self.localizer.predict(features)
                track.lat = float(pred["lat"])
                track.lon = float(pred["lon"])
                track.alt = float(pred["alt"])
                log("Localization", f"📍 Track ID={track.track_id}: lat={track.lat:.5f}, lon={track.lon:.5f}, alt={track.alt:.2f}m")

        # 10. Return confirmed tracks
        return [t for t in self.tracks if t.confirmed]

    def _merge_colocated_tracks(self):
        """Merge tracks that represent the same physical drone in the same area."""
        if len(self.tracks) < 2:
            return

        # Sort: confirmed tracks first, then more frames tracked, then lower lost count
        sorted_tracks = sorted(
            self.tracks,
            key=lambda t: (t.confirmed, t.frames_tracked, -t.lost_count),
            reverse=True
        )
        kept_tracks = []
        for t in sorted_tracks:
            if t.is_dead:
                continue
            duplicate = False
            tx, ty = t.centroid
            for k in kept_tracks:
                kx, ky = k.centroid
                dist = math.sqrt((tx - kx) ** 2 + (ty - ky) ** 2)
                iou = self._compute_bbox_iou(t.bbox, k.bbox)
                tw, th = t.bbox[2], t.bbox[3]
                kw, kh = k.bbox[2], k.bbox[3]
                max_dim = max(tw, th, kw, kh)
                if iou >= 0.10 or dist <= max(50.0, max_dim * 0.70):
                    duplicate = True
                    k.frames_tracked = max(k.frames_tracked, t.frames_tracked)
                    k.confirmed = k.confirmed or t.confirmed
                    k.confidence = max(k.confidence, t.confidence)
                    log("Tracker", f"🔀 Merged duplicate track ID={t.track_id} into primary track ID={k.track_id} (dist={dist:.1f}px, iou={iou:.2f})")
                    break
            if not duplicate:
                kept_tracks.append(t)
        self.tracks = kept_tracks

    @staticmethod
    def _compute_bbox_iou(box1: tuple, box2: tuple) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = (w1 * h1) + (w2 * h2) - inter
        return float(inter / union) if union > 0 else 0.0

    def _associate(self, detections: list, tracks: list) -> tuple[dict, list, list]:
        """
        Greedy nearest-centroid association between detections and tracks.
        """
        if not detections or not tracks:
            return {}, list(range(len(detections))), list(range(len(tracks)))

        pairs = []
        for t_idx, track in enumerate(tracks):
            tx, ty = track.centroid
            for d_idx, det in enumerate(detections):
                dx = float(det["center_x"])
                dy = float(det["center_y"])
                dist = math.sqrt((dx - tx)**2 + (dy - ty)**2)
                pairs.append((dist, t_idx, d_idx))

        pairs.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()
        matches = {}

        for dist, t_idx, d_idx in pairs:
            if dist > self.max_distance:
                break
            if t_idx not in assigned_tracks and d_idx not in assigned_dets:
                matches[t_idx] = d_idx
                assigned_tracks.add(t_idx)
                assigned_dets.add(d_idx)

        unmatched_detections = [i for i in range(len(detections)) if i not in assigned_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in assigned_tracks]

        return matches, unmatched_detections, unmatched_tracks
