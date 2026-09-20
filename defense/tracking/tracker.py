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

        log("Tracker", f"Initialized: max_dist={self.max_distance}px, "
                       f"confirm={self.min_confirm} frames, memory={self.track_memory} frames")

    def process_frame(self, frame: np.ndarray) -> list[DroneTrack]:
        """
        Process a single frame. Returns list of confirmed active tracks.
        """
        self.frame_count += 1

        # 1. Detection
        seg_candidates = []
        if self.image_processor is not None:
            _, seg_candidates = self.image_processor.process(frame)

        yolo_detections = []
        if self.detector is not None:
            yolo_detections = self.detector.detect(frame)

        # 2. Candidate fusion
        prev_centroids = [t.centroid for t in self.tracks if not t.is_dead]
        detections = self.fusion.fuse(
            seg_candidates=seg_candidates,
            yolo_detections=yolo_detections,
            active_tracks=self.tracks,
            prev_centroids=prev_centroids,
            image_shape=frame.shape[:2]
        )

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

        # 5. Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            cx = det["center_x"]
            cy = det["center_y"]
            w = det["width"]
            h = det["height"]
            bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))
            conf = det.get("confidence", 0.8)
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

        # 8. Localization on confirmed tracks
        for track in self.tracks:
            if track.confirmed and self.localizer is not None:
                features = encode_features(track.bbox, frame.shape[:2])
                pred = self.localizer.predict(features)
                track.lat = float(pred["lat"])
                track.lon = float(pred["lon"])
                track.alt = float(pred["alt"])
                log("Localization", f"📍 Track ID={track.track_id}: lat={track.lat:.5f}, lon={track.lon:.5f}, alt={track.alt:.2f}m")

        # 9. Return confirmed tracks
        return [t for t in self.tracks if t.confirmed]

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
