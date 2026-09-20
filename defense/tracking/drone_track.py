from dataclasses import dataclass

# 10 high-contrast colors (BGR format for OpenCV)
TRACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 255, 0),    # Light Green
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky Blue
]

@dataclass
class DroneTrack:
    """Stateful tracker for a single drone."""
    track_id: int
    bbox: tuple                # (x, y, w, h)
    confidence: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    frames_tracked: int = 1
    lost_count: int = 0
    confirmed: bool = False

    # Configurable thresholds
    min_confirm_frames: int = 3
    max_lost_frames: int = 30

    @property
    def color(self) -> tuple:
        """Get the BGR color for this track based on its ID."""
        return TRACK_COLORS[self.track_id % len(TRACK_COLORS)]

    @property
    def centroid(self) -> tuple[float, float]:
        """Return (cx, cy) center of the bounding box."""
        x, y, w, h = self.bbox
        return (float(x + w / 2.0), float(y + h / 2.0))

    @property
    def is_dead(self) -> bool:
        """True if track has been lost for more than max_lost_frames."""
        return self.lost_count > self.max_lost_frames

    def update(self, bbox: tuple, confidence: float, lat: float = None, lon: float = None, alt: float = None):
        """Update the track with a new detection (re-association)."""
        self.bbox = bbox
        self.confidence = float(confidence)
        self.frames_tracked += 1
        self.lost_count = 0
        if lat is not None:
            self.lat = float(lat)
            self.lon = float(lon)
            self.alt = float(alt)
        if self.frames_tracked >= self.min_confirm_frames:
            self.confirmed = True

    def mark_lost(self):
        """Mark this track as lost for the current frame (no detection)."""
        self.lost_count += 1
