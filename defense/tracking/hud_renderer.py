import cv2
import numpy as np

class HUDRenderer:
    """Renders tactical HUD overlay on video frames."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.45
        self.font_thickness = 1
        self.line_height = 16

    def render(self, frame: np.ndarray, tracks: list, frame_number: int) -> np.ndarray:
        """
        Draw all HUD elements on the frame. Returns the annotated frame.
        """
        annotated = frame.copy()
        confirmed = [t for t in tracks if getattr(t, "confirmed", False)]

        # 1. Global status bar
        self._draw_status_bar(annotated, frame_number, len(confirmed))

        # 2. Per-track annotations
        for track in confirmed:
            self._draw_bbox(annotated, track)
            self._draw_center_dot(annotated, track)
            self._draw_telemetry_panel(annotated, track)
            self._draw_track_badge(annotated, track)

        return annotated

    def _draw_status_bar(self, img: np.ndarray, frame_number: int, drone_count: int):
        text = f"Frame: {frame_number:05d}  Drones: {drone_count}"
        cv2.rectangle(img, (10, 10), (280, 42), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (280, 42), (0, 255, 255), 1)
        cv2.putText(img, text, (20, 32), self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_bbox(self, img: np.ndarray, track):
        x, y, w, h = track.bbox
        color = track.color
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    def _draw_center_dot(self, img: np.ndarray, track):
        cx, cy = track.centroid
        cv2.circle(img, (int(round(cx)), int(round(cy))), 3, track.color, -1)

    def _draw_telemetry_panel(self, img: np.ndarray, track):
        x, y, w, h = track.bbox
        lines = [
            f"ID: {track.track_id}",
            f"conf: {track.confidence:.3f}",
            f"lat: {track.lat:.5f}",
            f"lon: {track.lon:.5f}",
            f"alt: {track.alt:.2f}m",
        ]

        panel_w = 125
        panel_h = len(lines) * self.line_height + 8

        # Place panel above bbox unless within 80px of top
        if y < 80:
            panel_y = y + h + 6
        else:
            panel_y = max(0, y - panel_h - 6)

        panel_x = max(0, min(img.shape[1] - panel_w, x))

        # Solid black rect
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), track.color, 1)

        # Draw lines
        for i, line in enumerate(lines):
            text_y = panel_y + (i + 1) * self.line_height
            cv2.putText(img, line, (panel_x + 5, text_y), self.font, self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def _draw_track_badge(self, img: np.ndarray, track):
        x, y, w, h = track.bbox
        badge_text = f"ID:{track.track_id} ({track.confidence:.2f})"
        badge_x = x + w - 75
        badge_y = y + h + 14
        badge_x = max(0, min(img.shape[1] - 80, badge_x))
        badge_y = min(img.shape[0] - 5, badge_y)

        cv2.putText(img, badge_text, (badge_x, badge_y), self.font, 0.40, track.color, 1, cv2.LINE_AA)
