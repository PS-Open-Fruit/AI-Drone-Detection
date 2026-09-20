import base64
import time
import cv2
import numpy as np
import requests
from defense.utils.logger import log

class TelemetryDispatcher:
    """First Alert payload builder and optional network dispatcher."""

    def __init__(self, camera_config: dict = None, api_endpoint: str = ""):
        self.camera = camera_config or {}
        self.api_endpoint = api_endpoint
        self.first_alert_sent = False

    def build_first_alert(self, tracks: list, frame: np.ndarray, frame_id: int) -> dict:
        """
        Build the First Alert JSON payload.
        """
        # Encode frame to JPEG Base64
        success, buf = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buf).decode("utf-8") if success else ""

        objects = []
        for t in tracks:
            cx, cy = t.centroid
            x, y, w, h = t.bbox
            objects.append({
                "frame_id": int(frame_id),
                "track_id": int(t.track_id),
                "classification": "Drone",
                "confidence": float(round(t.confidence, 4)),
                "latitude": float(round(t.lat, 6)),
                "longitude": float(round(t.lon, 6)),
                "altitude_msl": float(round(t.alt, 2)),
                "velocity_mps": None,
                "direction_deg": None,
                "bbox": {
                    "center_x": int(round(cx)),
                    "center_y": int(round(cy)),
                    "width": int(round(w)),
                    "height": int(round(h)),
                }
            })

        payload = {
            "timestamp": time.time(),
            "station_id": self.camera.get("station_id", "STATION_ALPHA"),
            "camera_position": {
                "latitude": self.camera.get("latitude", 0.0),
                "longitude": self.camera.get("longitude", 0.0),
                "altitude_msl": self.camera.get("altitude", 0.0),
            },
            "alert_type": "FIRST_DETECTION",
            "total_drones_detected": len(tracks),
            "objects": objects,
            "image_base64": img_b64,
        }
        return payload

    def maybe_send_alert(self, tracks: list, frame: np.ndarray, frame_id: int):
        """
        Send First Alert if confirmed tracks exist and alert hasn't been sent yet.
        """
        if self.first_alert_sent or not self.api_endpoint:
            return

        confirmed = [t for t in tracks if getattr(t, "confirmed", False)]
        if confirmed:
            payload = self.build_first_alert(confirmed, frame, frame_id)
            log("Telemetry", f"📡 Sending First Alert: {len(confirmed)} drones at frame {frame_id}")
            try:
                resp = requests.post(self.api_endpoint, json=payload, timeout=5)
                log("Telemetry", f"📡 First Alert sent! Response: {resp.status_code}")
            except Exception as e:
                log("Telemetry", f"❌ Failed to send alert: {e}")
            self.first_alert_sent = True
