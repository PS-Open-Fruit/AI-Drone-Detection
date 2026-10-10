import math
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from defense.utils.geometry import EARTH_RADIUS_M
from defense.utils.logger import log


class GeoLocalizer:
    """Drone latitude / longitude / altitude from 20-dim box features, regressed in camera-centric coordinates.

    The competition scores bearing, height and horizontal range as seen from the camera, so the models predict
    those quantities directly instead of raw lat/lon/alt:
        bearing  - offset from the training data's circular-mean bearing (deg), so wrap-around at north is safe
        distance - log horizontal distance from the camera (m)
        height   - altitude relative to the camera (m)

    formulation="polar":  one GBDT per quantity.
    formulation="hybrid": bearing and elevation angle from robust linear fits on the normalized box centre
                          (they extrapolate across flights, unlike trees), log distance from a GBDT,
                          height = distance * tan(elevation).
    """

    FILENAME = "geo_localizer.bin"

    def __init__(self, camera: dict, models_dir: str = "models/", formulation: str = "polar"):
        self.cam_lat = float(camera["latitude"])
        self.cam_lon = float(camera["longitude"])
        self.cam_alt = float(camera["altitude"])
        self.models_dir = Path(models_dir)
        self.formulation = formulation
        self.models = {}
        self.bearing_ref = 0.0
        self.fallback = None  # mean training position, used when an image has no detection

    # --- coordinate helpers (local tangent plane; drones are within a few hundred metres) ---
    def _to_polar(self, lat, lon, alt):
        lat0 = math.radians(self.cam_lat)
        east = np.radians(np.asarray(lon) - self.cam_lon) * EARTH_RADIUS_M * math.cos(lat0)
        north = np.radians(np.asarray(lat) - self.cam_lat) * EARTH_RADIUS_M
        bearing = np.degrees(np.arctan2(east, north)) % 360.0
        return bearing, np.hypot(east, north), np.asarray(alt, dtype=float) - self.cam_alt

    def _to_geo(self, bearing, dist, height):
        lat0 = math.radians(self.cam_lat)
        b = np.radians(bearing)
        lat = self.cam_lat + np.degrees(dist * np.cos(b) / EARTH_RADIUS_M)
        lon = self.cam_lon + np.degrees(dist * np.sin(b) / (EARTH_RADIUS_M * math.cos(lat0)))
        return lat, lon, self.cam_alt + height

    @staticmethod
    def _angle_features(X):
        return np.column_stack([X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])  # nx, ny, nx*ny

    def train(self, features: np.ndarray, targets: dict,
              n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 5,
              min_samples_split: int = 5, min_samples_leaf: int = 2, subsample: float = 0.9):
        X = np.asarray(features, dtype=float)
        self.fallback = {k: float(np.mean(targets[k])) for k in ("lat", "lon", "alt")}
        bearing, dist, height = self._to_polar(targets["lat"], targets["lon"], targets["alt"])
        b = np.radians(bearing)
        self.bearing_ref = float(np.degrees(np.arctan2(np.sin(b).mean(), np.cos(b).mean())) % 360.0)
        b_off = (bearing - self.bearing_ref + 180.0) % 360.0 - 180.0
        gbdt = lambda: GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            subsample=subsample, random_state=42)
        log("GeoLoc", f"Training '{self.formulation}' localizer on {len(X)} samples")
        self.models = {"log_dist": gbdt().fit(X, np.log(np.maximum(dist, 1e-3)))}
        if self.formulation == "polar":
            self.models["bearing"] = gbdt().fit(X, b_off)
            self.models["height"] = gbdt().fit(X, height)
        elif self.formulation == "hybrid":
            A = self._angle_features(X)
            self.models["bearing"] = HuberRegressor(max_iter=2000).fit(A, b_off)
            self.models["elevation"] = HuberRegressor(max_iter=2000).fit(A, np.degrees(np.arctan2(height, dist)))
        else:
            raise ValueError(f"Unknown formulation: {self.formulation}")
        log("GeoLoc", "✅ Localizer trained")

    def predict(self, features: np.ndarray) -> dict:
        """Same contract as GBDTRegressor.predict: floats for one vector, arrays for a batch."""
        X = np.asarray(features, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(1, -1)
        dist = np.exp(self.models["log_dist"].predict(X))
        if self.formulation == "polar":
            b_off = self.models["bearing"].predict(X)
            height = self.models["height"].predict(X)
        else:
            A = self._angle_features(X)
            b_off = self.models["bearing"].predict(A)
            height = dist * np.tan(np.radians(self.models["elevation"].predict(A)))
        lat, lon, alt = self._to_geo((self.bearing_ref + b_off) % 360.0, dist, height)
        if is_1d:
            return {"lat": float(lat[0]), "lon": float(lon[0]), "alt": float(alt[0])}
        return {"lat": lat, "lon": lon, "alt": alt}

    def save(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / self.FILENAME
        joblib.dump({"formulation": self.formulation, "bearing_ref": self.bearing_ref, "models": self.models,
                     "fallback": self.fallback, "camera": [self.cam_lat, self.cam_lon, self.cam_alt]}, path)
        log("GeoLoc", f"✅ Saved: {path}")

    def load(self):
        path = self.models_dir / self.FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Localizer model not found: {path}\n"
                f"To train it, run:\n"
                f"  uv run python scripts/train_gbdt.py --data <training_data_dir> --out {self.models_dir}"
            )
        state = joblib.load(path)
        if [self.cam_lat, self.cam_lon, self.cam_alt] != state["camera"]:
            raise ValueError(f"{path} was trained for camera {state['camera']}, "
                             f"but the config camera is {[self.cam_lat, self.cam_lon, self.cam_alt]}")
        self.formulation = state["formulation"]
        self.bearing_ref = state["bearing_ref"]
        self.models = state["models"]
        self.fallback = state.get("fallback")
        log("GeoLoc", f"✅ Loaded '{self.formulation}' localizer from {path}")
