import math
import numpy as np
import pytest
from defense.localization.geo_localizer import GeoLocalizer

CAM = {"latitude": 14.305029, "longitude": 101.173010, "altitude": 37.2}


def _synthetic(n, bearing_center, seed=0):
    """Features whose nx drives bearing and whose box size drives distance, like a fixed pinhole camera."""
    rng = np.random.default_rng(seed)
    nx = rng.uniform(-1, 1, n)
    ny = rng.uniform(-1, 0.2, n)
    dist = rng.uniform(15, 150, n)
    bearing = (bearing_center + 30.0 * nx) % 360.0
    height = dist * np.tan(np.radians(10 - 15 * ny))
    X = np.zeros((n, 20))
    X[:, 0], X[:, 1] = nx, ny
    X[:, 4] = 1.0 / dist ** 2          # area ratio ~ 1/distance^2
    X[:, 6] = 1.0 / dist               # normalized width ~ 1/distance
    lat0 = math.radians(CAM["latitude"])
    R = 6371000.0
    lat = CAM["latitude"] + np.degrees(dist * np.cos(np.radians(bearing)) / R)
    lon = CAM["longitude"] + np.degrees(dist * np.sin(np.radians(bearing)) / (R * math.cos(lat0)))
    return X, {"lat": lat, "lon": lon, "alt": CAM["altitude"] + height}, bearing


@pytest.mark.parametrize("formulation", ["polar", "hybrid"])
def test_fits_synthetic_camera(formulation):
    X, y, _ = _synthetic(300, 220.0)
    loc = GeoLocalizer(CAM, formulation=formulation)
    loc.train(X, y, n_estimators=150)
    p = loc.predict(X)
    assert np.mean(np.abs(p["alt"] - y["alt"])) < 2.0
    assert np.mean(np.abs(p["lat"] - y["lat"])) < 2e-5   # ~2 m


def test_bearing_wraps_through_north():
    X, y, bearing = _synthetic(300, 0.0, seed=1)          # bearings span 330..30 degrees
    assert bearing.min() < 5 and bearing.max() > 355
    loc = GeoLocalizer(CAM)
    loc.train(X, y, n_estimators=150)
    p = loc.predict(X)
    assert np.mean(np.abs(p["lat"] - y["lat"])) < 2e-5
    assert np.mean(np.abs(p["lon"] - y["lon"])) < 2e-5


def test_single_vector_save_load_and_fallback(tmp_path):
    X, y, _ = _synthetic(120, 220.0)
    loc = GeoLocalizer(CAM, models_dir=str(tmp_path))
    loc.train(X, y, n_estimators=50)
    loc.save()
    loaded = GeoLocalizer(CAM, models_dir=str(tmp_path))
    loaded.load()
    one = loaded.predict(X[0])
    assert set(one) == {"lat", "lon", "alt"} and all(isinstance(v, float) for v in one.values())
    assert one == loc.predict(X[0])
    assert abs(loaded.fallback["alt"] - float(np.mean(y["alt"]))) < 1e-9

    other_cam = dict(CAM, latitude=13.0)
    with pytest.raises(ValueError):
        GeoLocalizer(other_cam, models_dir=str(tmp_path)).load()
