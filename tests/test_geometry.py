from defense.utils.geometry import haversine_distance, bearing, elevation_angle

def test_haversine_same_point():
    """Distance between identical points should be 0."""
    assert haversine_distance(14.305, 101.173, 14.305, 101.173) == 0.0

def test_haversine_known_distance():
    """Check against a known reference distance."""
    # Bangkok (13.7563, 100.5018) to Chiang Mai (18.7883, 98.9853) ≈ 585 km
    dist = haversine_distance(13.7563, 100.5018, 18.7883, 98.9853)
    assert 580_000 < dist < 590_000

def test_bearing_north():
    """Bearing due north should be ~0 degrees."""
    b = bearing(14.0, 101.0, 15.0, 101.0)
    assert abs(b) < 1.0 or abs(b - 360) < 1.0

def test_elevation_angle():
    """Elevation angle tests."""
    assert abs(elevation_angle(100.0, 100.0) - 45.0) < 1e-4
    assert abs(elevation_angle(100.0, 0.0)) < 1e-4
