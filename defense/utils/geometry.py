import math

EARTH_RADIUS_M = 6371000.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the geodesic (surface) distance in meters between two lat/lon points.
    Uses the Haversine formula. All inputs/outputs in degrees/meters.
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the initial bearing (azimuth) in degrees [0, 360) from point 1 to point 2.
    Uses the standard forward azimuth formula.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    initial_bearing = math.atan2(y, x)
    degrees = math.degrees(initial_bearing)
    return (degrees + 360.0) % 360.0

def elevation_angle(ground_dist_m: float, alt_diff_m: float) -> float:
    """
    Compute the elevation angle in degrees given horizontal distance and altitude difference.
    """
    if ground_dist_m <= 0.0:
        return 90.0 if alt_diff_m > 0 else (-90.0 if alt_diff_m < 0 else 0.0)
    return math.degrees(math.atan2(alt_diff_m, ground_dist_m))
