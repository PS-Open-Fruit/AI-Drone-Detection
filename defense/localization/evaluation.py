import numpy as np
from defense.utils.geometry import haversine_distance, bearing
from defense.utils.logger import log

def evaluate_localization(predictions: list[dict], ground_truth: list[dict],
                          camera_lat: float, camera_lon: float, camera_alt: float) -> dict:
    """
    Compute the competition weighted error metric.

    Args:
        predictions: list of dicts with keys "lat", "lon", "alt"
        ground_truth: list of dicts with keys "lat", "lon", "alt"
        camera_lat, camera_lon, camera_alt: camera station coordinates

    Returns:
        dict with keys:
            "mean_angle_error", "mean_height_error", "mean_range_error",
            "total_weighted_error",
            "median_angle_error", "median_height_error", "median_range_error"
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth")
    if len(predictions) == 0:
        return {
            "mean_angle_error": 0.0, "mean_height_error": 0.0, "mean_range_error": 0.0,
            "total_weighted_error": 0.0,
            "median_angle_error": 0.0, "median_height_error": 0.0, "median_range_error": 0.0
        }

    angle_errors = []
    height_errors = []
    range_errors = []

    for pred, gt in zip(predictions, ground_truth):
        p_lat, p_lon, p_alt = pred["lat"], pred["lon"], pred["alt"]
        g_lat, g_lon, g_alt = gt["lat"], gt["lon"], gt["alt"]

        b_pred = bearing(camera_lat, camera_lon, p_lat, p_lon)
        b_gt = bearing(camera_lat, camera_lon, g_lat, g_lon)
        diff_angle = abs(b_pred - b_gt)
        if diff_angle > 180.0:
            diff_angle = 360.0 - diff_angle
        angle_errors.append(diff_angle)

        height_errors.append(abs(p_alt - g_alt))

        r_pred = haversine_distance(camera_lat, camera_lon, p_lat, p_lon)
        r_gt = haversine_distance(camera_lat, camera_lon, g_lat, g_lon)
        range_errors.append(abs(r_pred - r_gt))

    results = {
        "mean_angle_error": float(np.mean(angle_errors)),
        "mean_height_error": float(np.mean(height_errors)),
        "mean_range_error": float(np.mean(range_errors)),
        "median_angle_error": float(np.median(angle_errors)),
        "median_height_error": float(np.median(height_errors)),
        "median_range_error": float(np.median(range_errors)),
    }
    results["total_weighted_error"] = (
        0.70 * results["mean_angle_error"] +
        0.15 * results["mean_height_error"] +
        0.15 * results["mean_range_error"]
    )

    log("Eval", f"Evaluated {len(predictions)} samples:")
    log("Eval", f"  Angle Error  — mean: {results['mean_angle_error']:.4f}°, median: {results['median_angle_error']:.4f}°")
    log("Eval", f"  Height Error — mean: {results['mean_height_error']:.4f}m, median: {results['median_height_error']:.4f}m")
    log("Eval", f"  Range Error  — mean: {results['mean_range_error']:.4f}m, median: {results['median_range_error']:.4f}m")
    log("Eval", f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log("Eval", f"  Total Weighted Error: {results['total_weighted_error']:.4f}")

    return results
