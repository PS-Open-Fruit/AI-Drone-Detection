import math
import numpy as np

def encode_features(box: tuple, image_shape: tuple) -> np.ndarray:
    """
    Encode a bounding box into a spatial feature vector for GBDT localization.

    Args:
        box: (x, y, w, h) — top-left corner + width/height in pixels
        image_shape: (image_height, image_width) in pixels

    Returns:
        np.ndarray of shape (20,) containing:
            [0]  nx          — normalized x centered at origin, range [-1, +1]
            [1]  ny          — normalized y centered at origin, range [-1, +1]
            [2]  rx          — normalized x in [0, 1]
            [3]  ry          — normalized y in [0, 1]
            [4]  area_ratio  — (w*h) / (iw*ih), inversely correlated with distance
            [5]  ar          — aspect ratio w/h
            [6]  nw          — normalized width w/iw
            [7]  nh          — normalized height h/ih
            [8]  radial_dist — Euclidean distance of (nx, ny) from origin
            [9]  ang         — polar angle atan2(ny, nx)
            [10] quadrant    — categorical 1..4 (counter-clockwise from top-right)
            [11-14] corner_dists — normalized distances to 4 image corners
            [15] nx²         — quadratic term
            [16] ny²         — quadratic term
            [17] nx·ny       — interaction term
            [18] sin(ang)    — trigonometric
            [19] cos(ang)    — trigonometric
    """
    x, y, w, h = box
    ih, iw = image_shape[:2]

    # Center of bounding box
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Features 0-1: Normalized optical coords centered at (0,0)
    nx = (cx - iw / 2.0) / (iw / 2.0)
    ny = (cy - ih / 2.0) / (ih / 2.0)

    # Features 2-3: Normalized image coords [0, 1]
    rx = cx / iw
    ry = cy / ih

    # Feature 4: Apparent area ratio
    area_ratio = (w * h) / (iw * ih)

    # Features 5-7: Aspect ratio and normalized dimensions
    ar = w / max(1e-6, h)
    nw = w / iw
    nh = h / ih

    # Feature 8: Radial distance from principal axis
    radial_dist = math.sqrt(nx * nx + ny * ny)

    # Feature 9: Polar angle
    ang = math.atan2(ny, nx)

    # Feature 10: Quadrant (1=top-right, 2=top-left, 3=bottom-left, 4=bottom-right)
    if nx >= 0 and ny >= 0:
        quadrant = 1.0
    elif nx < 0 and ny >= 0:
        quadrant = 2.0
    elif nx < 0:
        quadrant = 3.0
    else:
        quadrant = 4.0

    # Features 11-14: Normalized distances to four corners
    diag = math.sqrt(iw * iw + ih * ih)
    corners = [(0, 0), (iw, 0), (0, ih), (iw, ih)]
    corner_dists = [math.sqrt((cx - px)**2 + (cy - py)**2) / diag for px, py in corners]

    # Features 15-17: Polynomial interaction terms
    nx2 = nx * nx
    ny2 = ny * ny
    nxny = nx * ny

    # Features 18-19: Trigonometric
    sin_ang = math.sin(ang)
    cos_ang = math.cos(ang)

    return np.array([
        nx, ny, rx, ry, area_ratio, ar, nw, nh,
        radial_dist, ang, quadrant,
        *corner_dists,
        nx2, ny2, nxny,
        sin_ang, cos_ang
    ], dtype=np.float64)
