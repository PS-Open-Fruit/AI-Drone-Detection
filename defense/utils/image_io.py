import cv2
import numpy as np
from pathlib import Path
from defense.utils.logger import log

def load_image(path: str) -> np.ndarray:
    """Load an image as BGR numpy array. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img

def save_image(path: str, image: np.ndarray):
    """Save a BGR numpy array to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    log("ImageIO", f"Saved: {path}")
