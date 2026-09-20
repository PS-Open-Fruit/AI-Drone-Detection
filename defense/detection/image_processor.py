import cv2
import numpy as np
from defense.utils.logger import log

class ImageProcessor:
    """Classical computer vision preprocessing pipeline for small aerial target isolation."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.threshold_min = cfg.get("threshold_min", 0)
        self.threshold_max = cfg.get("threshold_max", 200)
        self.dilation_kernel_size = cfg.get("dilation_kernel_size", 50)
        self.close_kernel_size = cfg.get("close_kernel_size", 5)
        self.min_contour_area = cfg.get("min_contour_area", 70)
        self.max_area_ratio = cfg.get("max_area_ratio", 0.001)
        self.dim_factor = cfg.get("dim_factor", 0.3)

    def process(self, image: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        """
        Execute full preprocessing pipeline.
        Returns:
            (processed_mask, candidate_boxes) where candidate_boxes is a list of (x, y, w, h)
        """
        gray = self._to_grayscale(image)
        binary = self._binarize(gray)
        mask = self._morphological_filter(binary)
        contours = self._find_contours(mask)
        candidates = self._filter_contours(contours, image.shape)
        log("ImageProc", f"  Morphological filter: {len(candidates)} candidates after area gating")
        return mask, candidates

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        in_range = cv2.inRange(gray, self.threshold_min, self.threshold_max)
        return cv2.bitwise_not(in_range)

    def _morphological_filter(self, binary: np.ndarray) -> np.ndarray:
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilation_kernel_size, self.dilation_kernel_size))
        dilated = cv2.dilate(binary, k_dil)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_kernel_size, self.close_kernel_size))
        return cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close)

    def _find_contours(self, mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _filter_contours(self, contours: list, image_shape: tuple) -> list[tuple]:
        ih, iw = image_shape[:2]
        total_area = float(ih * iw)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_contour_area:
                continue
            if (area / total_area) > self.max_area_ratio:
                continue
            candidates.append((int(x), int(y), int(w), int(h)))
        return candidates

    def _dim_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        dimmed = (image.astype(np.float32) * self.dim_factor).astype(np.uint8)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.where(mask_3ch > 0, image, dimmed)
