import cv2
import numpy as np
from defense.utils.logger import log

class ImageProcessor:
    """Classical computer vision preprocessing pipeline for small aerial target isolation.

    Implements the exact 3-stage morphological pipeline from the TESA archive:
    1. Grayscale conversion & inRange binarization (sky 60..255 -> inverted to white drone blob)
    2. Rectangular dilation (50x50) to bridge rotor and airframe segments
    3. Rectangular erosion (9x9) to suppress spurious single-pixel noise
    4. Morphological closing (5x5) to eliminate pinholes and smooth boundaries
    5. Inverted mask output (255 background, 0 target object)
    6. Area-gated contour detection (0 to 1% frame area)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.threshold_min = int(cfg.get("threshold_min", 60))
        self.threshold_max = int(cfg.get("threshold_max", 255))
        self.dilation_kernel_size = int(cfg.get("dilation_kernel_size", 50))
        self.dilation_iterations = int(cfg.get("dilation_iterations", 1))
        self.erosion_kernel_size = int(cfg.get("erosion_kernel_size", 9))
        self.erosion_iterations = int(cfg.get("erosion_iterations", 1))
        self.close_kernel_size = int(cfg.get("close_kernel_size", 5))
        self.close_iterations = int(cfg.get("close_iterations", 1))
        self.min_contour_area = float(cfg.get("min_contour_area", 0))
        self.max_area_ratio = float(cfg.get("max_area_ratio", 0.01))
        self.dim_factor = float(cfg.get("dim_factor", 0.1))

    def process(self, image: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        """
        Execute full preprocessing pipeline.
        Returns:
            (binary_mask, candidate_boxes) where:
                binary_mask: uint8 image (255 background, 0 target object) matching stage 2 asset
                candidate_boxes: list of (x, y, w, h) bounding boxes
        """
        gray = self._to_grayscale(image)
        binary = self._binarize(gray)
        dilated, eroded, closed = self._morphological_filter(binary)
        # binary_mask has white background (255) and black objects (0)
        binary_mask = cv2.bitwise_not(closed)
        contours = self._find_contours(closed)
        candidates = self._filter_contours(contours, image.shape)
        log("ImageProc", f"  Morphological filter: {len(candidates)} candidates after area gating")
        return binary_mask, candidates

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        # Sky is bright (threshold_min to threshold_max -> 255)
        # Dark drone target is < threshold_min -> 0
        in_range = cv2.inRange(gray, self.threshold_min, self.threshold_max)
        # Inverted: drone target becomes white (255) blob on black (0) background
        return cv2.bitwise_not(in_range)

    def _morphological_filter(self, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Dilation: connect disconnected drone parts (rotor/body)
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilation_kernel_size, self.dilation_kernel_size))
        dilated = cv2.dilate(binary, k_dil, iterations=self.dilation_iterations)

        # 2. Erosion: clean up scattered single-pixel noise
        k_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erosion_kernel_size, self.erosion_kernel_size))
        eroded = cv2.erode(dilated, k_ero, iterations=self.erosion_iterations)

        # 3. Morphological close: fill pinholes and smooth perimeter
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_kernel_size, self.close_kernel_size))
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, k_close, iterations=self.close_iterations)

        return dilated, eroded, closed

    def _find_contours(self, closed: np.ndarray) -> list:
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _filter_contours(self, contours: list, image_shape: tuple) -> list[tuple]:
        ih, iw = image_shape[:2]
        total_area = float(ih * iw)
        max_area = total_area * self.max_area_ratio
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((int(x), int(y), int(w), int(h)))
        return candidates

    def dim_background(self, image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """
        Dim background while keeping detected candidate regions at full intensity.
        binary_mask: 255 for background, 0 for target objects.
        """
        bg_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        dimmed = image.astype(np.float32) * (1.0 - bg_mask_3ch * (1.0 - self.dim_factor))
        return np.clip(dimmed, 0, 255).astype(np.uint8)

    def _dim_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.dim_background(image, mask)
