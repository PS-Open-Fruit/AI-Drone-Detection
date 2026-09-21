import cv2
import numpy as np
from pathlib import Path
from defense.detection.image_processor import ImageProcessor

def test_image_processor_synthetic():
    # 200x200 bright sky (gray=220) with a dark drone patch (gray=30)
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    # Put dark drone in center: 20x20 box at (90, 90)
    img[90:110, 90:110] = 30

    processor = ImageProcessor({
        "threshold_min": 60,
        "threshold_max": 255,
        "dilation_kernel_size": 15,
        "erosion_kernel_size": 5,
        "close_kernel_size": 3,
        "min_contour_area": 0,
        "max_area_ratio": 0.5,
    })

    mask, candidates = processor.process(img)
    assert mask.shape == (200, 200)
    # Background must be 255
    assert mask[10, 10] == 255
    # Center target must be 0 (black in binary mask)
    assert mask[100, 100] == 0
    # Must detect candidate around center
    assert len(candidates) >= 1
    cx = candidates[0][0] + candidates[0][2] // 2
    cy = candidates[0][1] + candidates[0][3] // 2
    assert abs(cx - 100) < 10
    assert abs(cy - 100) < 10

def test_image_processor_archive_regression():
    raw_path = Path("docs/assets/stage1_real_drone_raw.jpg")
    ref_mask_path = Path("docs/assets/stage2_real_drone_mask.jpg")

    if not raw_path.exists() or not ref_mask_path.exists():
        return

    raw = cv2.imread(str(raw_path))
    ref_mask = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)

    processor = ImageProcessor({
        "threshold_min": 60,
        "threshold_max": 255,
        "dilation_kernel_size": 50,
        "erosion_kernel_size": 9,
        "close_kernel_size": 5,
        "min_contour_area": 0,
        "max_area_ratio": 0.01,
    })

    mask, candidates = processor.process(raw)

    # Pixel difference with archived reference asset must be under 3.0 (JPEG compression noise)
    mean_diff = np.mean(np.abs(mask.astype(float) - ref_mask.astype(float)))
    assert mean_diff < 3.0, f"Mask difference too high: {mean_diff}"

    # Verify drone candidate detected
    assert len(candidates) == 1
    x, y, w, h = candidates[0]
    assert 750 <= x <= 780
    assert 260 <= y <= 290

def test_image_processor_dim_background():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    # Target region at center is 0 in binary mask
    mask[40:60, 40:60] = 0

    processor = ImageProcessor({"dim_factor": 0.1})
    dimmed = processor.dim_background(img, mask)

    # Background should be dimmed to ~20 (200 * 0.1)
    assert abs(int(dimmed[10, 10, 0]) - 20) <= 2
    # Target area should stay at 200
    assert dimmed[50, 50, 0] == 200
