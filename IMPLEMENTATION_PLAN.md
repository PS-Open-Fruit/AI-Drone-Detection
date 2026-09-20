# TESA 2025 Defensive Track — Full Implementation Guide

> **Purpose**: This document is a self-contained guide for an agent to implement the entire Defensive Track codebase from scratch. Follow it **file-by-file, top to bottom**. You should NOT need to read the archived `TESA 2025 Reorganize/` folder — everything you need is here.

---

## Ground Rules

1. **Keep it simple.** Prefer plain functions over deep class hierarchies. Use `@dataclass` for configs. Avoid over-engineering.
2. **Readability first.** Short functions, clear variable names, docstrings on every public function/class. No clever one-liners.
3. **Lightweight target.** The inference pipeline must run on a Raspberry Pi 5 (ARM CPU, no GPU, no AI accelerator). Training scripts may use CUDA.
4. **General-purpose.** No hardcoded competition values. Camera position, thresholds, model paths — all configurable via YAML + CLI args.
5. **Graceful degradation.** If model weights are missing, print a helpful error message explaining how to obtain/train them. Never crash silently.
6. **Use `uv` as the Python package manager.** All setup instructions, scripts, and documentation must use `uv` instead of `pip`. The project uses `pyproject.toml` (not `setup.py` + `requirements.txt`).
7. **Never hide output.** Every script must show the user what is happening in real time. Use `print()` for progress, status, and debug info. Never pass `verbose=False` or suppress stdout/stderr. The user must always know what the system is doing.

---

## Logging & Progress Output Rules

> [!IMPORTANT]
> **This is a critical requirement.** Every module and script must provide clear, real-time feedback to the user. The user should never stare at a blank terminal wondering if the program is working.

### Rules for all code:

1. **Startup banner**: Every CLI script must print what it's doing, what config it loaded, and what files it's operating on.
   ```
   [Defense] Starting drone detection...
   [Defense] Config loaded from: config/default.yaml
   [Defense] YOLO model: models/yolo11n-obb.pt
   [Defense] Input: test_images/ (96 images found)
   [Defense] Confidence threshold: 0.75
   ```

2. **Per-item progress**: When processing multiple items (images, frames), print progress with counts.
   ```
   [Detection] Processing image 1/96: test_0001.jpg
   [Detection]   → 3 drones detected (conf: 0.89, 0.82, 0.76)
   [Detection] Processing image 2/96: test_0002.jpg
   [Detection]   → 0 drones detected
   ```

3. **Frame-by-frame video progress**: During video tracking, print a status line regularly (every frame or every N frames, but NEVER silently process thousands of frames).
   ```
   [Tracker] Frame 100/3600 | FPS: 12.4 | Active drones: 2 | Tracks: 3 (1 lost)
   [Tracker] Frame 200/3600 | FPS: 11.8 | Active drones: 2 | Tracks: 3 (1 lost)
   ```

4. **Key events**: Always announce important state changes:
   ```
   [Tracker] 🆕 New track created: ID=1 at (640, 320) conf=0.89
   [Tracker] ✅ Track ID=1 confirmed (3 consecutive frames)
   [Tracker] ⚠️  Track ID=2 lost (last seen frame 142)
   [Tracker] 💀 Track ID=2 removed (lost for 30 frames)
   [Tracker] 📡 First Alert sent! 2 drones detected at frame 145
   [Localization] 📍 Track ID=1: lat=14.30485, lon=101.17280, alt=40.52m
   ```

5. **Completion summary**: Every script must print a final summary when done.
   ```
   [Defense] ✅ Done! Processed 96 images in 34.2s
   [Defense] Total detections: 142 | Avg per image: 1.48
   [Defense] Output saved to: predictions.csv
   ```

6. **YOLO inference**: Do NOT pass `verbose=False` to YOLO's `.predict()` method. Let YOLO's own progress output show. The user should see model loading, inference speed, etc.

7. **Training progress**: During GBDT or YOLO training, print epoch/iteration progress, metrics, and estimated time remaining.

8. **Error context**: When something fails, print what was being processed and why.
   ```
   [Detection] ❌ Failed to process test_0042.jpg: image file is corrupted
   [Detection]    Skipping and continuing...
   ```

### Implementation pattern:

Use a simple prefix-based logging pattern — just `print()` with `[ModuleName]` prefixes. Do NOT use Python's `logging` module (it adds complexity and can suppress output by default). Keep it simple:

```python
def log(module: str, msg: str):
    """Simple logging helper. Always prints to stdout."""
    print(f"[{module}] {msg}")
```

Put this in `defense/utils/logger.py` and import it everywhere.

---

## Final Repo Structure

```
AI-Drone-Detection/
├── README.md
├── pyproject.toml                      # Project config for uv (replaces setup.py + requirements.txt)
├── config/
│   └── default.yaml
├── defense/
│   ├── __init__.py
│   ├── config.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── image_processor.py
│   │   ├── yolo_detector.py
│   │   ├── sahi_detector.py
│   │   └── confidence_tuner.py
│   ├── localization/
│   │   ├── __init__.py
│   │   ├── feature_encoder.py
│   │   ├── gbdt_regressor.py
│   │   ├── fusion_engine.py
│   │   └── evaluation.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── drone_track.py
│   │   ├── tracker.py
│   │   ├── hud_renderer.py
│   │   └── telemetry.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── video_io.py
│       ├── image_io.py
│       └── geometry.py
├── scripts/
│   ├── detect_images.py
│   ├── localize_drones.py
│   ├── track_video.py
│   ├── train_gbdt.py
│   ├── tune_confidence.py
│   ├── prepare_dataset.py
│   └── generate_submission.py
├── notebooks/
│   └── train_yolo_obb.ipynb
├── models/
│   ├── .gitkeep
│   └── README.md               # Instructions for obtaining/training models
├── docs/
│   ├── 01_object_detection.md
│   ├── 02_drone_localization.md
│   ├── 03_drone_tracking.md
│   ├── 04_edge_deployment.md
│   ├── architecture.md
│   └── assets/                 # Copy from TESA 2025 Reorganize/Defense_Challenge_Documentation/assets/
├── tests/
│   ├── test_feature_encoder.py
│   ├── test_fusion_engine.py
│   ├── test_drone_track.py
│   └── test_geometry.py
└── .gitignore
```

---

## Phase 0: Project Scaffolding

Create all directories and boilerplate files first, then fill them in.

---

### [NEW] `.gitignore`

```gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.pt
*.bin
*.onnx
*.ncnn
models/*.pt
models/*.bin
!models/.gitkeep
!models/README.md
.venv/
venv/
*.mp4
*.avi
debug_*/
```

---

### [NEW] `pyproject.toml`

This replaces both `setup.py` and `requirements.txt`. It's the single project definition file for `uv`.

```toml
[project]
name = "defense"
version = "1.0.0"
description = "AI-Based Drone Detection, Localization & Tracking System"
readme = "README.md"
requires-python = ">=3.9"

dependencies = [
    "opencv-python>=4.8.0",
    "ultralytics>=8.1.0",
    "sahi>=0.11.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "joblib>=1.3.0",
    "requests>=2.31.0",
    "Pillow>=10.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[project.scripts]
defense-detect = "scripts.detect_images:main"
defense-localize = "scripts.localize_drones:main"
defense-track = "scripts.track_video:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Setup commands using `uv`:**

```bash
# Initialize the project (creates .venv automatically)
uv sync

# Install with dev dependencies (pytest)
uv sync --extra dev

# Run a script
uv run python scripts/track_video.py input.mp4 -o output.mp4

# Run tests
uv run pytest tests/ -v

# Add a new dependency
uv add <package-name>
```

---

### [NEW] `config/default.yaml`

This is the single source of truth for all default parameters. Every script loads this, then CLI args override.

```yaml
# === Camera Station ===
camera:
  latitude: 14.305029     # degrees North
  longitude: 101.173010   # degrees East
  altitude: 37.2          # meters above MSL
  station_id: "STATION_ALPHA"

# === Module 1: Detection ===
detection:
  # Image processing (morphological)
  threshold_min: 0
  threshold_max: 200
  dilation_kernel_size: 50
  close_kernel_size: 5
  min_contour_area: 70
  max_area_ratio: 0.001
  dim_factor: 0.3

  # YOLO
  yolo_model_path: "models/yolo11n-obb.pt"
  confidence: 0.75
  device: "cpu"             # "cpu", "cuda", "0", etc.

  # SAHI
  slice_size: 640
  overlap_ratio: 0.25
  nms_threshold: 0.60

# === Module 2: Localization ===
localization:
  gbdt_models_dir: "models/"    # expects lat.bin, lon.bin, alt.bin inside
  # GBDT hyperparameters (for training)
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 5
  min_samples_split: 5
  min_samples_leaf: 2
  subsample: 0.9

  # Fusion
  fusion_mode: "adaptive"       # adaptive | filter | soft | hard | yolo | seg
  soft_threshold: 0.25
  iou_weight: 0.60
  shape_weight: 0.30
  motion_weight: 0.10
  area_ratio_min: 0.00005
  area_ratio_max: 0.003
  aspect_ratio_min: 0.4
  aspect_ratio_max: 3.0

# === Module 3: Tracking ===
tracking:
  min_track_confidence: 3       # frames before a track is confirmed
  track_memory: 30              # frames to keep a lost track alive
  max_distance_threshold: 100   # pixels for association gating

# === Telemetry ===
telemetry:
  api_endpoint: ""              # leave empty to disable API dispatch
  enabled: false

# === Video Output ===
output:
  codec: "mp4v"
  max_file_size_mb: 200
```

---

## Phase 1: Utilities (`defense/utils/`)

Build these first — everything else depends on them.

---

### [NEW] `defense/utils/__init__.py`

Empty file.

---

### [NEW] `defense/utils/logger.py`

The simple logging helper used by every module. **Do NOT use Python's `logging` module** — it's overkill and can accidentally suppress output. Just print.

```python
import time

def log(module: str, msg: str):
    """Print a timestamped, prefixed log message. Always goes to stdout."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{module}] {msg}")
```

Usage everywhere:
```python
from defense.utils.logger import log

log("Detection", "Processing image 1/96: test_0001.jpg")
log("Detection", "  → 3 drones detected (conf: 0.89, 0.82, 0.76)")
log("Tracker", "🆕 New track created: ID=1 at (640, 320) conf=0.89")
```

---

### [NEW] `defense/utils/geometry.py`

Geodesic math helpers. Used by localization evaluation and telemetry.

Must implement these functions:

```python
import math

EARTH_RADIUS_M = 6371000.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the geodesic (surface) distance in meters between two lat/lon points.
    Uses the Haversine formula. All inputs/outputs in degrees/meters.
    """
    # Convert to radians, apply Haversine, return distance in meters.
    ...

def bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing (azimuth) in degrees [0, 360) from point 1 to point 2.
    Uses the standard forward azimuth formula.
    """
    # atan2(sin(Δlon)·cos(lat2), cos(lat1)·sin(lat2) - sin(lat1)·cos(lat2)·cos(Δlon))
    # Convert result to [0, 360).
    ...

def elevation_angle(ground_dist_m, alt_diff_m):
    """
    Compute the elevation angle in degrees given horizontal distance and altitude difference.
    """
    # atan2(alt_diff, ground_dist), convert to degrees.
    ...
```

Keep it dead simple. No classes, just pure functions.

---

### [NEW] `defense/utils/image_io.py`

```python
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
```

---

### [NEW] `defense/utils/video_io.py`

```python
import cv2
from pathlib import Path
from defense.utils.logger import log

def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    info = get_video_info(cap)
    log("VideoIO", f"Opened: {path}")
    log("VideoIO", f"  Resolution: {info['width']}x{info['height']}, "
                   f"FPS: {info['fps']:.1f}, Frames: {info['frame_count']}")
    return cap

def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Return dict with 'width', 'height', 'fps', 'frame_count'."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

def create_video_writer(path: str, fps: float, width: int, height: int,
                        codec: str = "mp4v") -> cv2.VideoWriter:
    """Create a VideoWriter. Auto-creates parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    log("VideoIO", f"Writer created: {path} ({width}x{height} @ {fps:.1f} FPS, codec={codec})")
    return writer

def extract_frames(video_path: str, output_dir: str, interval: int = 1):
    """Extract every Nth frame from a video to output_dir as JPEG files."""
    cap = open_video(video_path)
    info = get_video_info(cap)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = Path(output_dir) / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if saved % 100 == 0:
                log("VideoIO", f"  Extracted {saved} frames so far (at frame {frame_idx}/{info['frame_count']})")
        frame_idx += 1
    cap.release()
    log("VideoIO", f"✅ Extracted {saved} frames to {output_dir}")
```

---

## Phase 2: Configuration (`defense/config.py`)

### [NEW] `defense/config.py`

Load the YAML config and expose it as a simple dict. Don't over-engineer this — just a loader function and a few helper getters.

```python
import yaml
from pathlib import Path
from defense.utils.logger import log

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"

def load_config(config_path: str = None) -> dict:
    """
    Load YAML config. Falls back to default.yaml if no path given.
    Returns a plain dict.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log("Config", f"Loaded config from: {path}")
    return cfg

def get_camera_config(cfg: dict) -> dict:
    """Shortcut to get cfg['camera']."""
    return cfg.get("camera", {})

def get_detection_config(cfg: dict) -> dict:
    return cfg.get("detection", {})

def get_localization_config(cfg: dict) -> dict:
    return cfg.get("localization", {})

def get_tracking_config(cfg: dict) -> dict:
    return cfg.get("tracking", {})
```

### [NEW] `defense/__init__.py`

```python
"""Defense: AI-Based Drone Detection, Localization & Tracking System."""
__version__ = "1.0.0"
```

---

## Phase 3: Module 1 — Detection (`defense/detection/`)

### [NEW] `defense/detection/__init__.py`

```python
from .image_processor import ImageProcessor
from .yolo_detector import YoloDetector
from .sahi_detector import SahiDetector
```

---

### [NEW] `defense/detection/image_processor.py`

This is the classical computer vision preprocessing pipeline. It isolates small aerial targets from sky backgrounds before feeding to YOLO.

**Class: `ImageProcessor`**

Constructor takes a config dict (from `detection` section of YAML). Store the parameters.

**Methods to implement:**

#### `process(image: np.ndarray) -> tuple[np.ndarray, list[tuple]]`

The main pipeline. Returns `(processed_mask, candidate_boxes)`.

Steps (each should be its own private method for readability):

1. **`_to_grayscale(image)`** → `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`

2. **`_binarize(gray)`** → `cv2.inRange(gray, threshold_min, threshold_max)` then `cv2.bitwise_not(result)`. This inverts so that dark objects on bright sky become white blobs.

3. **`_morphological_filter(binary)`** → Two operations:
   - **Dilation**: `cv2.dilate(binary, kernel)` where kernel is `cv2.getStructuringElement(cv2.MORPH_RECT, (K, K))` with `K = dilation_kernel_size` (default 50). This bridges rotor blurs and airframe fragments into cohesive blobs.
   - **Closing**: `cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)` with a smaller kernel (default 5). Fills internal pinhole noise.

4. **`_find_contours(mask)`** → `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`. Returns list of contours.

5. **`_filter_contours(contours, image_shape)`** → For each contour:
   - Compute bounding rect: `x, y, w, h = cv2.boundingRect(contour)`
   - Compute area: `w * h`
   - Reject if `area < min_contour_area` (default 70)
   - Reject if `area / (image_width * image_height) > max_area_ratio` (default 0.001)
   - Keep the rest as `(x, y, w, h)` tuples

6. **`_dim_background(image, mask)`** → Scale background pixels by `dim_factor` (default 0.3). Candidate regions stay at full intensity.

Keep each method under 15 lines. The `process()` method just chains them together and prints a summary:

```python
log("ImageProc", f"  Morphological filter: {len(candidates)} candidates after area gating")
```

---

### [NEW] `defense/detection/yolo_detector.py`

YOLO11-OBB inference wrapper.

**Class: `YoloDetector`**

```python
from defense.utils.logger import log

class YoloDetector:
    def __init__(self, model_path: str, confidence: float = 0.75, device: str = "cpu"):
        """
        Load YOLO model. If model_path doesn't exist, raise FileNotFoundError
        with a helpful message explaining how to obtain the model.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"To obtain this model:\n"
                f"  1. Train with: see notebooks/train_yolo_obb.ipynb\n"
                f"  2. Or download from the project's model registry\n"
                f"  3. Place the .pt file at: {model_path}"
            )
        log("YOLO", f"Loading model: {model_path}")
        log("YOLO", f"Device: {device}, Confidence: {confidence}")
        # self.model = YOLO(model_path)
        # self.confidence = confidence
        # self.device = device

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Run inference on a single image.
        Returns list of detections, each a dict:
        {
            "center_x": int, "center_y": int,
            "width": int, "height": int,
            "confidence": float,
            "polygon": np.ndarray or None  # OBB polygon if available
        }
        """
        # IMPORTANT: Do NOT pass verbose=False. Let YOLO print its own output.
        # results = self.model.predict(image, conf=self.confidence, device=self.device)
        ...

    @staticmethod
    def obb_polygon_to_bbox(polygon: np.ndarray) -> tuple[int, int, int, int]:
        """
        Convert OBB polygon (4 corner points) to axis-aligned (center_x, center_y, width, height).
        polygon: np.ndarray of shape (4, 2)
        """
        x_min, x_max = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        y_min, y_max = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2
        center_y = y_min + height / 2
        return int(round(center_x)), int(round(center_y)), int(round(width)), int(round(height))
```

> [!NOTE]
> When loading the model, use `from ultralytics import YOLO`. The detect method calls `self.model.predict(image, conf=self.confidence, device=self.device)` — **do NOT pass `verbose=False`**. Let YOLO's own output show so the user sees model loading, inference speed, etc. Parse `results[0].obb` if available (OBB mode) or `results[0].boxes` (standard mode).

---

### [NEW] `defense/detection/sahi_detector.py`

SAHI (Slicing Aided Hyper Inference) integration for detecting small distant drones in high-res images.

**Class: `SahiDetector`**

```python
from defense.utils.logger import log

class SahiDetector:
    def __init__(self, model_path: str, confidence: float = 0.75,
                 device: str = "cpu", slice_size: int = 640,
                 overlap_ratio: float = 0.25, nms_threshold: float = 0.60):
        ...
        # Use sahi.AutoDetectionModel.from_pretrained("yolov8", ...)
        # Store slice params
        log("SAHI", f"Initialized: model={model_path}, slice={slice_size}x{slice_size}, "
                     f"overlap={overlap_ratio}, nms={nms_threshold}")

    def detect(self, image_path: str) -> list[dict]:
        """
        Run SAHI sliced prediction on a single image file.
        Returns same format as YoloDetector.detect().
        """
        log("SAHI", f"Running sliced inference on: {image_path}")
        # Use sahi.predict.get_sliced_prediction()
        # For each prediction object:
        #   - If it has a polygon (OBB), convert with obb_polygon_to_bbox()
        #   - Otherwise use the standard bbox
        # Log the result count
        log("SAHI", f"  → {len(results)} detections after NMS merge")
        # Return list of detection dicts
        ...
```

> [!NOTE]
> SAHI slices the image into overlapping tiles (e.g. 640×640 with 25% overlap), runs YOLO on each tile, then merges results via NMS. This preserves the native pixel resolution of small targets that would otherwise be crushed when resizing the full image to 640×640.

---

### [NEW] `defense/detection/confidence_tuner.py`

Automated sweep to find the optimal confidence threshold.

**Function: `tune_confidence`**

```python
from defense.utils.logger import log

def tune_confidence(detector, image_dir: str, ground_truth_csv: str,
                    thresholds: list[float] = None) -> pd.DataFrame:
    """
    Sweep confidence thresholds and measure detection accuracy.

    Args:
        detector: YoloDetector or SahiDetector instance
        image_dir: Directory of test images
        ground_truth_csv: CSV with columns: image_name, center_x, center_y, width, height
        thresholds: List of thresholds to try. Defaults to [0.10, 0.25, 0.40, ..., 0.90]

    Returns:
        DataFrame with columns: threshold, accuracy_pct, correct, total, rmse, false_positives, false_negatives
    """
    ...
```

The logic for each threshold:
1. Set `detector.confidence = threshold`
2. Run detection on all images
3. Compare predicted drone count per image vs ground truth count
4. "Exact Image Accuracy" = % of images where predicted count == ground truth count
5. RMSE = sqrt(mean of (predicted_count - gt_count)²)
6. FP = sum of over-predictions, FN = sum of under-predictions

**Must print progress per threshold:**
```
[ConfTuner] Testing threshold 0.10 ...
[ConfTuner]   Accuracy: 42.71% | RMSE: 2.572 | FP: 53 | FN: 2
[ConfTuner] Testing threshold 0.25 ...
[ConfTuner]   Accuracy: 58.33% | RMSE: 1.531 | FP: 37 | FN: 3
...
[ConfTuner] ✅ Optimal threshold: 0.75 (Accuracy: 77.08%, RMSE: 0.568)
```

---

## Phase 4: Module 2 — Localization (`defense/localization/`)

### [NEW] `defense/localization/__init__.py`

```python
from .feature_encoder import encode_features
from .gbdt_regressor import GBDTRegressor
from .fusion_engine import FusionEngine
```

---

### [NEW] `defense/localization/feature_encoder.py`

The **exact 19-dimensional feature vector** that maps 2D bounding box coordinates to spatial features for GBDT regression. This is a critical function — implement it precisely.

```python
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
    ih, iw = image_shape

    # Center of bounding box
    cx = x + w / 2
    cy = y + h / 2

    # Features 0-1: Normalized optical coords centered at (0,0)
    nx = (cx - iw / 2) / (iw / 2)
    ny = (cy - ih / 2) / (ih / 2)

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
        quadrant = 1
    elif nx < 0 and ny >= 0:
        quadrant = 2
    elif nx < 0:
        quadrant = 3
    else:
        quadrant = 4

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
    ])
```

> [!IMPORTANT]
> Count the return vector carefully: 11 scalars + 4 corner dists + 3 polynomial + 2 trig = **20 elements**. The original archive documentation claimed "19 dimensions" but miscounted. The actual implementation above produces 20. Make sure GBDT models are trained on this same 20-element feature vector.

---

### [NEW] `defense/localization/gbdt_regressor.py`

Three independent Gradient Boosted Decision Tree regressors — one each for Latitude, Longitude, and Altitude.

```python
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from defense.utils.logger import log

class GBDTRegressor:
    """Multi-target GBDT regressor for drone geo-localization."""

    TARGETS = ["lat", "lon", "alt"]

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = Path(models_dir)
        self.models = {}       # {"lat": GBR, "lon": GBR, "alt": GBR}
        self.scaler = None     # StandardScaler fitted on training features

    def train(self, features: np.ndarray, targets: dict,
              n_estimators=200, learning_rate=0.1, max_depth=5,
              min_samples_split=5, min_samples_leaf=2, subsample=0.9):
        """
        Train 3 GBDT regressors.

        Args:
            features: np.ndarray of shape (N, feature_dim)
            targets: dict {"lat": np.ndarray, "lon": np.ndarray, "alt": np.ndarray}
        """
        log("GBDT", f"Training on {features.shape[0]} samples, {features.shape[1]} features")

        # 1. Fit StandardScaler on features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # 2. Train one GBR per target
        for target_name in self.TARGETS:
            log("GBDT", f"  Training {target_name} regressor (n_estimators={n_estimators})...")
            gbr = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=42,
            )
            gbr.fit(X_scaled, targets[target_name])
            self.models[target_name] = gbr
            log("GBDT", f"  ✅ {target_name} regressor trained (R² on train: {gbr.score(X_scaled, targets[target_name]):.4f})")

        log("GBDT", "✅ All 3 regressors trained successfully")

    def save(self):
        """Save models and scaler to models_dir/"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            path = self.models_dir / f"{name}.bin"
            joblib.dump(model, path)
            log("GBDT", f"  Saved: {path}")
        scaler_path = self.models_dir / "scaler.bin"
        joblib.dump(self.scaler, scaler_path)
        log("GBDT", f"  Saved: {scaler_path}")
        log("GBDT", f"✅ All models saved to {self.models_dir}")

    def load(self):
        """Load models and scaler. Raises FileNotFoundError with helpful message if missing."""
        required = [f"{t}.bin" for t in self.TARGETS] + ["scaler.bin"]
        for fname in required:
            path = self.models_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"GBDT model file not found: {path}\n"
                    f"To train these models, run:\n"
                    f"  uv run python scripts/train_gbdt.py --data <training_data_dir> --out {self.models_dir}"
                )
        for name in self.TARGETS:
            self.models[name] = joblib.load(self.models_dir / f"{name}.bin")
        self.scaler = joblib.load(self.models_dir / "scaler.bin")
        log("GBDT", f"✅ Loaded models from {self.models_dir}: {', '.join(f'{t}.bin' for t in self.TARGETS)}, scaler.bin")

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict lat, lon, alt from feature vector(s).

        Args:
            features: np.ndarray of shape (feature_dim,) or (N, feature_dim)

        Returns:
            dict {"lat": float/array, "lon": float/array, "alt": float/array}
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        X_scaled = self.scaler.transform(features)
        return {name: self.models[name].predict(X_scaled) for name in self.TARGETS}
```

---

### [NEW] `defense/localization/fusion_engine.py`

The fusion engine merges YOLO deep learning detections with classical morphological segmentation candidates, scoring each candidate to filter out noise.

**Class: `FusionEngine`**

```python
from defense.utils.logger import log

class FusionEngine:
    def __init__(self, mode="adaptive", iou_weight=0.60, shape_weight=0.30,
                 motion_weight=0.10, soft_threshold=0.25, **kwargs):
        """
        mode: one of "adaptive", "filter", "soft", "hard", "yolo", "seg"
        """
        self.mode = mode
        self.iou_weight = iou_weight
        self.shape_weight = shape_weight
        self.motion_weight = motion_weight
        self.soft_threshold = soft_threshold
        # Store shape/area plausibility thresholds from kwargs or defaults
        self.area_ratio_min = kwargs.get("area_ratio_min", 0.00005)
        self.area_ratio_max = kwargs.get("area_ratio_max", 0.003)
        self.aspect_ratio_min = kwargs.get("aspect_ratio_min", 0.4)
        self.aspect_ratio_max = kwargs.get("aspect_ratio_max", 3.0)
        log("Fusion", f"Initialized: mode={mode}, weights=(IoU={iou_weight}, "
                       f"Shape={shape_weight}, Motion={motion_weight}), threshold={soft_threshold}")
```

**Core scoring function** (used by `soft` and `adaptive` modes):

```python
def score_candidate(self, candidate_box, yolo_detections, prev_centroids,
                    image_shape) -> float:
    """
    Compute fusion score for a single candidate box.
    S(b) = w_iou * T_iou + w_shape * T_shape + w_motion * T_motion
    """
    # T_iou: max IoU between candidate and any YOLO detection
    t_iou = max(self._compute_iou(candidate_box, det) for det in yolo_detections) \
            if yolo_detections else 0.0

    # T_shape: physical plausibility
    #   area_ratio must be in [area_ratio_min, area_ratio_max], otherwise penalty 0.4
    #   aspect_ratio must be in [aspect_ratio_min, aspect_ratio_max], otherwise penalty 0.6
    t_shape = self._shape_score(candidate_box, image_shape)

    # T_motion: proximity to previous frame centroid
    #   1.0 if distance <= 10% of image diagonal, else 0.7
    t_motion = self._motion_score(candidate_box, prev_centroids, image_shape)

    return self.iou_weight * t_iou + self.shape_weight * t_shape + self.motion_weight * t_motion
```

**Fusion modes** — implement as a `fuse()` method:

| Mode | Logic |
|:---|:---|
| `yolo` | Return only YOLO detections directly. No segmentation. |
| `seg` | Return only morphological segmentation candidates. No YOLO. |
| `soft` | Score all candidates with `score_candidate()`. Return those above `soft_threshold`. |
| `hard` | Keep segmentation candidates only if IoU with any YOLO detection ≥ 0.20. |
| `filter` | Keep segmentation candidates that YOLO confirms, with track persistence fallback. |
| `adaptive` | **(Default)** Use YOLO for initialization. Boost segmentation candidate scores by +0.50 if they're near a confirmed active track. Best for continuity during visual dropouts. |

```python
def fuse(self, seg_candidates, yolo_detections, active_tracks, prev_centroids,
         image_shape) -> list[dict]:
    """
    Apply the configured fusion mode to merge candidates.
    Returns a list of accepted detection dicts.
    """
    if self.mode == "yolo":
        return yolo_detections
    elif self.mode == "seg":
        return seg_candidates
    elif self.mode == "soft":
        ...
    # etc.
    # Always log the result:
    log("Fusion", f"  {self.mode}: {len(seg_candidates)} seg + {len(yolo_detections)} yolo → {len(accepted)} accepted")
```

---

### [NEW] `defense/localization/evaluation.py`

The official competition scoring metric. Useful for evaluating localization model quality.

```python
from defense.utils.logger import log

def evaluate_localization(predictions, ground_truth, camera_lat, camera_lon, camera_alt):
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
    # For each (pred, gt) pair:
    #   angle_error = |bearing(cam, pred) - bearing(cam, gt)|  (in degrees)
    #   height_error = |pred.alt - gt.alt|  (in meters)
    #   range_error = |haversine(cam, pred) - haversine(cam, gt)|  (in meters)
    #
    # Total = 0.70 * mean(angle_errors) + 0.15 * mean(height_errors) + 0.15 * mean(range_errors)
    ...

    # Print results table
    log("Eval", f"Evaluated {len(predictions)} samples:")
    log("Eval", f"  Angle Error  — mean: {results['mean_angle_error']:.4f}°, median: {results['median_angle_error']:.4f}°")
    log("Eval", f"  Height Error — mean: {results['mean_height_error']:.4f}m, median: {results['median_height_error']:.4f}m")
    log("Eval", f"  Range Error  — mean: {results['mean_range_error']:.4f}m, median: {results['median_range_error']:.4f}m")
    log("Eval", f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log("Eval", f"  Total Weighted Error: {results['total_weighted_error']:.4f}")
    return results
```

---

## Phase 5: Module 3 — Tracking (`defense/tracking/`)

### [NEW] `defense/tracking/__init__.py`

```python
from .drone_track import DroneTrack
from .tracker import VideoTracker
from .hud_renderer import HUDRenderer
```

---

### [NEW] `defense/tracking/drone_track.py`

The state machine for a single tracked drone.

```python
from dataclasses import dataclass, field

# 10 high-contrast colors (BGR format for OpenCV)
TRACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 255, 0),    # Light Green
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky Blue
]

@dataclass
class DroneTrack:
    """Stateful tracker for a single drone."""
    track_id: int
    bbox: tuple                # (x, y, w, h)
    confidence: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    frames_tracked: int = 1
    lost_count: int = 0
    confirmed: bool = False

    # Configurable thresholds
    min_confirm_frames: int = 3
    max_lost_frames: int = 30

    @property
    def color(self) -> tuple:
        """Get the BGR color for this track based on its ID."""
        return TRACK_COLORS[self.track_id % len(TRACK_COLORS)]

    @property
    def centroid(self) -> tuple:
        """Return (cx, cy) center of the bounding box."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    @property
    def is_dead(self) -> bool:
        return self.lost_count > self.max_lost_frames

    def update(self, bbox, confidence, lat=None, lon=None, alt=None):
        """Update the track with a new detection (re-association)."""
        self.bbox = bbox
        self.confidence = confidence
        self.frames_tracked += 1
        self.lost_count = 0
        if lat is not None:
            self.lat = lat
            self.lon = lon
            self.alt = alt
        if self.frames_tracked >= self.min_confirm_frames:
            self.confirmed = True

    def mark_lost(self):
        """Mark this track as lost for the current frame (no detection)."""
        self.lost_count += 1
```

---

### [NEW] `defense/tracking/tracker.py`

The main video processing loop.

**Class: `VideoTracker`**

```python
import math
import time
from defense.utils.logger import log

class VideoTracker:
    def __init__(self, detector, localizer, fusion_engine, config: dict):
        """
        Args:
            detector: YoloDetector or SahiDetector
            localizer: GBDTRegressor (or None if localization not needed)
            fusion_engine: FusionEngine
            config: tracking section from YAML config
        """
        self.detector = detector
        self.localizer = localizer
        self.fusion = fusion_engine
        self.max_distance = config.get("max_distance_threshold", 100)
        self.min_confirm = config.get("min_track_confidence", 3)
        self.track_memory = config.get("track_memory", 30)

        self.tracks: list[DroneTrack] = []
        self.next_id: int = 1
        self.frame_count: int = 0

        log("Tracker", f"Initialized: max_dist={self.max_distance}px, "
                        f"confirm={self.min_confirm} frames, memory={self.track_memory} frames")

    def process_frame(self, frame) -> list[DroneTrack]:
        """
        Process a single frame. Returns list of confirmed active tracks.

        Steps:
        1. Run detection (YOLO + optional segmentation)
        2. Fuse candidates
        3. Associate detections to existing tracks (greedy nearest centroid)
        4. Update matched tracks
        5. Create new tracks for unmatched detections — log: "🆕 New track created: ID=N"
        6. Mark unmatched tracks as lost — log: "⚠️ Track ID=N lost"
        7. Reap dead tracks — log: "💀 Track ID=N removed"
        8. Run localization on confirmed tracks (if localizer available)
           — log: "📍 Track ID=N: lat=..., lon=..., alt=..."
        9. Return confirmed tracks for HUD rendering
        """
        self.frame_count += 1
        ...

    def _associate(self, detections, tracks) -> tuple[dict, list, list]:
        """
        Greedy nearest-centroid association.

        For each detection, compute Euclidean distance to each track's centroid.
        Greedily assign closest pairs if distance <= max_distance.

        Returns:
            matches: dict {track_index: detection_index}
            unmatched_detections: list of detection indices
            unmatched_tracks: list of track indices
        """
        # Build cost matrix of Euclidean distances
        # Greedy assignment: pick minimum distance pair, assign, remove both, repeat
        # Reject if distance > self.max_distance
        ...
```

> [!NOTE]
> The association algorithm is **greedy**, not Hungarian. Keep it simple:
> 1. Compute all pairwise distances between detection centroids and track centroids
> 2. Sort all pairs by distance ascending
> 3. Greedily assign the closest available pair (if distance ≤ threshold)
> 4. Skip if either the detection or track is already assigned

---

### [NEW] `defense/tracking/hud_renderer.py`

Draws the tactical HUD overlay on each frame.

**Class: `HUDRenderer`**

```python
import cv2

class HUDRenderer:
    """Renders the tactical HUD overlay on video frames."""

    def render(self, frame, tracks: list, frame_number: int) -> np.ndarray:
        """
        Draw all HUD elements on the frame. Returns the annotated frame.

        Elements:
        1. Global status bar (top-left): "Frame: XXXXX  Drones: N"
        2. For each confirmed track:
           a. Color-coded bounding box (2px thick)
           b. Center dot (3px radius)
           c. Telemetry panel (solid black rect, top-left of bbox):
              - track_id: <ID>
              - conf: <confidence, 3 decimals>
              - lat: <latitude, 5 decimals>
              - lon: <longitude, 5 decimals>
              - alt: <altitude, 2 decimals>m
           d. Track badge (bottom-right of bbox): "ID:N (conf)"
        """
        annotated = frame.copy()
        confirmed = [t for t in tracks if t.confirmed]

        # 1. Global status bar
        self._draw_status_bar(annotated, frame_number, len(confirmed))

        # 2. Per-track annotations
        for track in confirmed:
            self._draw_bbox(annotated, track)
            self._draw_center_dot(annotated, track)
            self._draw_telemetry_panel(annotated, track)
            self._draw_track_badge(annotated, track)

        return annotated
```

**Telemetry panel placement rule**: If the bounding box top edge is within 80 pixels of the image top, place the telemetry panel **below** the bbox instead of above it. This prevents HUD clipping.

Use `cv2.putText` with `cv2.FONT_HERSHEY_SIMPLEX`, scale 0.45, thickness 1. White text on solid black rectangle background.

---

### [NEW] `defense/tracking/telemetry.py`

First Alert payload builder and optional API dispatch.

```python
import json
import base64
import time
import requests
import cv2
import numpy as np
from defense.utils.logger import log

class TelemetryDispatcher:
    def __init__(self, camera_config: dict, api_endpoint: str = ""):
        self.camera = camera_config
        self.api_endpoint = api_endpoint
        self.first_alert_sent = False

    def build_first_alert(self, tracks: list, frame: np.ndarray, frame_id: int) -> dict:
        """
        Build the First Alert JSON payload.

        Schema:
        {
            "timestamp": <unix_epoch>,
            "station_id": "<station_id>",
            "camera_position": {"latitude": ..., "longitude": ..., "altitude_msl": ...},
            "alert_type": "FIRST_DETECTION",
            "total_drones_detected": <N>,
            "objects": [
                {
                    "frame_id": <int>,
                    "track_id": <int>,
                    "classification": "Drone",
                    "confidence": <float>,
                    "latitude": <float>,
                    "longitude": <float>,
                    "altitude_msl": <float>,
                    "velocity_mps": null,
                    "direction_deg": null,
                    "bbox": {"center_x": ..., "center_y": ..., "width": ..., "height": ...}
                },
                ...
            ],
            "image_base64": "<base64_encoded_jpeg>"
        }
        """
        ...

    def maybe_send_alert(self, tracks, frame, frame_id):
        """
        Send First Alert if:
        - Any tracks are confirmed AND
        - First alert hasn't been sent yet AND
        - API endpoint is configured
        """
        if self.first_alert_sent or not self.api_endpoint:
            return
        confirmed = [t for t in tracks if t.confirmed]
        if confirmed:
            payload = self.build_first_alert(confirmed, frame, frame_id)
            log("Telemetry", f"📡 Sending First Alert: {len(confirmed)} drones at frame {frame_id}")
            try:
                resp = requests.post(self.api_endpoint, json=payload, timeout=5)
                log("Telemetry", f"📡 First Alert sent! Response: {resp.status_code}")
            except Exception as e:
                log("Telemetry", f"❌ Failed to send alert: {e}")
            self.first_alert_sent = True
```

---

## Phase 6: CLI Scripts (`scripts/`)

Each script is a standalone entry point that wires together the library modules. Use `argparse` for all CLI arguments.

> [!IMPORTANT]
> Every script must:
> 1. Print a startup banner showing config, inputs, and what it will do
> 2. Print per-item progress as it runs
> 3. Print a completion summary with timing and output locations
> 4. Use `uv run python scripts/<name>.py` in all documentation and help text

---

### [NEW] `scripts/detect_images.py`

**Purpose**: Batch image detection → CSV output (Problem 1).

```
usage: detect_images.py [-h] --model MODEL --test-dir DIR --output CSV
                        [--slice SIZE] [--overlap RATIO] [--conf FLOAT]
                        [--device DEVICE] [--use-sahi] [--config YAML]
```

Logic:
1. Load config (YAML defaults + CLI overrides)
2. Print startup banner:
   ```
   [Defense] 🔍 Drone Detection — Batch Image Mode
   [Defense] Config: config/default.yaml
   [Defense] Model: models/yolo11n-obb.pt
   [Defense] Input: test_images/ (96 images found)
   [Defense] Confidence: 0.75, Device: cpu
   ```
3. Initialize `YoloDetector` or `SahiDetector` (if `--use-sahi`)
4. Iterate all `.jpg`/`.png` in `--test-dir`, printing per-image progress:
   ```
   [Detection] Processing image 1/96: test_0001.jpg → 3 drones
   [Detection] Processing image 2/96: test_0002.jpg → 0 drones
   ```
5. Write CSV: `image_name, center_x, center_y, width, height`
6. Print completion summary:
   ```
   [Defense] ✅ Done! 96 images processed in 34.2s
   [Defense] Total detections: 142 | Avg per image: 1.48
   [Defense] Output: predictions.csv
   ```

---

### [NEW] `scripts/localize_drones.py`

**Purpose**: Run full detection + localization pipeline → CSV output (Problem 2).

```
usage: localize_drones.py [-h] --src DIR --yolo-model PATH --gbdt-models DIR
                          --output CSV [--fusion MODE] [--conf FLOAT]
                          [--config YAML]
```

Logic:
1. Load config, print startup banner
2. Initialize `YoloDetector`, `GBDTRegressor` (`.load()`), `FusionEngine`, `ImageProcessor`
3. For each image in `--src`:
   a. Run `ImageProcessor.process()` to get segmentation candidates
   b. Run `YoloDetector.detect()` to get YOLO detections
   c. Run `FusionEngine.fuse()` to merge
   d. For each accepted detection, `encode_features()` → `GBDTRegressor.predict()`
   e. Print: `[Localize] Image 1/459: img_0001.jpg → 1 drone: lat=14.30485, lon=101.17280, alt=40.52m`
4. Write CSV: `ImageName, Latitude, Longitude, Altitude`
5. Print completion summary

---

### [NEW] `scripts/track_video.py`

**Purpose**: Full video tracking pipeline → annotated MP4 (Problem 3). This is the **main showcase script**.

```
usage: track_video.py [-h] input_video -o OUTPUT [--yolo-model PATH]
                      [--gbdt-models DIR] [--fusion MODE] [--conf FLOAT]
                      [--debug DIR] [--config YAML]
```

Logic:
1. Load config, print startup banner:
   ```
   [Defense] 🎯 Drone Video Tracker
   [Defense] Input: input.mp4 (1280x720, 30.0 FPS, 3600 frames)
   [Defense] Output: output_tracked.mp4
   [Defense] Model: models/yolo11n-obb.pt (conf=0.75)
   [Defense] Fusion: adaptive | Localization: enabled
   ```
2. Initialize all modules: `YoloDetector`, `ImageProcessor`, `GBDTRegressor`, `FusionEngine`, `VideoTracker`, `HUDRenderer`, `TelemetryDispatcher`
3. Open input video, create output `VideoWriter`
4. Frame loop:
   a. Read frame
   b. `tracker.process_frame(frame)` → list of active tracks
   c. `hud.render(frame, tracks, frame_num)` → annotated frame
   d. `telemetry.maybe_send_alert(tracks, frame, frame_num)`
   e. Write annotated frame to output video
   f. Print progress **every frame** (or every 10 frames if performance is a concern):
      ```
      [Tracker] Frame 100/3600 | FPS: 12.4 | Drones: 2 | Tracks: 3 (1 lost)
      ```
5. If `--debug DIR` specified, save intermediate stage frames to 5 subdirectories:
   - `1_original/`, `2_binary_mask/`, `3_detections/`, `4_tracking/`, `5_final_annotated/`
6. Release resources, print summary:
   ```
   [Defense] ✅ Video tracking complete!
   [Defense] Processed 3600 frames in 5m 12s (avg 11.5 FPS)
   [Defense] Total tracks created: 5 | Max simultaneous: 3
   [Defense] Output: output_tracked.mp4 (67.2 MB)
   ```

---

### [NEW] `scripts/train_gbdt.py`

**Purpose**: Train the GBDT localization models from labeled data.

```
usage: train_gbdt.py [-h] --data DIR --out DIR [--yolo-model PATH]
                     [--fusion MODE] [--config YAML]
```

Logic:
1. Print startup banner
2. Load labeled data directory (images with paired CSV ground-truth: `ImageName, Lat, Lon, Alt` or per-drone box labels)
3. For each image, detect drones, match detections to ground-truth labels
   - Print: `[Train] Processing 1/459: img_0001.jpg → 1 detection matched`
4. For each matched detection: `encode_features(box, image_shape)` → feature vector
5. Collect all (feature, target) pairs
6. `GBDTRegressor.train(features, targets)` then `.save()`
7. Print training summary:
   ```
   [Train] ✅ Training complete!
   [Train] Samples: 459 | Features: 20
   [Train] R² scores — lat: 0.9987, lon: 0.9991, alt: 0.9823
   [Train] Models saved to: models/
   ```
8. Optional: evaluate on a held-out split and print localization error metrics

---

### [NEW] `scripts/tune_confidence.py`

```
usage: tune_confidence.py [-h] --model PATH --test-dir DIR --ground-truth CSV
                          --output CSV [--device DEVICE]
```

Runs the confidence tuner and prints results table. Must print each threshold result as it runs (see confidence_tuner.py logging section above).

---

### [NEW] `scripts/prepare_dataset.py`

**Purpose**: Prepare YOLO training dataset from raw video + labels.

```
usage: prepare_dataset.py [-h] --video PATH --labels CSV --output-dir DIR
                          [--frame-interval INT]
```

Logic:
1. Print startup banner
2. Extract frames from video at given interval — print extraction progress
3. Parse label CSV to map frames to annotations
4. Create YOLO-format dataset directory structure:
   ```
   output-dir/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml
   ```
5. Generate `data.yaml` with class names and paths
6. Print summary:
   ```
   [Dataset] ✅ Dataset prepared!
   [Dataset] Train: 380 images | Val: 79 images
   [Dataset] Output: output-dir/data.yaml
   ```

---

### [NEW] `scripts/generate_submission.py`

**Purpose**: Format detection/localization outputs into competition submission CSVs.

This is a simple formatter script — takes our output CSVs and reformats them to match the expected submission columns. Keep it short. Print what it reads and what it writes.

---

## Phase 7: Training Notebook

### [NEW] `notebooks/train_yolo_obb.ipynb`

A Jupyter/Colab notebook for training YOLO11-OBB models. Include:

1. **Environment setup**: Install `ultralytics`, mount Google Drive
2. **Dataset prep**: Download from Roboflow or load from Drive
3. **Training commands**:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n-obb.pt")  # or yolo11x-obb.pt for champion model
   model.train(
       data="data.yaml",
       epochs=100,    # 200 for champion model
       imgsz=640,
       device=0,      # CUDA GPU
       project="runs/obb",
   )
   ```
4. **Validation**: Evaluate mAP, export confusion matrix
5. **Export**: Save best weights for deployment

Include markdown cells explaining each step, referencing the original training runs (Train 6 = nano edge model, Train 10 = champion heavy model).

---

## Phase 8: Documentation (`docs/`)

### Migration Instructions

Copy the 4 markdown files from `TESA 2025 Reorganize/Defense_Challenge_Documentation/` to `docs/`:
- `01_PROBLEM_1_OBJECT_DETECTION.md` → `docs/01_object_detection.md`
- `02_PROBLEM_2_DRONE_LOCALIZATION.md` → `docs/02_drone_localization.md`
- `03_PROBLEM_3_DRONE_TRACKING.md` → `docs/03_drone_tracking.md`
- `04_EDGE_DEPLOYMENT_AND_INTEGRATION.md` → `docs/04_edge_deployment.md`

Copy all files from `Defense_Challenge_Documentation/assets/` → `docs/assets/`.

Update all image paths in the markdown files to use `assets/` relative paths.

### [NEW] `docs/architecture.md`

Write a new architecture overview with:
- Updated Mermaid diagram reflecting the new file structure
- Module dependency graph
- Data flow from camera input → detection → localization → tracking → HUD → telemetry

### [NEW] `models/README.md`

Instructions for obtaining model weights:
1. How to train YOLO11n-OBB (link to notebook)
2. How to train GBDT models (command for `train_gbdt.py`)
3. Expected file names: `yolo11n-obb.pt`, `lat.bin`, `lon.bin`, `alt.bin`, `scaler.bin`

All commands must use `uv run`.

---

## Phase 9: README.md

### [NEW] `README.md`

Structure:
1. **Title & badges** — "AI-Based Drone Detection, Localization & Real-Time Tracking System"
2. **One-paragraph summary** — what this is, what it does
3. **Architecture diagram** — Mermaid flowchart (copy from the original docs, update module names)
4. **Pipeline visualization** — embed `docs/assets/real_drone_pipeline_5_stages.jpg`
5. **Quick Start** (using `uv`):
   ```bash
   # Install dependencies
   uv sync

   # See models/README.md for obtaining model weights

   # Run video tracking
   uv run python scripts/track_video.py input.mp4 -o output.mp4 --yolo-model models/yolo11n-obb.pt

   # Run tests
   uv run pytest tests/ -v
   ```
6. **Module overview table** — 3 rows (Detection, Localization, Tracking) with script, description, key metric
7. **Performance summary** — the key numbers from the archive (97.7% mAP, 3.33 total error, zero ID switches)
8. **Hardware** — Raspberry Pi 5 (ARM CPU, no accelerator) + training on CUDA GPU
9. **Project structure** — the directory tree
10. **License** — placeholder

---

## Phase 10: Tests (`tests/`)

Use `pytest`. Keep tests simple — no fixtures needed beyond basic setup.

Run with: `uv run pytest tests/ -v`

### [NEW] `tests/test_feature_encoder.py`

```python
def test_feature_vector_length():
    """Feature encoder should return a vector of the correct length."""
    from defense.localization.feature_encoder import encode_features
    box = (100, 200, 50, 30)
    shape = (720, 1280)
    features = encode_features(box, shape)
    assert len(features) == 20  # verify exact count

def test_center_box_normalized_coords():
    """A box at the exact image center should have nx=0, ny=0."""
    box = (615, 335, 50, 50)  # center of 1280x720
    shape = (720, 1280)
    features = encode_features(box, shape)
    assert abs(features[0]) < 0.01  # nx ≈ 0
    assert abs(features[1]) < 0.01  # ny ≈ 0

def test_area_ratio_range():
    """Area ratio should be between 0 and 1."""
    box = (0, 0, 100, 100)
    shape = (1080, 1920)
    features = encode_features(box, shape)
    assert 0 < features[4] < 1
```

### [NEW] `tests/test_fusion_engine.py`

```python
def test_soft_fusion_scoring():
    """Score should be in [0, 1] range."""
    ...

def test_yolo_mode_passthrough():
    """In 'yolo' mode, fusion should return YOLO detections unchanged."""
    ...

def test_seg_mode_passthrough():
    """In 'seg' mode, fusion should return segmentation candidates unchanged."""
    ...
```

### [NEW] `tests/test_drone_track.py`

```python
def test_track_confirmation():
    """Track should be confirmed after min_confirm_frames updates."""
    track = DroneTrack(track_id=1, bbox=(0,0,10,10), min_confirm_frames=3)
    assert not track.confirmed
    track.update((1,1,10,10), 0.9)
    track.update((2,2,10,10), 0.9)
    assert track.confirmed  # 3 frames total (1 init + 2 updates)

def test_track_death():
    """Track should die after max_lost_frames of no detections."""
    track = DroneTrack(track_id=1, bbox=(0,0,10,10), max_lost_frames=5)
    for _ in range(6):
        track.mark_lost()
    assert track.is_dead

def test_track_color_cycling():
    """Track colors should cycle through the palette."""
    t1 = DroneTrack(track_id=0, bbox=(0,0,10,10))
    t2 = DroneTrack(track_id=10, bbox=(0,0,10,10))
    assert t1.color == t2.color  # both index 0 in palette
```

### [NEW] `tests/test_geometry.py`

```python
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
```

---

## Implementation Order

Follow this exact sequence:

| Step | Files | Notes |
|:---:|:---|:---|
| 1 | `.gitignore`, `pyproject.toml`, `config/default.yaml` | Scaffolding. Then run `uv sync` to create venv. |
| 2 | `defense/__init__.py`, `defense/utils/logger.py`, `defense/config.py` | Core infra: logging + configuration |
| 3 | `defense/utils/geometry.py`, `defense/utils/image_io.py`, `defense/utils/video_io.py` | Utility functions |
| 4 | `tests/test_geometry.py` | Test geometry utils immediately. Run: `uv run pytest tests/test_geometry.py -v` |
| 5 | `defense/detection/image_processor.py` | Classical CV pipeline |
| 6 | `defense/detection/yolo_detector.py` | YOLO wrapper |
| 7 | `defense/detection/sahi_detector.py` | SAHI wrapper |
| 8 | `defense/detection/confidence_tuner.py` | Threshold optimizer |
| 9 | `defense/localization/feature_encoder.py` | 20-dim spatial features |
| 10 | `tests/test_feature_encoder.py` | Test immediately. Run: `uv run pytest tests/test_feature_encoder.py -v` |
| 11 | `defense/localization/gbdt_regressor.py` | GBDT train/predict |
| 12 | `defense/localization/fusion_engine.py` | Fusion scoring |
| 13 | `tests/test_fusion_engine.py` | Test immediately |
| 14 | `defense/localization/evaluation.py` | Scoring metric |
| 15 | `defense/tracking/drone_track.py` | Track state machine |
| 16 | `tests/test_drone_track.py` | Test immediately |
| 17 | `defense/tracking/tracker.py` | VideoTracker loop |
| 18 | `defense/tracking/hud_renderer.py` | HUD overlay |
| 19 | `defense/tracking/telemetry.py` | First Alert API |
| 20 | `scripts/detect_images.py` | CLI for detection |
| 21 | `scripts/localize_drones.py` | CLI for localization |
| 22 | `scripts/track_video.py` | CLI for tracking (main script) |
| 23 | `scripts/train_gbdt.py` | GBDT training script |
| 24 | `scripts/tune_confidence.py` | Confidence sweep |
| 25 | `scripts/prepare_dataset.py` | Dataset prep |
| 26 | `scripts/generate_submission.py` | Submission formatter |
| 27 | `notebooks/train_yolo_obb.ipynb` | Training notebook |
| 28 | Migrate docs + assets | Copy from archive |
| 29 | `docs/architecture.md`, `models/README.md` | New docs |
| 30 | `README.md` | Project README |

---

## Verification Checklist

After implementation, run these checks:

```bash
# 0. Install dependencies
uv sync --extra dev

# 1. All imports work
uv run python -c "from defense.detection import ImageProcessor, YoloDetector, SahiDetector"
uv run python -c "from defense.localization import encode_features, GBDTRegressor, FusionEngine"
uv run python -c "from defense.tracking import DroneTrack, VideoTracker, HUDRenderer"
uv run python -c "from defense.config import load_config; print(load_config())"

# 2. Unit tests pass
uv run pytest tests/ -v

# 3. CLI scripts show help without errors
uv run python scripts/detect_images.py --help
uv run python scripts/localize_drones.py --help
uv run python scripts/track_video.py --help
uv run python scripts/train_gbdt.py --help

# 4. Config loads correctly (should print config + log message)
uv run python -c "from defense.config import load_config; cfg = load_config(); print(cfg['camera'])"

# 5. Feature encoder produces correct output shape
uv run python -c "
from defense.localization.feature_encoder import encode_features
import numpy as np
f = encode_features((100, 200, 50, 30), (720, 1280))
print(f'Feature vector length: {len(f)}')
print(f'Features: {f}')
"

# 6. Graceful model-missing errors (should print helpful message, not crash)
uv run python -c "
from defense.detection.yolo_detector import YoloDetector
try:
    d = YoloDetector('nonexistent.pt')
except FileNotFoundError as e:
    print('Good:', e)
"
```
