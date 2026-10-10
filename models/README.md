# Model Weights Registry & Training Guide

This directory stores the trained machine learning weights for drone detection and 3D spatial geo-localization.

---

## 1. Expected Model Files

When the system is fully populated, the following files should be located in this directory:

| Filename | Description | Architecture / Model |
| :--- | :--- | :--- |
| `yolo11n-obb.pt` | YOLO11 Nano Oriented Bounding Box detector weights | Ultralytics YOLO11n-OBB |
| `geo_localizer.bin` | Drone geo-localizer: bearing, log horizontal distance and height relative to the camera, converted to latitude / longitude / altitude MSL | `defense.localization.GeoLocalizer` (3 scikit-learn GradientBoostingRegressors) |

`lat.bin`, `lon.bin`, `alt.bin` and `scaler.bin` are from the previous `GBDTRegressor` format and are no longer loaded by the pipeline.

---

## 2. Obtaining / Training Models

### A. YOLO11-OBB Detection Weights
1. Open and follow the training workflow in [`notebooks/train_yolo_obb.ipynb`](../notebooks/train_yolo_obb.ipynb).
2. For edge deployment on Raspberry Pi 5, train the Nano model:
   ```bash
   uv run python -c "from ultralytics import YOLO; model = YOLO('yolo11n-obb.pt'); model.train(data='data.yaml', epochs=100, imgsz=640)"
   ```
3. Copy the resulting `best.pt` file to:
   `models/yolo11n-obb.pt`

### B. Geo-Localizer
Train from the competition's Problem 2 data: a directory of `img_XXXX.jpg` + `img_XXXX.csv` pairs, where each CSV holds the drone's ground-truth `Latitude, Longitude, Altitude`:

```bash
uv run python scripts/train_gbdt.py --data data/p2_train
```

Training boxes come from the same detection path used at inference (`DroneDetector.best_box`), so the YOLO weights above must be in place first. This writes `models/geo_localizer.bin`. The training fit is not an accuracy measure; for a held-out score, run:

```bash
uv run python scripts/cv_localization.py
```
