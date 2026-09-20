# Model Weights Registry & Training Guide

This directory stores the trained machine learning weights for drone detection and 3D spatial geo-localization.

---

## 1. Expected Model Files

When the system is fully populated, the following files should be located in this directory:

| Filename | Description | Architecture / Model |
| :--- | :--- | :--- |
| `yolo11n-obb.pt` | YOLO11 Nano Oriented Bounding Box detector weights | Ultralytics YOLO11n-OBB |
| `lat.bin` | Latitude regressor model | scikit-learn GradientBoostingRegressor |
| `lon.bin` | Longitude regressor model | scikit-learn GradientBoostingRegressor |
| `alt.bin` | Altitude MSL regressor model | scikit-learn GradientBoostingRegressor |
| `scaler.bin` | Fitted StandardScaler for 20-dim feature vector | scikit-learn StandardScaler |

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

### B. GBDT Geo-Localization Models
To train the Latitude, Longitude, and Altitude regressors from a dataset with paired image coordinates and ground-truth GNSS labels:

```bash
uv run python scripts/train_gbdt.py --data path/to/dataset/ --out models/
```

This will automatically train the three regressors and fit `scaler.bin`, saving them directly to `models/`.
