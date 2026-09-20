import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from defense.config import load_config, get_localization_config, get_detection_config
from defense.localization import GBDTRegressor, encode_features
from defense.detection import YoloDetector
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Train GBDT Drone Geo-Localization Models")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset or directory containing labels.csv")
    parser.add_argument("--out", type=str, default="models/", help="Output directory to save trained .bin models")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing images if not in data path")
    parser.add_argument("--yolo-model", type=str, default=None, help="YOLO model path for extracting bboxes if not in CSV")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    cfg = load_config(args.config)
    loc_cfg = get_localization_config(cfg)
    det_cfg = get_detection_config(cfg)

    # Resolve CSV path
    data_path = Path(args.data)
    if data_path.is_file():
        csv_file = data_path
        img_dir = Path(args.image_dir) if args.image_dir else data_path.parent
    else:
        csv_file = data_path / "labels.csv"
        if not csv_file.exists():
            csv_candidates = list(data_path.glob("*.csv"))
            if csv_candidates:
                csv_file = csv_candidates[0]
            else:
                raise FileNotFoundError(f"No CSV file found in {data_path}")
        img_dir = Path(args.image_dir) if args.image_dir else data_path

    log("Train", "🚀 GBDT Drone Geo-Localization Training")
    log("Train", f"Config: {args.config or 'config/default.yaml'}")
    log("Train", f"Data: {csv_file}")
    log("Train", f"Output models directory: {args.out}")

    df = pd.read_csv(csv_file)
    log("Train", f"Loaded {len(df)} records from CSV")

    # Column mapping helper
    def find_col(possible_names):
        for name in possible_names:
            for col in df.columns:
                if col.strip().lower() == name.lower():
                    return col
        return None

    img_col = find_col(["ImageName", "image_name", "filename", "file"])
    lat_col = find_col(["Latitude", "lat", "latitude"])
    lon_col = find_col(["Longitude", "lon", "longitude"])
    alt_col = find_col(["Altitude", "alt", "altitude", "altitude_msl"])

    cx_col = find_col(["center_x", "cx", "x_center"])
    cy_col = find_col(["center_y", "cy", "y_center"])
    w_col = find_col(["width", "w"])
    h_col = find_col(["height", "h"])
    x_col = find_col(["x", "x_min", "left"])
    y_col = find_col(["y", "y_min", "top"])

    detector = None
    if not (cx_col and cy_col and w_col and h_col) and not (x_col and y_col and w_col and h_col):
        yolo_path = args.yolo_model or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
        if Path(yolo_path).exists():
            log("Train", f"Bounding box coordinates not in CSV. Loading detector from {yolo_path} to detect drones...")
            detector = YoloDetector(model_path=yolo_path, confidence=0.5, device=det_cfg.get("device", "cpu"))
        else:
            raise ValueError("CSV does not contain bounding box columns (center_x, center_y, width, height) and YOLO model is not found.")

    features_list = []
    targets_lat = []
    targets_lon = []
    targets_alt = []

    for idx, row in df.iterrows():
        # Get image shape and bbox
        img_name = str(row[img_col]) if img_col else f"img_{idx:04d}.jpg"
        img_file = img_dir / img_name
        if not img_file.exists():
            img_file = img_dir / f"{img_name}.jpg"

        img_shape = (720, 1280)
        if img_file.exists():
            img = cv2.imread(str(img_file))
            if img is not None:
                img_shape = img.shape[:2]

        bbox = None
        if cx_col and cy_col and w_col and h_col:
            cx = float(row[cx_col])
            cy = float(row[cy_col])
            w = float(row[w_col])
            h = float(row[h_col])
            bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))
        elif x_col and y_col and w_col and h_col:
            bbox = (int(round(float(row[x_col]))), int(round(float(row[y_col]))), int(round(float(row[w_col]))), int(round(float(row[h_col]))))
        elif detector is not None and img_file.exists() and img is not None:
            dets = detector.detect(img)
            if dets:
                best = max(dets, key=lambda d: d["confidence"])
                cx, cy, w, h = best["center_x"], best["center_y"], best["width"], best["height"]
                bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))

        if bbox is None:
            continue

        feat = encode_features(bbox, img_shape)
        features_list.append(feat)
        targets_lat.append(float(row[lat_col]))
        targets_lon.append(float(row[lon_col]))
        targets_alt.append(float(row[alt_col]))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            log("Train", f"Processing {idx + 1}/{len(df)}: {img_name} → matched")

    X = np.array(features_list)
    targets = {
        "lat": np.array(targets_lat),
        "lon": np.array(targets_lon),
        "alt": np.array(targets_alt),
    }

    regressor = GBDTRegressor(models_dir=args.out)
    regressor.train(
        features=X,
        targets=targets,
        n_estimators=int(loc_cfg.get("n_estimators", 200)),
        learning_rate=float(loc_cfg.get("learning_rate", 0.1)),
        max_depth=int(loc_cfg.get("max_depth", 5)),
        min_samples_split=int(loc_cfg.get("min_samples_split", 5)),
        min_samples_leaf=int(loc_cfg.get("min_samples_leaf", 2)),
        subsample=float(loc_cfg.get("subsample", 0.9)),
    )
    regressor.save()

    r2_lat = regressor.models["lat"].score(regressor.scaler.transform(X), targets["lat"])
    r2_lon = regressor.models["lon"].score(regressor.scaler.transform(X), targets["lon"])
    r2_alt = regressor.models["alt"].score(regressor.scaler.transform(X), targets["alt"])

    log("Train", "✅ Training complete!")
    log("Train", f"Samples: {len(X)} | Features: {X.shape[1]}")
    log("Train", f"R² scores — lat: {r2_lat:.4f}, lon: {r2_lon:.4f}, alt: {r2_alt:.4f}")
    log("Train", f"Models saved to: {args.out}")

if __name__ == "__main__":
    main()
