import argparse
from pathlib import Path
import time
import pandas as pd
import cv2
from defense.config import load_config, get_detection_config, get_localization_config, get_camera_config
from defense.detection import build_drone_detector
from defense.localization import GeoLocalizer, FusionEngine, encode_features
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Drone Localization — Full Detection + Geo-Regression Pipeline")
    parser.add_argument("--src", type=str, required=True, help="Input directory of images")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO model (.pt)")
    parser.add_argument("--models-dir", "--gbdt-models", dest="models_dir", type=str, default=None,
                        help="Directory containing the trained localizer (geo_localizer.bin)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confirmation confidence threshold")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    cfg = load_config(args.config)
    det_cfg = get_detection_config(cfg)
    loc_cfg = get_localization_config(cfg)
    if args.conf is not None:
        det_cfg["confidence"] = args.conf
    models_dir = args.models_dir or loc_cfg.get("models_dir", "models/")

    src_dir = Path(args.src)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    image_paths = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg")))

    # Startup Banner
    log("Defense", "📍 Drone Localization — Detection + Geo-Regression")
    log("Defense", f"Config: {args.config or 'config/default.yaml'}")
    log("Defense", f"Localizer: {models_dir} | Input: {src_dir} ({len(image_paths)} images found)")

    detector = build_drone_detector(det_cfg, yolo_path=args.yolo_model)
    fusion_engine = FusionEngine.from_config(loc_cfg)
    localizer = GeoLocalizer(get_camera_config(cfg), models_dir=models_dir)
    localizer.load()

    start_time = time.time()
    records = []
    n_fallback = 0

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            log("Localize", f"❌ Failed to read {img_path.name}")
            continue

        # One drone per image: the single most likely box (same rule used for training)
        result = detector.detect(img)
        bbox, source, _ = detector.best_box(result, fusion_engine, img.shape[:2])
        if bbox is not None:
            pred = localizer.predict(encode_features(bbox, img.shape[:2]))
        else:
            pred = localizer.fallback
            n_fallback += 1
            source = "fallback (no detection)"

        records.append({
            "ImageName": img_path.name,
            "Latitude": float(pred["lat"]),
            "Longitude": float(pred["lon"]),
            "Altitude": float(pred["alt"]),
        })
        log("Localize", f"Image {idx}/{len(image_paths)}: {img_path.name} [{source}] → "
                        f"lat={pred['lat']:.5f}, lon={pred['lon']:.5f}, alt={pred['alt']:.2f}m")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=["ImageName", "Latitude", "Longitude", "Altitude"])
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start_time
    log("Defense", "✅ Localization complete!")
    log("Defense", f"Processed {len(image_paths)} images in {elapsed:.1f}s")
    log("Defense", f"Localized {len(records) - n_fallback} drones; {n_fallback} images used the fallback position")
    log("Defense", f"Output: {out_path}")

if __name__ == "__main__":
    main()
