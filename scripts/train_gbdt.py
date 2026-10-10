import argparse
from pathlib import Path
import numpy as np
from defense.config import load_config, get_localization_config, get_detection_config, get_camera_config
from defense.detection import build_drone_detector
from defense.localization import GeoLocalizer, FusionEngine
from defense.localization.dataset import load_labeled_images, extract_samples
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Train the drone geo-localizer (camera-centric GBDT regression)")
    parser.add_argument("--data", type=str, required=True,
                        help="Directory of img_XXXX.jpg + img_XXXX.csv ground-truth pairs (competition format), "
                             "or a CSV with an image-name column plus Latitude/Longitude/Altitude")
    parser.add_argument("--out", type=str, default=None, help="Output directory for geo_localizer.bin")
    parser.add_argument("--image-dir", type=str, default=None, help="Image directory when --data is a single CSV")
    parser.add_argument("--yolo-model", type=str, default=None, help="YOLO model path (defaults to config)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    cfg = load_config(args.config)
    loc_cfg = get_localization_config(cfg)
    det_cfg = get_detection_config(cfg)
    out_dir = args.out or loc_cfg.get("models_dir", "models/")

    log("Train", "🚀 Drone Geo-Localizer Training")
    log("Train", f"Config: {args.config or 'config/default.yaml'} | Data: {args.data} | Output: {out_dir}")

    records = load_labeled_images(args.data, args.image_dir)
    if not records:
        raise ValueError(f"No labeled images found in {args.data}")
    log("Train", f"Loaded {len(records)} labeled images")

    # Training boxes come from the exact inference path (DroneDetector.best_box), never from a separate rule
    detector = build_drone_detector(det_cfg, yolo_path=args.yolo_model)
    samples = extract_samples(records, detector, FusionEngine.from_config(loc_cfg))
    X, Y, has_box = samples["X"], samples["Y"], samples["has_box"]
    if has_box.sum() < 10:
        raise ValueError(f"Only {int(has_box.sum())} images produced a detection; need at least 10 to train")
    sources = np.array(samples["sources"])
    log("Train", f"Boxes: {int(has_box.sum())}/{len(has_box)} images "
                 f"(yolo+seg {int(np.sum(sources == 'yolo+seg'))}, yolo {int(np.sum(sources == 'yolo'))}, "
                 f"seg {int(np.sum(sources == 'seg'))}, none {int(np.sum(~has_box))})")

    localizer = GeoLocalizer(get_camera_config(cfg), models_dir=out_dir,
                             formulation=loc_cfg.get("model_formulation", "polar"))
    localizer.train(
        X[has_box],
        {"lat": Y[has_box, 0], "lon": Y[has_box, 1], "alt": Y[has_box, 2]},
        n_estimators=int(loc_cfg.get("n_estimators", 200)),
        learning_rate=float(loc_cfg.get("learning_rate", 0.1)),
        max_depth=int(loc_cfg.get("max_depth", 5)),
        min_samples_split=int(loc_cfg.get("min_samples_split", 5)),
        min_samples_leaf=int(loc_cfg.get("min_samples_leaf", 2)),
        subsample=float(loc_cfg.get("subsample", 0.9)),
    )
    localizer.save()

    log("Train", "✅ Training complete!")
    log("Train", f"Samples: {int(has_box.sum())} | Features: {X.shape[1]} | Model: {Path(out_dir) / GeoLocalizer.FILENAME}")
    log("Train", "Held-out accuracy: run `uv run python scripts/cv_localization.py` (training-set fit is not a score)")

if __name__ == "__main__":
    main()
