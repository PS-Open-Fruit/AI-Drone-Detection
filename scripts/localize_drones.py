import argparse
from pathlib import Path
import time
import pandas as pd
import cv2
from defense.config import load_config, get_detection_config, get_localization_config
from defense.detection import YoloDetector, ImageProcessor
from defense.localization import GBDTRegressor, FusionEngine, encode_features
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Drone Localization — Full Detection + Geo-Regression Pipeline")
    parser.add_argument("--src", type=str, required=True, help="Input directory of images")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO model (.pt)")
    parser.add_argument("--gbdt-models", type=str, default=None, help="Directory containing GBDT models")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--fusion", type=str, default=None, help="Fusion mode: adaptive, filter, soft, hard, yolo, seg")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    cfg = load_config(args.config)
    det_cfg = get_detection_config(cfg)
    loc_cfg = get_localization_config(cfg)

    yolo_path = args.yolo_model or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
    gbdt_dir = args.gbdt_models or loc_cfg.get("gbdt_models_dir", "models/")
    fusion_mode = args.fusion or loc_cfg.get("fusion_mode", "adaptive")
    confidence = args.conf if args.conf is not None else float(det_cfg.get("confidence", 0.75))

    src_dir = Path(args.src)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    image_paths = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg")))

    # Startup Banner
    log("Defense", "📍 Drone Localization — Detection + Geo-Regression")
    log("Defense", f"Config: {args.config or 'config/default.yaml'}")
    log("Defense", f"YOLO: {yolo_path} | GBDT: {gbdt_dir}")
    log("Defense", f"Input: {src_dir} ({len(image_paths)} images found)")
    log("Defense", f"Fusion mode: {fusion_mode}, Confidence: {confidence}")

    detector = YoloDetector(model_path=yolo_path, confidence=confidence, device=det_cfg.get("device", "cpu"))
    image_processor = ImageProcessor(det_cfg)
    localizer = GBDTRegressor(models_dir=gbdt_dir)
    localizer.load()
    fusion_engine = FusionEngine(
        mode=fusion_mode,
        iou_weight=loc_cfg.get("iou_weight", 0.60),
        shape_weight=loc_cfg.get("shape_weight", 0.30),
        motion_weight=loc_cfg.get("motion_weight", 0.10),
        soft_threshold=loc_cfg.get("soft_threshold", 0.25),
        area_ratio_min=loc_cfg.get("area_ratio_min", 0.00005),
        area_ratio_max=loc_cfg.get("area_ratio_max", 0.003),
        aspect_ratio_min=loc_cfg.get("aspect_ratio_min", 0.4),
        aspect_ratio_max=loc_cfg.get("aspect_ratio_max", 3.0),
    )

    start_time = time.time()
    records = []

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            log("Localize", f"❌ Failed to read {img_path.name}")
            continue

        _, seg_candidates = image_processor.process(img)
        clean_img = image_processor.clean_image(img, seg_candidates)
        yolo_dets = detector.detect(clean_img)

        ih, iw = img.shape[:2]
        yolo_dets = [
            d for d in yolo_dets
            if not (d["center_y"] < 120 and d["center_x"] > 0.60 * iw)
            and not (d["center_y"] > 0.88 * ih)
        ]

        fused = fusion_engine.fuse(seg_candidates, yolo_dets, image_shape=img.shape[:2])

        if fused:
            for det in fused:
                cx = det["center_x"]
                cy = det["center_y"]
                w = det["width"]
                h = det["height"]
                bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))
                feats = encode_features(bbox, img.shape[:2])
                pred = localizer.predict(feats)
                lat = float(pred["lat"])
                lon = float(pred["lon"])
                alt = float(pred["alt"])

                records.append({
                    "ImageName": img_path.name,
                    "Latitude": lat,
                    "Longitude": lon,
                    "Altitude": alt,
                })
                log("Localize", f"Image {idx}/{len(image_paths)}: {img_path.name} → 1 drone: lat={lat:.5f}, lon={lon:.5f}, alt={alt:.2f}m")
        else:
            log("Localize", f"Image {idx}/{len(image_paths)}: {img_path.name} → 0 drones detected")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=["ImageName", "Latitude", "Longitude", "Altitude"])
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start_time
    log("Defense", "✅ Localization complete!")
    log("Defense", f"Processed {len(image_paths)} images in {elapsed:.1f}s")
    log("Defense", f"Localized {len(records)} drones")
    log("Defense", f"Output: {out_path}")

if __name__ == "__main__":
    main()
