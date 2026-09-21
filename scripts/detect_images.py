import argparse
from pathlib import Path
import time
import pandas as pd
import cv2
from defense.config import load_config, get_detection_config
from defense.detection import YoloDetector, SahiDetector, ImageProcessor
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Drone Detection — Batch Image Mode")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model (.pt)")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--slice", type=int, default=None, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=None, help="SAHI overlap ratio")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu, cuda, etc.)")
    parser.add_argument("--use-sahi", action="store_true", help="Enable SAHI sliced inference")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    cfg = load_config(args.config)
    det_cfg = get_detection_config(cfg)

    model_path = args.model or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
    confidence = args.conf if args.conf is not None else float(det_cfg.get("confidence", 0.50))
    device = args.device or det_cfg.get("device", "cpu")

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {test_dir}")

    image_paths = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")))

    # Startup Banner
    log("Defense", "🔍 Drone Detection — Batch Image Mode")
    log("Defense", f"Config: {args.config or 'config/default.yaml'}")
    log("Defense", f"Model: {model_path}")
    log("Defense", f"Input: {test_dir} ({len(image_paths)} images found)")
    log("Defense", f"Confidence: {confidence}, Device: {device}, Mode: {'SAHI' if args.use_sahi else 'Standard YOLO'}")

    image_processor = ImageProcessor(det_cfg)

    if args.use_sahi:
        slice_size = args.slice or det_cfg.get("slice_size", 640)
        overlap = args.overlap if args.overlap is not None else det_cfg.get("overlap_ratio", 0.25)
        detector = SahiDetector(
            model_path=model_path,
            confidence=confidence,
            device=device,
            slice_size=slice_size,
            overlap_ratio=overlap,
            nms_threshold=det_cfg.get("nms_threshold", 0.60)
        )
    else:
        detector = YoloDetector(model_path=model_path, confidence=confidence, device=device)

    start_time = time.time()
    records = []

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            log("Detection", f"❌ Failed to process {img_path.name}: image could not be loaded")
            continue

        mask, cands = image_processor.process(img)
        clean_img = image_processor.clean_image(img, cands)
        dets = detector.detect(clean_img)

        # Spatial filter: exclude detections in camera timestamp and foliage exclusion zones
        ih, iw = img.shape[:2]
        dets = [
            d for d in dets
            if not (d["center_y"] < 120 and d["center_x"] > 0.60 * iw)
            and not (d["center_y"] > 0.88 * ih)
        ]

        log("Detection", f"Processing image {idx}/{len(image_paths)}: {img_path.name} → {len(dets)} drones")

        for d in dets:
            records.append({
                "image_name": img_path.name,
                "center_x": d["center_x"],
                "center_y": d["center_y"],
                "width": d["width"],
                "height": d["height"],
            })

    # Save to CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=["image_name", "center_x", "center_y", "width", "height"])
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start_time
    avg_per_img = len(records) / max(1, len(image_paths))
    log("Defense", f"✅ Done! {len(image_paths)} images processed in {elapsed:.1f}s")
    log("Defense", f"Total detections: {len(records)} | Avg per image: {avg_per_img:.2f}")
    log("Defense", f"Output: {out_path}")

if __name__ == "__main__":
    main()
