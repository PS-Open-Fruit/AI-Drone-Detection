import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from defense.utils.logger import log

def tune_confidence(detector, image_dir: str, ground_truth_csv: str,
                    thresholds: list[float] = None) -> pd.DataFrame:
    """
    Sweep confidence thresholds and measure detection accuracy.

    Args:
        detector: YoloDetector or SahiDetector instance
        image_dir: Directory of test images
        ground_truth_csv: CSV with columns including image name/file and bounding boxes
        thresholds: List of thresholds to try. Defaults to [0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

    Returns:
        DataFrame with columns: threshold, accuracy_pct, correct, total, rmse, false_positives, false_negatives
    """
    if thresholds is None:
        thresholds = [0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

    img_dir = Path(image_dir)
    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Load ground truth CSV
    gt_df = pd.read_csv(ground_truth_csv)
    # Find image column name (image_name, ImageName, filename, etc.)
    img_col = None
    for col in ["image_name", "ImageName", "filename", "image", "file"]:
        if col in gt_df.columns:
            img_col = col
            break
    if img_col is None:
        img_col = gt_df.columns[0]

    # Ground truth count per image
    gt_counts = {}
    for p in image_paths:
        name = p.name
        # Count rows matching either filename or stem
        matches = gt_df[gt_df[img_col].astype(str) == name]
        if len(matches) == 0:
            matches = gt_df[gt_df[img_col].astype(str) == p.stem]
        gt_counts[p] = len(matches)

    # Pre-load images in memory or read on demand
    images = {}
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images[p] = img

    results = []
    best_row = None
    best_acc = -1.0

    for thresh in thresholds:
        log("ConfTuner", f"Testing threshold {thresh:.2f} ...")
        detector.confidence = float(thresh)

        correct = 0
        total = len(images)
        sq_err_sum = 0.0
        false_positives = 0
        false_negatives = 0

        for p, img in images.items():
            dets = detector.detect(img)
            pred_count = len(dets)
            gt_count = gt_counts.get(p, 0)

            diff = pred_count - gt_count
            sq_err_sum += diff * diff

            if diff > 0:
                false_positives += diff
            elif diff < 0:
                false_negatives += -diff

            if pred_count == gt_count:
                correct += 1

        acc_pct = (correct / total * 100.0) if total > 0 else 0.0
        rmse = math.sqrt(sq_err_sum / total) if total > 0 else 0.0

        row = {
            "threshold": thresh,
            "accuracy_pct": acc_pct,
            "correct": correct,
            "total": total,
            "rmse": rmse,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        results.append(row)
        log("ConfTuner", f"  Accuracy: {acc_pct:.2f}% | RMSE: {rmse:.3f} | FP: {false_positives} | FN: {false_negatives}")

        if acc_pct > best_acc or (acc_pct == best_acc and (best_row is None or rmse < best_row["rmse"])):
            best_acc = acc_pct
            best_row = row

    if best_row is not None:
        log("ConfTuner", f"✅ Optimal threshold: {best_row['threshold']:.2f} (Accuracy: {best_row['accuracy_pct']:.2f}%, RMSE: {best_row['rmse']:.3f})")

    return pd.DataFrame(results)
