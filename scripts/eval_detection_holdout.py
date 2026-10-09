import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from defense.utils.logger import log


def _aabb(poly: np.ndarray) -> np.ndarray:
    """Axis-aligned [x1, y1, x2, y2] box around an OBB polygon (as in competition submissions)."""
    return np.array([poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _load_gt(label_path: str, w: int, h: int) -> list[np.ndarray]:
    """Read YOLO-OBB labels (class x1 y1 ... x4 y4, normalized) as pixel AABBs."""
    boxes = []
    for line in open(label_path).read().split("\n"):
        v = line.split()
        if len(v) == 9:
            boxes.append(_aabb(np.array(v[1:], dtype=float).reshape(4, 2) * [w, h]))
    return boxes


def competition_f1(preds: dict, label_dir: str, conf: float, iou_thr: float) -> dict:
    """Greedy one-to-one matching of AABBs at IoU >= iou_thr, highest confidence first."""
    tp = fp = fn = 0
    false_positives = []
    for img_path, r in preds.items():
        h, w = r.orig_shape
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gts = _load_gt(os.path.join(label_dir, stem + ".txt"), w, h)
        dets = []
        if r.obb is not None and len(r.obb):
            for poly, c in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                if c >= conf:
                    dets.append((float(c), _aabb(poly)))
        dets.sort(key=lambda d: -d[0])
        used = set()
        for c, d in dets:
            best, best_i = 0.0, -1
            for i, g in enumerate(gts):
                if i not in used:
                    v = _iou(d, g)
                    if v > best:
                        best, best_i = v, i
            if best >= iou_thr:
                tp += 1
                used.add(best_i)
            else:
                fp += 1
                false_positives.append({"image": stem, "center": [round(float((d[0] + d[2]) / 2), 1),
                                                                   round(float((d[1] + d[3]) / 2), 1)],
                                        "confidence": round(c, 3)})
        fn += len(gts) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "false_positives": false_positives}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO-OBB detector on a held-out YOLO-format split")
    parser.add_argument("--model", type=str, default="models/yolo11n-obb.pt", help="Path to YOLO model (.pt)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset dir containing data.yaml (e.g. drone.v5i)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (never used for training)")
    parser.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.50, 0.75], help="Confidence thresholds")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for the competition F1")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, etc.)")
    parser.add_argument("--output", type=str, default="data/detection_holdout_results.json", help="Results JSON path")
    args = parser.parse_args()

    model = YOLO(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("HoldoutEval", f"Model: {args.model} | split: {args.split} | dataset: {args.dataset}")
    val = model.val(data=os.path.join(args.dataset, "data.yaml"), split=args.split, imgsz=640,
                    device=args.device, project=str(out_path.parent / "holdout_val"), name=args.split,
                    exist_ok=True, verbose=False, plots=False)

    images = sorted(glob.glob(os.path.join(args.dataset, args.split, "images", "*")))
    label_dir = os.path.join(args.dataset, args.split, "labels")
    preds = {p: model.predict(p, imgsz=640, conf=min(args.confs), device=args.device, verbose=False)[0]
             for p in images}

    results = {
        "model": args.model,
        "split": args.split,
        "n_images": len(images),
        "ultralytics": {k: float(v) for k, v in val.results_dict.items()},
        "competition_f1": {str(c): competition_f1(preds, label_dir, c, args.iou) for c in args.confs},
    }
    for c, m in results["competition_f1"].items():
        log("HoldoutEval", f"conf {c}: F1 {m['f1']:.3f} (TP {m['tp']}, FP {m['fp']}, FN {m['fn']})")
    out_path.write_text(json.dumps(results, indent=2))
    log("HoldoutEval", f"✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
