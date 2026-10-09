"""Held-out (cross-validated) localization score for the Problem 2 geo-regression pipeline.

Protocol (see --help for knobs):
  1. Box extraction = the shipped inference path of scripts/localize_drones.py:
     ImageProcessor.process -> clean_image -> YoloDetector.detect(clean) -> OSD/foliage
     filter -> FusionEngine.fuse (configured mode). Run once over every labeled image and
     cached to JSON, so CV folds never re-run YOLO.
  2. One box per image, chosen WITHOUT ground truth: highest fused confidence (ties -> first
     in fusion output order). The same rule is applied to training and test images.
  3. Features = encode_features(bbox, image_shape); model = GBDTRegressor (scaler + 3 GBRs)
     fitted on training-fold images only, never saved to disk.
  4. Folds are assigned over ALL labeled images (detected or not). Test images without a
     detection are reported separately and also scored with a GT-free fallback prediction
     (the training-fold mean lat/lon/alt), so they are never dropped silently.
  5. Splits: random KFold (shuffle) and contiguous index blocks; plus the in-sample
     (train-on-all / evaluate-on-all) score for contrast.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from defense.config import load_config, get_camera_config, get_detection_config, get_localization_config
from defense.detection import YoloDetector, ImageProcessor
from defense.localization import GBDTRegressor, FusionEngine, encode_features, evaluate_localization
from defense.utils.geometry import EARTH_RADIUS_M, haversine_distance, bearing
from defense.utils.logger import log

CACHE_VERSION = 1
DEG = math.pi / 180.0


# ----------------------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------------------
def _find_col(df, names):
    for name in names:
        for col in df.columns:
            if str(col).strip().lower() == name.lower():
                return col
    return None


def _image_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def load_ground_truth(data_dir: Path) -> list[dict]:
    """One record per image; `gt_rows` holds every coordinate row of its paired CSV."""
    image_paths = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpeg")),
                         key=lambda p: (_image_number(p), p.name))
    records = []
    for img_path in image_paths:
        csv_path = img_path.with_suffix(".csv")
        rows = []
        if csv_path.exists():
            df = pd.read_csv(csv_path, skipinitialspace=True)
            lat_c = _find_col(df, ["Latitude", "lat", "latitude"])
            lon_c = _find_col(df, ["Longitude", "lon", "longitude"])
            alt_c = _find_col(df, ["Altitude", "alt", "altitude", "altitude_msl"])
            if lat_c is not None and lon_c is not None and alt_c is not None:
                for _, r in df.dropna(subset=[lat_c, lon_c, alt_c]).iterrows():
                    rows.append({"lat": float(r[lat_c]), "lon": float(r[lon_c]), "alt": float(r[alt_c])})
        records.append({
            "image": img_path.name,
            "number": _image_number(img_path),
            "path": str(img_path),
            "csv_exists": csv_path.exists(),
            "gt_rows": rows,
        })
    return records


# ----------------------------------------------------------------------------------------
# Detection (localize_drones.py inference path) + cache
# ----------------------------------------------------------------------------------------
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fusion_params(loc_cfg: dict) -> dict:
    # Mirrors the FusionEngine construction in scripts/localize_drones.py
    return {
        "iou_weight": loc_cfg.get("iou_weight", 0.60),
        "shape_weight": loc_cfg.get("shape_weight", 0.30),
        "motion_weight": loc_cfg.get("motion_weight", 0.10),
        "soft_threshold": loc_cfg.get("soft_threshold", 0.25),
        "area_ratio_min": loc_cfg.get("area_ratio_min", 0.00005),
        "area_ratio_max": loc_cfg.get("area_ratio_max", 0.003),
        "aspect_ratio_min": loc_cfg.get("aspect_ratio_min", 0.4),
        "aspect_ratio_max": loc_cfg.get("aspect_ratio_max", 3.0),
    }


def _det_to_json(d: dict) -> dict:
    return {
        "center_x": float(d["center_x"]),
        "center_y": float(d["center_y"]),
        "width": float(d["width"]),
        "height": float(d["height"]),
        "confidence": float(d.get("confidence", 0.0)),
        "source": d.get("source", "seg"),
    }


def build_cache_meta(records, det_cfg, loc_cfg, yolo_path, confidence, fusion_mode) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "box_path": "scripts/localize_drones.py: ImageProcessor.process -> clean_image -> "
                    "YoloDetector.detect(clean) -> OSD/foliage filter -> FusionEngine.fuse",
        "yolo_model": str(yolo_path),
        "yolo_sha256": _file_sha256(yolo_path),
        "confidence": float(confidence),
        "device": det_cfg.get("device", "cpu"),
        "fusion_mode": fusion_mode,
        "fusion_params": _fusion_params(loc_cfg),
        "detection_config": {k: v for k, v in det_cfg.items()},
        "images": [r["image"] for r in records],
    }


def run_detection(records, det_cfg, loc_cfg, yolo_path, confidence, fusion_mode) -> dict:
    detector = YoloDetector(model_path=yolo_path, confidence=confidence, device=det_cfg.get("device", "cpu"))
    image_processor = ImageProcessor(det_cfg)
    fusion_engine = FusionEngine(mode=fusion_mode, **_fusion_params(loc_cfg))

    per_image = {}
    for i, rec in enumerate(records, 1):
        img = cv2.imread(rec["path"])
        if img is None:
            log("CV-Detect", f"❌ Failed to read {rec['image']}")
            per_image[rec["image"]] = {"read_ok": False}
            continue

        _, seg_candidates = image_processor.process(img)
        clean_img = image_processor.clean_image(img, seg_candidates)
        yolo_raw = detector.detect(clean_img)
        for d in yolo_raw:
            d["source"] = "yolo"

        ih, iw = img.shape[:2]
        # Same hard-coded OSD / foliage filter as scripts/localize_drones.py
        yolo_kept = [
            d for d in yolo_raw
            if not (d["center_y"] < 120 and d["center_x"] > 0.60 * iw)
            and not (d["center_y"] > 0.88 * ih)
        ]
        fused = fusion_engine.fuse(seg_candidates, yolo_kept, image_shape=img.shape[:2])

        per_image[rec["image"]] = {
            "read_ok": True,
            "shape": [int(ih), int(iw)],
            "seg_candidates": [[int(v) for v in c] for c in seg_candidates],
            "yolo_raw": [_det_to_json(d) for d in yolo_raw],
            "yolo_kept": [_det_to_json(d) for d in yolo_kept],
            "fused": [_det_to_json(d) for d in fused],  # seg-only boxes carry source="seg"
        }
        if i % 25 == 0 or i == len(records):
            log("CV-Detect", f"Processed {i}/{len(records)} images")
    return per_image


def load_or_build_detections(cache_path: Path, records, det_cfg, loc_cfg, yolo_path, confidence,
                             fusion_mode, redetect: bool) -> tuple[dict, dict, float]:
    meta = build_cache_meta(records, det_cfg, loc_cfg, yolo_path, confidence, fusion_mode)
    if cache_path.exists() and not redetect:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        old = dict(cached.get("meta", {}))
        old.pop("runtime_s", None)
        old.pop("created", None)
        if old == json.loads(json.dumps(meta)):
            log("CV-Detect", f"✅ Using cached detections: {cache_path}")
            return cached["meta"], cached["per_image"], 0.0
        log("CV-Detect", f"⚠️ Cache {cache_path} does not match current settings — re-running detection")

    log("CV-Detect", f"Running detection over {len(records)} images (this is the only YOLO pass)...")
    t0 = time.time()
    per_image = run_detection(records, det_cfg, loc_cfg, yolo_path, confidence, fusion_mode)
    runtime = time.time() - t0
    meta["runtime_s"] = runtime
    meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "per_image": per_image}, f)
    log("CV-Detect", f"✅ Detection done in {runtime:.1f}s, cached to {cache_path}")
    return meta, per_image, runtime


def select_box(fused: list[dict]):
    """GT-free selection rule: highest fused confidence (max() keeps the first on ties)."""
    if not fused:
        return None
    return max(fused, key=lambda d: d["confidence"])


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def per_sample_errors(pred: np.ndarray, gt: np.ndarray, cam: dict) -> dict:
    """pred/gt: (N, 3) arrays of lat, lon, alt. Returns per-sample error arrays."""
    clat, clon = cam["lat"], cam["lon"]
    ang, hgt, rng, lat_m, lon_m, horiz = [], [], [], [], [], []
    for (plat, plon, palt), (glat, glon, galt) in zip(pred, gt):
        d = abs(bearing(clat, clon, plat, plon) - bearing(clat, clon, glat, glon))
        ang.append(360.0 - d if d > 180.0 else d)
        hgt.append(abs(palt - galt))
        rng.append(abs(haversine_distance(clat, clon, plat, plon) - haversine_distance(clat, clon, glat, glon)))
        lat_m.append(abs(plat - glat) * DEG * EARTH_RADIUS_M)
        lon_m.append(abs(plon - glon) * DEG * EARTH_RADIUS_M * math.cos(glat * DEG))
        horiz.append(haversine_distance(glat, glon, plat, plon))
    return {k: np.asarray(v, dtype=np.float64) for k, v in
            [("angle", ang), ("height", hgt), ("range", rng), ("lat_m", lat_m), ("lon_m", lon_m), ("horiz", horiz)]}


def compute_metrics(pred: np.ndarray, gt: np.ndarray, cam: dict) -> dict:
    n = int(len(pred))
    if n == 0:
        return {"n": 0}
    ev = evaluate_localization(
        [{"lat": p[0], "lon": p[1], "alt": p[2]} for p in pred],
        [{"lat": g[0], "lon": g[1], "alt": g[2]} for g in gt],
        cam["lat"], cam["lon"], cam["alt"],
    )
    e = per_sample_errors(pred, gt, cam)
    own_total = 0.70 * e["angle"].mean() + 0.15 * e["height"].mean() + 0.15 * e["range"].mean()
    if not math.isclose(own_total, ev["total_weighted_error"], rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"evaluate_localization disagrees with independent check: "
                           f"{ev['total_weighted_error']} vs {own_total}")
    return {
        "n": n,
        "competition_score": ev["total_weighted_error"],
        "angle_mean_deg": ev["mean_angle_error"],
        "angle_median_deg": ev["median_angle_error"],
        "angle_p90_deg": float(np.percentile(e["angle"], 90)),
        "height_mean_m": ev["mean_height_error"],
        "height_median_m": ev["median_height_error"],
        "range_mean_m": ev["mean_range_error"],
        "range_median_m": ev["median_range_error"],
        "range_p90_m": float(np.percentile(e["range"], 90)),
        "lat_mae_m": float(e["lat_m"].mean()),
        "lon_mae_m": float(e["lon_m"].mean()),
        "alt_mae_m": float(e["height"].mean()),
        "horizontal_pos_err_mean_m": float(e["horiz"].mean()),
        "horizontal_pos_err_median_m": float(np.median(e["horiz"])),
    }


def _fold_summary(fold_metrics: list[dict], key: str) -> dict:
    vals = np.array([m[key] for m in fold_metrics if m.get("n", 0) > 0], dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "std_ddof1": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


# ----------------------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------------------
def fit_predict(X, Y, train_idx, test_idx, hp):
    reg = GBDTRegressor()  # save() is never called: fold models stay in memory only
    reg.train(X[train_idx], {"lat": Y[train_idx, 0], "lon": Y[train_idx, 1], "alt": Y[train_idx, 2]}, **hp)
    if len(test_idx) == 0:
        return np.zeros((0, 3))
    p = reg.predict(X[test_idx])
    return np.column_stack([p["lat"], p["lon"], p["alt"]])


def run_cv(name, folds, X, Y, has_box, hp, cam, meta_rows) -> dict:
    """folds: list of (train_positions, test_positions) over the evaluable-image arrays."""
    n = len(Y)
    oof = np.full((n, 3), np.nan)
    oof_fold = np.full(n, -1, dtype=int)
    is_fallback = np.zeros(n, dtype=bool)
    per_fold = []

    for k, (tr, te) in enumerate(folds):
        tr_det = tr[has_box[tr]]
        te_det = te[has_box[te]]
        te_miss = te[~has_box[te]]
        log("CV", f"[{name}] fold {k + 1}/{len(folds)}: train {len(tr)} imgs ({len(tr_det)} with box), "
                  f"test {len(te)} imgs ({len(te_det)} with box, {len(te_miss)} without)")
        oof[te_det] = fit_predict(X, Y, tr_det, te_det, hp)
        oof[te_miss] = Y[tr_det].mean(axis=0)  # GT-free fallback: training-fold mean coordinate
        is_fallback[te_miss] = True
        oof_fold[te] = k

        nums = [meta_rows[i]["number"] for i in te]
        per_fold.append({
            "fold": k + 1,
            "test_image_numbers": [int(min(nums)), int(max(nums))] if name.startswith("blocked") else None,
            "n_train_images": int(len(tr)),
            "n_train_fitted": int(len(tr_det)),
            "n_test_images": int(len(te)),
            "n_test_no_detection": int(len(te_miss)),
            "detected_only": compute_metrics(oof[te_det], Y[te_det], cam),
            "all_with_fallback": compute_metrics(oof[te], Y[te], cam),
        })

    det_mask = has_box & (oof_fold >= 0)
    all_mask = oof_fold >= 0
    pooled_det = compute_metrics(oof[det_mask], Y[det_mask], cam)
    pooled_all = compute_metrics(oof[all_mask], Y[all_mask], cam)
    summary = {}
    for key in ["competition_score", "angle_mean_deg", "height_mean_m", "range_mean_m",
                "lat_mae_m", "lon_mae_m", "alt_mae_m"]:
        summary[key] = _fold_summary([f["detected_only"] for f in per_fold], key)
    summary_fb = {"competition_score": _fold_summary([f["all_with_fallback"] for f in per_fold],
                                                     "competition_score")}
    return {
        "per_fold": per_fold,
        "fold_summary_detected_only": summary,
        "fold_summary_all_with_fallback": summary_fb,
        "pooled_detected_only": pooled_det,
        "pooled_all_with_fallback": pooled_all,
        "_oof": oof, "_oof_fold": oof_fold, "_is_fallback": is_fallback,
    }


def breakdown(oof, Y, mask_groups: dict, cam) -> dict:
    return {name: compute_metrics(oof[m], Y[m], cam) for name, m in mask_groups.items() if m.sum() > 0}


# ----------------------------------------------------------------------------------------
# Image-sequence correlation
# ----------------------------------------------------------------------------------------
def to_local_xyz(Y: np.ndarray, cam: dict) -> np.ndarray:
    east = (Y[:, 1] - cam["lon"]) * DEG * EARTH_RADIUS_M * math.cos(cam["lat"] * DEG)
    north = (Y[:, 0] - cam["lat"]) * DEG * EARTH_RADIUS_M
    return np.column_stack([east, north, Y[:, 2]])


def _pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pairwise_dist(P: np.ndarray) -> np.ndarray:
    return np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))


def nearest_train_stats(D: np.ndarray, folds) -> dict:
    """Distance (3D, m) from each test GT position to the closest GT position in its training fold."""
    out = np.full(len(D), np.nan)
    for tr, te in folds:
        out[te] = D[np.ix_(te, tr)].min(axis=1)
    return {"median_m": float(np.nanmedian(out)), "mean_m": float(np.nanmean(out)),
            "p90_m": float(np.nanpercentile(out, 90))}


def chunk_folds(n: int, chunk_size: int, n_folds: int, seed: int):
    """Shuffled KFold over contiguous chunks of `chunk_size` images (GroupKFold-style)."""
    groups = np.arange(n) // chunk_size
    ug = np.unique(groups)
    folds = []
    for gtr, gte in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(ug):
        folds.append((np.where(np.isin(groups, ug[gtr]))[0], np.where(np.isin(groups, ug[gte]))[0]))
    return folds, len(ug)


def sequence_analysis(Y, numbers, centers, cam, blocks, random_folds) -> dict:
    P = to_local_xyz(Y, cam)
    n = len(P)
    D = _pairwise_dist(P)
    iu = np.triu_indices(n, 1)
    all_pairs = D[iu]

    lag = {}
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        d = [D[i, i + k] for i in range(n - k) if numbers[i + k] - numbers[i] == k]
        if d:
            lag[str(k)] = {"n_pairs": len(d), "median_3d_dist_m": float(np.median(d)), "mean_3d_dist_m": float(np.mean(d))}

    step = np.array([D[i, i + 1] for i in range(n - 1) if numbers[i + 1] - numbers[i] == 1])
    med_step = float(np.median(step))
    jump_thresholds = {}
    for thr in [10.0, 20.0, 50.0]:
        jump_thresholds[f">{int(thr)}m"] = int((step > thr).sum())

    rng = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    autocorr = {
        "east": _pearson(P[:-1, 0], P[1:, 0]),
        "north": _pearson(P[:-1, 1], P[1:, 1]),
        "alt": _pearson(P[:-1, 2], P[1:, 2]),
        "range_from_camera": _pearson(rng[:-1], rng[1:]),
    }

    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    nn = Dn.argmin(axis=1)
    gap = np.abs(numbers[nn] - numbers)
    nn_stats = {
        "median_nn_3d_dist_m": float(np.median(Dn.min(axis=1))),
        "median_index_gap_to_nn": float(np.median(gap)),
        "frac_nn_within_1_index": float((gap <= 1).mean()),
        "frac_nn_within_2_index": float((gap <= 2).mean()),
        "frac_nn_within_5_index": float((gap <= 5).mean()),
        "frac_nn_within_10_index": float((gap <= 10).mean()),
    }

    # Trajectory segments: split wherever consecutive GT positions jump by more than a threshold
    segments = {}
    for thr in [10.0, 15.0]:
        cuts = [i for i in range(n - 1) if D[i, i + 1] > thr or numbers[i + 1] - numbers[i] != 1]
        bounds = [0] + [c + 1 for c in cuts] + [n]
        segments[f"jump>{int(thr)}m"] = [[int(numbers[a]), int(numbers[b - 1])] for a, b in zip(bounds[:-1], bounds[1:])]

    # Box-center pixel distance (feature side) for consecutive frames vs random pairs
    box_stats = None
    ok = ~np.isnan(centers[:, 0])
    cons = [np.hypot(*(centers[i + 1] - centers[i])) for i in range(n - 1)
            if ok[i] and ok[i + 1] and numbers[i + 1] - numbers[i] == 1]
    if cons:
        C = centers[ok]
        Dc = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(C), 1)]
        box_stats = {"consecutive_median_px": float(np.median(cons)), "random_pair_median_px": float(np.median(Dc)),
                     "n_consecutive_pairs": len(cons)}

    per_block = []
    for b, (_, te) in enumerate(blocks):
        per_block.append({
            "block": b + 1,
            "image_numbers": [int(numbers[te].min()), int(numbers[te].max())],
            "range_m": [float(rng[te].min()), float(np.median(rng[te])), float(rng[te].max())],
            "alt_m": [float(P[te, 2].min()), float(np.median(P[te, 2])), float(P[te, 2].max())],
            "bearing_deg": [float(v) for v in np.percentile(
                [bearing(cam["lat"], cam["lon"], la, lo) for la, lo in Y[te, :2]], [0, 50, 100])],
        })

    return {
        "n_images": int(n),
        "consecutive_step_3d_m": {"median": med_step, "mean": float(step.mean()), "p90": float(np.percentile(step, 90)),
                                  "max": float(step.max()), "n_steps": int(len(step)), "n_jumps": jump_thresholds},
        "random_pair_3d_m": {"median": float(np.median(all_pairs)), "mean": float(all_pairs.mean())},
        "lag_k_distance": lag,
        "lag1_autocorrelation": autocorr,
        "nearest_neighbour_in_gt_space": nn_stats,
        "trajectory_segments_[first,last]_image_number": segments,
        "test_to_nearest_train_position": {
            "random_kfold": nearest_train_stats(D, random_folds),
            "blocked_kfold": nearest_train_stats(D, blocks),
        },
        "box_center_similarity": box_stats,
        "per_block_gt_ranges_[min,median,max]": per_block,
    }


# ----------------------------------------------------------------------------------------
def _strip_private(d):
    if isinstance(d, dict):
        return {k: _strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [_strip_private(v) for v in d]
    return d


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main():
    parser = argparse.ArgumentParser(description="Cross-Validated Drone Geo-Localization Score (competition formula)")
    parser.add_argument("--data", type=str, default="data/p2_train", help="Directory of img_XXXX.jpg + img_XXXX.csv pairs")
    parser.add_argument("--output", type=str, default="data/cv_localization_results.json", help="Results JSON path")
    parser.add_argument("--cache", type=str, default="data/cv_localization_detections.json",
                        help="Detection cache JSON (created on first run)")
    parser.add_argument("--redetect", action="store_true", help="Ignore the cache and re-run detection")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    parser.add_argument("--fusion", type=str, default=None, help="Fusion mode: adaptive, filter, soft, hard, yolo, seg")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds / contiguous blocks")
    parser.add_argument("--seed", type=int, default=42, help="KFold shuffle random_state")
    parser.add_argument("--extra-seeds", type=int, default=10,
                        help="Also repeat random KFold with seeds 0..N-1 (pooled score sensitivity); 0 disables")
    parser.add_argument("--chunk-sizes", type=str, default="5,10,20,44",
                        help="Comma-separated chunk sizes for shuffled KFold over contiguous image chunks "
                             "(sequence-leakage sweep between random and blocked CV); empty disables")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()
    t_start = time.time()

    cfg = load_config(args.config)
    cam_cfg = get_camera_config(cfg)
    det_cfg = get_detection_config(cfg)
    loc_cfg = get_localization_config(cfg)
    cam = {"lat": float(cam_cfg["latitude"]), "lon": float(cam_cfg["longitude"]), "alt": float(cam_cfg["altitude"])}

    yolo_path = args.yolo_model or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
    confidence = args.conf if args.conf is not None else float(det_cfg.get("confidence", 0.50))
    fusion_mode = args.fusion or loc_cfg.get("fusion_mode", "adaptive")
    hp = {
        "n_estimators": int(loc_cfg.get("n_estimators", 200)),
        "learning_rate": float(loc_cfg.get("learning_rate", 0.1)),
        "max_depth": int(loc_cfg.get("max_depth", 5)),
        "min_samples_split": int(loc_cfg.get("min_samples_split", 5)),
        "min_samples_leaf": int(loc_cfg.get("min_samples_leaf", 2)),
        "subsample": float(loc_cfg.get("subsample", 0.9)),
    }

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    log("CV", "📊 Cross-Validated Drone Geo-Localization")
    log("CV", f"Config: {args.config or 'config/default.yaml'}")
    log("CV", f"Data: {data_dir} | YOLO: {yolo_path} (conf={confidence}) | Fusion: {fusion_mode}")
    log("CV", f"GBDT hyperparameters: {hp}")

    # 1. Ground truth -------------------------------------------------------------------
    records = load_ground_truth(data_dir)
    rows_hist = {}
    for r in records:
        rows_hist[len(r["gt_rows"])] = rows_hist.get(len(r["gt_rows"]), 0) + 1
    evaluable = [r for r in records if len(r["gt_rows"]) == 1]
    excluded = [{"image": r["image"], "n_gt_rows": len(r["gt_rows"]), "csv_exists": r["csv_exists"]}
                for r in records if len(r["gt_rows"]) != 1]
    log("CV", f"Images: {len(records)} | GT rows per image histogram: {rows_hist}")
    if excluded:
        log("CV", f"⚠️ Excluding {len(excluded)} images whose CSV does not hold exactly one drone row: "
                  f"{[e['image'] for e in excluded]}")

    # 2. Detection (single pass, cached) -------------------------------------------------
    det_meta, per_image, det_runtime = load_or_build_detections(
        Path(args.cache), records, det_cfg, loc_cfg, yolo_path, confidence, fusion_mode, args.redetect)

    n = len(evaluable)
    Y = np.array([[r["gt_rows"][0]["lat"], r["gt_rows"][0]["lon"], r["gt_rows"][0]["alt"]] for r in evaluable])
    numbers = np.array([r["number"] for r in evaluable])
    X = np.zeros((n, 20))
    has_box = np.zeros(n, dtype=bool)
    centers = np.full((n, 2), np.nan)
    box_source = np.array(["none"] * n, dtype=object)
    n_fused = np.zeros(n, dtype=int)
    n_yolo_kept = np.zeros(n, dtype=int)
    chosen = [None] * n
    for i, r in enumerate(evaluable):
        info = per_image.get(r["image"], {"read_ok": False})
        if not info.get("read_ok"):
            continue
        fused = info["fused"]
        n_fused[i] = len(fused)
        n_yolo_kept[i] = len(info["yolo_kept"])
        det = select_box(fused)
        if det is None:
            continue
        cx, cy, w, h = det["center_x"], det["center_y"], det["width"], det["height"]
        bbox = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))
        X[i] = encode_features(bbox, tuple(info["shape"]))
        has_box[i] = True
        centers[i] = [cx, cy]
        box_source[i] = det["source"]
        chosen[i] = {"bbox_xywh": list(bbox), "confidence": det["confidence"], "source": det["source"]}

    read_fail = sum(1 for r in evaluable if not per_image.get(r["image"], {}).get("read_ok"))
    coverage = {
        "n_labeled_images": len(records),
        "n_evaluable_images_one_gt_row": n,
        "n_excluded_images": len(excluded),
        "excluded_images": excluded,
        "n_read_failures": int(read_fail),
        "n_with_ge1_yolo_raw": int(sum(1 for r in evaluable if per_image.get(r["image"], {}).get("yolo_raw"))),
        "n_with_ge1_yolo_after_osd_foliage_filter": int((n_yolo_kept >= 1).sum()),
        "n_with_ge1_fused_detection": int(has_box.sum()),
        "n_with_multiple_fused_detections": int((n_fused >= 2).sum()),
        "fused_detection_count_histogram": {str(k): int((n_fused == k).sum()) for k in sorted(set(n_fused.tolist()))},
        "n_with_multiple_yolo_after_filter": int((n_yolo_kept >= 2).sum()),
        "chosen_box_source": {"yolo": int((box_source == "yolo").sum()), "seg_only_morphology": int((box_source == "seg").sum())},
        "n_no_detection": int((~has_box).sum()),
        "no_detection_images": [evaluable[i]["image"] for i in np.where(~has_box)[0]],
        "selection_rule": "highest fused confidence among FusionEngine outputs (ties -> first in fusion order); "
                          "applied identically to training and test images; no ground truth used",
    }
    log("CV", f"Coverage: {coverage['n_with_ge1_fused_detection']}/{n} images with >=1 fused detection, "
              f"{coverage['n_with_multiple_fused_detections']} with multiple, "
              f"{coverage['n_no_detection']} with none; chosen sources {coverage['chosen_box_source']}")

    # 3. Splits -------------------------------------------------------------------------
    pos = np.arange(n)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    random_folds = [(tr, te) for tr, te in kf.split(pos)]
    blocks_te = np.array_split(pos, args.folds)  # evaluable images are sorted by image number
    blocked_folds = [(np.setdiff1d(pos, te), te) for te in blocks_te]

    t_cv = time.time()
    res_random = run_cv(f"random{args.folds}fold", random_folds, X, Y, has_box, hp, cam, evaluable)
    res_blocked = run_cv(f"blocked{args.folds}fold", blocked_folds, X, Y, has_box, hp, cam, evaluable)

    # 4. In-sample (train on all, evaluate on all) ---------------------------------------
    det_pos = pos[has_box]
    ins_pred = np.full((n, 3), np.nan)
    ins_pred[det_pos] = fit_predict(X, Y, det_pos, det_pos, hp)
    ins_pred[~has_box] = Y[det_pos].mean(axis=0)
    in_sample = {
        "detected_only": compute_metrics(ins_pred[has_box], Y[has_box], cam),
        "all_with_fallback": compute_metrics(ins_pred, Y, cam),
    }

    # 5. Seed sensitivity for random KFold ----------------------------------------------
    seed_sens = None
    if args.extra_seeds > 0:
        scores = []
        for s in range(args.extra_seeds):
            folds_s = list(KFold(n_splits=args.folds, shuffle=True, random_state=s).split(pos))
            oof_s = np.full((n, 3), np.nan)
            for tr, te in folds_s:
                tr_det, te_det = tr[has_box[tr]], te[has_box[te]]
                oof_s[te_det] = fit_predict(X, Y, tr_det, te_det, hp)
            m = compute_metrics(oof_s[has_box], Y[has_box], cam)
            scores.append({"seed": s, "pooled_competition_score": m["competition_score"],
                           "angle_mean_deg": m["angle_mean_deg"], "height_mean_m": m["height_mean_m"],
                           "range_mean_m": m["range_mean_m"]})
        v = np.array([x["pooled_competition_score"] for x in scores])
        seed_sens = {"seeds": scores, "mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "min": float(v.min()), "max": float(v.max())}

    # 5b. Sequence-leakage sweep: shuffled KFold over contiguous chunks of k images ------
    D_gt = _pairwise_dist(to_local_xyz(Y, cam))
    chunk_sweep = [{
        "chunk_size": 1, "n_chunks": n, "note": "identical to random_kfold",
        "fold_mean_competition_score": res_random["fold_summary_detected_only"]["competition_score"],
        "pooled_detected_only": res_random["pooled_detected_only"],
        "pooled_all_with_fallback_score": res_random["pooled_all_with_fallback"]["competition_score"],
        "test_to_nearest_train_position": nearest_train_stats(D_gt, random_folds),
    }]
    for cs in [int(c) for c in args.chunk_sizes.split(",") if c.strip()]:
        folds_c, n_chunks = chunk_folds(n, cs, args.folds, args.seed)
        res_c = run_cv(f"chunk{cs}", folds_c, X, Y, has_box, hp, cam, evaluable)
        chunk_sweep.append({
            "chunk_size": cs, "n_chunks": n_chunks,
            "fold_mean_competition_score": res_c["fold_summary_detected_only"]["competition_score"],
            "pooled_detected_only": res_c["pooled_detected_only"],
            "pooled_all_with_fallback_score": res_c["pooled_all_with_fallback"]["competition_score"],
            "test_to_nearest_train_position": nearest_train_stats(D_gt, folds_c),
        })
    chunk_sweep.append({
        "chunk_size": "blocked", "n_chunks": args.folds, "note": "identical to blocked_kfold",
        "fold_mean_competition_score": res_blocked["fold_summary_detected_only"]["competition_score"],
        "pooled_detected_only": res_blocked["pooled_detected_only"],
        "pooled_all_with_fallback_score": res_blocked["pooled_all_with_fallback"]["competition_score"],
        "test_to_nearest_train_position": nearest_train_stats(D_gt, blocked_folds),
    })
    cv_runtime = time.time() - t_cv

    # 6. Diagnostics --------------------------------------------------------------------
    groups = {
        "chosen_box_yolo": has_box & (box_source == "yolo"),
        "chosen_box_seg_only": has_box & (box_source == "seg"),
        "single_fused_detection": has_box & (n_fused == 1),
        "multiple_fused_detections": has_box & (n_fused >= 2),
    }
    diagnostics = {
        "random_kfold_pooled_breakdown": breakdown(res_random["_oof"], Y, groups, cam),
        "blocked_kfold_pooled_breakdown": breakdown(res_blocked["_oof"], Y, groups, cam),
    }
    seq = sequence_analysis(Y, numbers, centers, cam, blocked_folds, random_folds)

    # 7. Per-image out-of-fold records ---------------------------------------------------
    per_image_out = []
    e_r = per_sample_errors(res_random["_oof"], Y, cam)
    e_b = per_sample_errors(res_blocked["_oof"], Y, cam)
    for i, r in enumerate(evaluable):
        per_image_out.append({
            "image": r["image"],
            "gt": Y[i].tolist(),
            "n_fused": int(n_fused[i]),
            "chosen_box": chosen[i],
            "random_fold": int(res_random["_oof_fold"][i]) + 1,
            "random_pred": res_random["_oof"][i].tolist(),
            "random_err": {"angle_deg": float(e_r["angle"][i]), "height_m": float(e_r["height"][i]), "range_m": float(e_r["range"][i])},
            "blocked_fold": int(res_blocked["_oof_fold"][i]) + 1,
            "blocked_pred": res_blocked["_oof"][i].tolist(),
            "blocked_err": {"angle_deg": float(e_b["angle"][i]), "height_m": float(e_b["height"][i]), "range_m": float(e_b["range"][i])},
            "prediction_is_fallback_no_detection": bool(not has_box[i]),
        })

    total_runtime = time.time() - t_start
    results = {
        "description": "Held-out localization score (TESA 2025 Problem 2 competition formula: "
                       "0.70*mean angle err [deg] + 0.15*mean height err [m] + 0.15*mean range err [m])",
        "command": "uv run python " + " ".join(sys.argv),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": {"total": total_runtime, "detection": det_runtime, "cv_and_fitting": cv_runtime},
        "protocol": {
            "ground_truth": f"{data_dir}/img_XXXX.csv (one row per image)",
            "box_extraction": det_meta.get("box_path"),
            "yolo_model": det_meta.get("yolo_model"),
            "yolo_sha256": det_meta.get("yolo_sha256"),
            "yolo_confidence": det_meta.get("confidence"),
            "fusion_mode": det_meta.get("fusion_mode"),
            "selection_rule": coverage["selection_rule"],
            "features": "defense.localization.encode_features(bbox_xywh, image_shape) -> 20 dims",
            "model": "defense.localization.GBDTRegressor (StandardScaler + 3 GradientBoostingRegressor, random_state=42), "
                     "fitted on training-fold images with a detection only",
            "gbdt_hyperparameters": hp,
            "camera": cam,
            "random_kfold": f"sklearn KFold(n_splits={args.folds}, shuffle=True, random_state={args.seed}) over evaluable images "
                            "(one sample per image, so folds are grouped by image)",
            "blocked_kfold": f"{args.folds} contiguous blocks of image number (np.array_split over sorted images)",
            "no_detection_handling": "folds assigned over all evaluable images; test images without a detection are excluded "
                                     "from 'detected_only' metrics and scored in 'all_with_fallback' with the training-fold "
                                     "mean (lat, lon, alt) as a GT-free fallback prediction",
            "fold_std": "sample std (ddof=1) across folds",
            "pooled": "metrics over the concatenated out-of-fold predictions",
            "lat_lon_mae_m": "|dlat|*R and |dlon|*R*cos(lat_gt), R=6371000 m (same R as haversine)",
        },
        "coverage": coverage,
        "random_kfold": _strip_private(res_random),
        "blocked_kfold": _strip_private(res_blocked),
        "in_sample": in_sample,
        "random_kfold_seed_sensitivity": seed_sens,
        "chunked_kfold_sweep": chunk_sweep,
        "diagnostics": diagnostics,
        "sequence_correlation": seq,
        "per_image": per_image_out,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2)

    def line(tag, m):
        return (f"{tag:<34} n={m['n']:>3}  score={m['competition_score']:.4f}  "
                f"angle={m['angle_mean_deg']:.3f}/{m['angle_median_deg']:.3f}°  "
                f"height={m['height_mean_m']:.3f}/{m['height_median_m']:.3f}m  "
                f"range={m['range_mean_m']:.3f}/{m['range_median_m']:.3f}m  "
                f"MAE lat/lon/alt={m['lat_mae_m']:.2f}/{m['lon_mae_m']:.2f}/{m['alt_mae_m']:.2f}m")

    log("CV", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log("CV", "Results (mean/median):")
    for tag, res in [("random", res_random), ("blocked", res_blocked)]:
        fs = res["fold_summary_detected_only"]["competition_score"]
        log("CV", f"  {tag} folds: score {fs['mean']:.4f} ± {fs['std_ddof1']:.4f} "
                  f"(per fold: {[round(f['detected_only']['competition_score'], 4) for f in res['per_fold']]})")
        log("CV", "  " + line(f"{tag} pooled (detected only)", res["pooled_detected_only"]))
        log("CV", "  " + line(f"{tag} pooled (all, fallback)", res["pooled_all_with_fallback"]))
    log("CV", "  " + line("in-sample (detected only)", in_sample["detected_only"]))
    if seed_sens:
        log("CV", f"  random KFold seeds 0..{args.extra_seeds - 1}: pooled score mean {seed_sens['mean']:.4f}, "
                  f"range [{seed_sens['min']:.4f}, {seed_sens['max']:.4f}]")
    for c in chunk_sweep:
        fs = c["fold_mean_competition_score"]
        log("CV", f"  chunk={str(c['chunk_size']):>7} ({c['n_chunks']:>3} chunks): pooled score "
                  f"{c['pooled_detected_only']['competition_score']:.4f} | folds {fs['mean']:.4f} ± {fs['std_ddof1']:.4f} | "
                  f"test→nearest train GT median {c['test_to_nearest_train_position']['median_m']:.2f} m")
    log("CV", f"✅ Results written to {out_path} (total {total_runtime:.1f}s, detection {det_runtime:.1f}s)")


if __name__ == "__main__":
    main()
