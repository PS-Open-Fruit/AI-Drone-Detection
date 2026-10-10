from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from defense.localization.feature_encoder import encode_features
from defense.utils.logger import log

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _find_col(df: pd.DataFrame, names: list[str]):
    for name in names:
        for col in df.columns:
            if str(col).strip().lower() == name.lower():
                return col
    return None


def _image_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def _rows(df: pd.DataFrame) -> list[dict]:
    lat_c = _find_col(df, ["Latitude", "lat", "latitude"])
    lon_c = _find_col(df, ["Longitude", "lon", "longitude"])
    alt_c = _find_col(df, ["Altitude", "alt", "altitude", "altitude_msl"])
    if lat_c is None or lon_c is None or alt_c is None:
        return []
    return [{"lat": float(r[lat_c]), "lon": float(r[lon_c]), "alt": float(r[alt_c])}
            for _, r in df.dropna(subset=[lat_c, lon_c, alt_c]).iterrows()]


def load_labeled_images(data_path, image_dir=None) -> list[dict]:
    """Load ground-truth drone coordinates for localization training / evaluation.

    Supported layouts:
      * competition format: a directory of `img_XXXX.jpg` + `img_XXXX.csv` pairs (Latitude, Longitude, Altitude)
      * a single CSV with an image-name column plus Latitude/Longitude/Altitude (images in `image_dir`
        or next to the CSV)
    Returns one record per image, sorted by image number: {image, number, path, gt_rows}.
    """
    data_path = Path(data_path)
    records = []
    if data_path.is_dir():
        images = sorted((p for p in data_path.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                        key=lambda p: (_image_number(p), p.name))
        for img in images:
            csv_path = img.with_suffix(".csv")
            rows = _rows(pd.read_csv(csv_path, skipinitialspace=True)) if csv_path.exists() else []
            records.append({"image": img.name, "number": _image_number(img), "path": str(img), "gt_rows": rows})
        return records

    df = pd.read_csv(data_path, skipinitialspace=True)
    img_col = _find_col(df, ["ImageName", "image_name", "filename", "file", "image"])
    if img_col is None:
        raise ValueError(f"{data_path} has no image-name column; pass the image/CSV pair directory instead")
    img_dir = Path(image_dir) if image_dir else data_path.parent
    for name, group in df.groupby(img_col, sort=False):
        img = img_dir / str(name)
        records.append({"image": img.name, "number": _image_number(img), "path": str(img), "gt_rows": _rows(group)})
    return sorted(records, key=lambda r: (r["number"], r["image"]))


def extract_samples(records: list[dict], detector, fusion_engine) -> dict:
    """Run the shared inference box path (DroneDetector.best_box) on every labeled image.

    Training and inference must see the same kind of box, so this is the only place training boxes come from.
    Images without exactly one ground-truth row are skipped (logged); images without a detection keep
    has_box=False so evaluations can count them instead of silently dropping them.
    """
    X, Y, has_box, sources, kept = [], [], [], [], []
    for i, rec in enumerate(records, 1):
        if len(rec["gt_rows"]) != 1:
            log("Dataset", f"⚠️ Skipping {rec['image']}: {len(rec['gt_rows'])} ground-truth rows (expected 1)")
            continue
        img = cv2.imread(rec["path"])
        if img is None:
            log("Dataset", f"❌ Failed to read {rec['image']}")
            continue
        bbox, source, _ = detector.best_box(detector.detect(img), fusion_engine, img.shape[:2])
        gt = rec["gt_rows"][0]
        X.append(encode_features(bbox, img.shape[:2]) if bbox is not None else np.zeros(20))
        Y.append([gt["lat"], gt["lon"], gt["alt"]])
        has_box.append(bbox is not None)
        sources.append(source)
        kept.append(rec)
        if i % 50 == 0 or i == len(records):
            log("Dataset", f"Extracted boxes for {i}/{len(records)} images")
    return {"X": np.array(X, dtype=float).reshape(-1, 20), "Y": np.array(Y, dtype=float).reshape(-1, 3),
            "has_box": np.array(has_box, dtype=bool), "sources": sources, "records": kept}
