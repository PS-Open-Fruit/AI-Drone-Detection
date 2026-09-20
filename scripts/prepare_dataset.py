import argparse
from pathlib import Path
import random
import cv2
import pandas as pd
import yaml
from defense.utils.video_io import open_video, get_video_info
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO Training Dataset from Video + CSV Labels")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Target YOLO dataset directory")
    parser.add_argument("--frame-interval", type=int, default=1, help="Interval between extracted frames")
    parser.add_argument("--val-split", type=float, default=0.20, help="Fraction for validation split")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_train_dir = out_dir / "images" / "train"
    img_val_dir = out_dir / "images" / "val"
    lbl_train_dir = out_dir / "labels" / "train"
    lbl_val_dir = out_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log("Dataset", "🚀 Preparing YOLO Dataset")
    log("Dataset", f"Video: {args.video}")
    log("Dataset", f"Labels: {args.labels}")
    log("Dataset", f"Output: {out_dir}")

    # Load labels
    df = pd.read_csv(args.labels)
    log("Dataset", f"Loaded {len(df)} annotations from CSV")

    # Helper to find column names
    cols = {c.lower(): c for c in df.columns}
    frame_col = cols.get("frame_id") or cols.get("frame") or cols.get("imagename") or cols.get("image_name")
    cx_col = cols.get("center_x") or cols.get("cx")
    cy_col = cols.get("center_y") or cols.get("cy")
    w_col = cols.get("width") or cols.get("w")
    h_col = cols.get("height") or cols.get("h")

    cap = open_video(args.video)
    info = get_video_info(cap)
    width = info["width"]
    height = info["height"]
    total_frames = info["frame_count"]

    frame_idx = 0
    extracted_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % args.frame_interval != 0:
            continue

        # Look up labels for this frame
        # Check integer frame number or frame filename
        matches = pd.DataFrame()
        if frame_col:
            matches = df[(df[frame_col] == frame_idx) |
                         (df[frame_col].astype(str) == f"{frame_idx:06d}.jpg") |
                         (df[frame_col].astype(str) == f"frame_{frame_idx:06d}.jpg")]

        extracted_records.append((frame_idx, frame, matches))

        if len(extracted_records) % 100 == 0:
            log("Dataset", f"  Extracted {len(extracted_records)} frames so far (frame {frame_idx}/{total_frames})")

    cap.release()

    random.seed(42)
    random.shuffle(extracted_records)

    n_val = int(len(extracted_records) * args.val_split)
    val_set = extracted_records[:n_val]
    train_set = extracted_records[n_val:]

    def save_split(records, img_dest, lbl_dest):
        count = 0
        for f_idx, img_mat, label_rows in records:
            base_name = f"frame_{f_idx:06d}"
            cv2.imwrite(str(img_dest / f"{base_name}.jpg"), img_mat)

            # Write YOLO labels: class_id cx cy w h (normalized [0, 1])
            txt_lines = []
            if len(label_rows) > 0 and cx_col and cy_col and w_col and h_col:
                for _, r in label_rows.iterrows():
                    cx = float(r[cx_col]) / width
                    cy = float(r[cy_col]) / height
                    w = float(r[w_col]) / width
                    h = float(r[h_col]) / height
                    txt_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            with open(lbl_dest / f"{base_name}.txt", "w", encoding="utf-8") as f:
                f.writelines(txt_lines)
            count += 1
        return count

    saved_train = save_split(train_set, img_train_dir, lbl_train_dir)
    saved_val = save_split(val_set, img_val_dir, lbl_val_dir)

    # Generate data.yaml
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "drone"},
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    log("Dataset", "✅ Dataset prepared!")
    log("Dataset", f"Train: {saved_train} images | Val: {saved_val} images")
    log("Dataset", f"Output: {out_dir / 'data.yaml'}")

if __name__ == "__main__":
    main()
