import argparse
from pathlib import Path
import pandas as pd
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Format Output CSVs into Competition Submission Files")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file from detection or localization")
    parser.add_argument("--output", type=str, required=True, help="Path for formatted submission CSV")
    parser.add_argument("--problem", type=int, choices=[1, 2], required=True,
                        help="Competition problem number: 1 (Detection) or 2 (Localization)")

    args = parser.parse_args()

    in_file = Path(args.input)
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    log("Submission", f"Reading raw results from: {in_file}")
    df = pd.read_csv(in_file)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.problem == 1:
        # Problem 1: image_name, center_x, center_y, width, height
        expected_cols = ["image_name", "center_x", "center_y", "width", "height"]
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "_")
            if cl in ["image_name", "imagename", "image"]:
                col_map[c] = "image_name"
            elif cl in ["center_x", "cx", "x"]:
                col_map[c] = "center_x"
            elif cl in ["center_y", "cy", "y"]:
                col_map[c] = "center_y"
            elif cl in ["width", "w"]:
                col_map[c] = "width"
            elif cl in ["height", "h"]:
                col_map[c] = "height"
        formatted = df.rename(columns=col_map)
        missing = [c for c in expected_cols if c not in formatted.columns]
        if missing:
            raise ValueError(f"Missing required columns for Problem 1: {missing}")
        res = formatted[expected_cols]
        res.to_csv(out_file, index=False)
        log("Submission", f"✅ Formatted Problem 1 submission: {len(res)} rows written to {out_file}")

    elif args.problem == 2:
        # Problem 2: ImageName, Latitude, Longitude, Altitude
        expected_cols = ["ImageName", "Latitude", "Longitude", "Altitude"]
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "_")
            if cl in ["imagename", "image_name", "image"]:
                col_map[c] = "ImageName"
            elif cl in ["latitude", "lat"]:
                col_map[c] = "Latitude"
            elif cl in ["longitude", "lon"]:
                col_map[c] = "Longitude"
            elif cl in ["altitude", "alt"]:
                col_map[c] = "Altitude"
        formatted = df.rename(columns=col_map)
        missing = [c for c in expected_cols if c not in formatted.columns]
        if missing:
            raise ValueError(f"Missing required columns for Problem 2: {missing}")
        res = formatted[expected_cols]
        res.to_csv(out_file, index=False)
        log("Submission", f"✅ Formatted Problem 2 submission: {len(res)} rows written to {out_file}")

if __name__ == "__main__":
    main()
