import argparse
from pathlib import Path
from defense.detection import YoloDetector, tune_confidence
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Tune YOLO Confidence Threshold")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground truth CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path for results")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, etc.)")

    args = parser.parse_args()

    log("ConfTuner", "🔍 Starting Confidence Threshold Optimization")
    log("ConfTuner", f"Model: {args.model}")
    log("ConfTuner", f"Test Images: {args.test_dir}")
    log("ConfTuner", f"Ground Truth: {args.ground_truth}")

    detector = YoloDetector(model_path=args.model, confidence=0.5, device=args.device)
    df_results = tune_confidence(
        detector=detector,
        image_dir=args.test_dir,
        ground_truth_csv=args.ground_truth,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    log("ConfTuner", f"✅ Results saved to {out_path}")

if __name__ == "__main__":
    main()
