# Agent Guide

Drone detection, geolocation and tracking for the TESA Top Gun Rally 2025 (Defensive Track), targeting a CPU-only Raspberry Pi 5. This repo is a tested reimplementation of the original competition code; it is cited on the maintainer's resume, so **every number in README.md and docs/ must be reproducible**.

## Commands

```bash
uv sync --extra dev                  # add --extra edge for the NCNN runtime
uv run pytest tests/ -q              # 32 tests, no data or models needed
uv run python scripts/track_video.py <video.mp4> -o out.mp4
uv run python scripts/train_gbdt.py --data data/p2_train          # writes models/geo_localizer.bin
```

Evaluation (all need local data, see below):

```bash
uv run python scripts/eval_detection_holdout.py --dataset data/v5 --full-frames data/p1_test \
    --counts data/p1_drone_gt.csv --confs 0.4 0.5                  # Problem 1, held out
uv run python scripts/cv_localization.py --redetect               # Problem 2, blocked + random CV
uv run python scripts/benchmark_pipeline.py data/p3_video/P1_VIDEO_5.mp4 --repeats 2   # speed per stage
```

## Architecture

- `defense/detection/drone_detector.py`: **the single detection path** for every script.
  - Morphology proposes candidates; YOLO11n-OBB runs on 320 px raw-frame crops around them.
  - A YOLO box counts only if a candidate confirms it.
  - `best_box()` picks the one box per image that localization uses.
- `defense/localization/geo_localizer.py`: GBDTs predict bearing, log distance and height relative to the camera, then convert to lat/lon/alt. `dataset.py` builds training samples through `DroneDetector.best_box`.
- `defense/tracking/tracker.py`: greedy association. With YOLO loaded, morphology-only detections can extend a track but not start one.
- `config/default.yaml`: all tunable values. Camera position is fixed there.
- `gbdt_regressor.py` and `ImageProcessor.clean_image` are the previous approach, kept for reference; nothing on the main path uses them.

## Local data and models (gitignored, not in the repo)

| Path | Contents |
| :--- | :--- |
| `models/yolo11n-obb.pt` | Detector (= `tesa2025_defense` run `best.pt` in the archive) |
| `models/geo_localizer.bin` | Localizer; rebuild with `train_gbdt.py` |
| `models/yolo11n-obb_ncnn_model/` | NCNN export for the Pi (`export(format='ncnn', imgsz=320)`) |
| `data/p1_test/`, `data/p1_drone_gt.csv` | Problem 1 frames (1920×1080) and per-image drone counts |
| `data/v5/` | `drone.v5i` dataset; its test/valid images are unscaled 460 px crops of P1 frames at x=730, y=310 |
| `data/p2_train/` | 438 `img_XXXX.jpg` + `img_XXXX.csv` localization labels |
| `data/p3_video/` | Competition videos (`P1_VIDEO_5`: 2 drones, `P1_VIDEO_4`: 1 drone) |
| `TESA 2025 Reorganize/Archived Project/*.zip` | Original competition code, runs and datasets |

## Rules for results and claims

- **Never copy numbers from the archive's markdown docs** (`TESA 2025 Reorganize/Defense_Challenge_Documentation/`): some are wrong. Recompute from raw logs or re-run the scripts.
- **Localization headline = blocked CV.** The 438 images are consecutive frames of 4–5 flights, so random splits leak near-duplicate neighbours and look about 2× better than reality.
- **Label every metric** as held out or in-sample, and state what it was measured on (images, frames, hardware). Desktop speed is not Pi speed.
- **Re-measuring "after" numbers:** update the commit named in README's "Where these numbers come from" note.
- **Tuning:** if you tune on an evaluation set, say so. `docs/02` §5.5 also reports nested CV for this reason.

## Gotchas

- **Windows + Git Bash:** use `tar --force-local` for paths containing `C:`; set `MSYS_NO_PATHCONV=1` before passing remote `/home/...` paths; `core.autocrlf=true`, so LF/CRLF warnings are expected.
- **Raspberry Pi (aarch64):** `pyproject.toml` pins torch/torchvision to the PyTorch CPU index; PyPI's ARM torch drags in about 2.5 GB of CUDA libraries. Use the NCNN model on the Pi: YOLO takes ~16 ms per crop against ~105 ms with PyTorch. Measured results and setup are in `docs/04_edge_deployment.md` §1.3.
- **Benchmarks:** run on an idle machine, one heavy job at a time; each run starts a fresh process.
- **Training boxes:** never add a second box-extraction path for training; train/inference mismatch was a real bug here.

## Git conventions

- Conventional commits (`feat(scope):`, `fix(...)`, `docs:`, `build:`), one concern per commit.
- Subject line plus 2–4 short bullets explaining what and why; keep bodies easy to review.
- Branch off `main`; the maintainer merges with fast-forward.
