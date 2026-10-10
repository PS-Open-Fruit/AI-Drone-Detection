"""CPU throughput and per-stage latency benchmark for the video tracking pipeline.

Reproduces the scripts/track_video.py path (same config loading, component
construction, call order and arguments) and times every stage by wrapping the
component instances with timers at runtime. Library code is not modified.

Each (video, thread setting, repeat) runs in a fresh subprocess, so model
loading and warm-up are measured cold every time. Results are written to
data/benchmark_results.json (summary) and data/benchmark_results_frames.json
(raw per-frame timings).

The numbers describe the machine the script runs on. A run limited to 4
threads on a desktop CPU is a thread-sensitivity experiment, NOT a
Raspberry Pi 5 measurement, and must not be used to estimate Pi throughput.

Example:
    uv run python scripts/benchmark_pipeline.py data/p3_video/P1_VIDEO_5.mp4 \
        data/p3_video/P1_VIDEO_4.mp4 --max-frames 600 --threads default 4 --repeats 3
"""
import os

_ORIG_ENV = dict(os.environ)  # captured before ultralytics sets OMP_NUM_THREADS on import

import argparse
import datetime
import functools
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil

from defense.config import (
    load_config,
    get_detection_config,
    get_localization_config,
    get_tracking_config,
    get_camera_config,
)
from defense.detection import build_drone_detector
from defense.localization import GeoLocalizer, FusionEngine
from defense.tracking import VideoTracker, HUDRenderer, TelemetryDispatcher
from defense.utils.video_io import open_video, get_video_info, create_video_writer
from defense.utils.logger import log
import defense.tracking.tracker as tracker_module

REPO_ROOT = Path(__file__).resolve().parents[1]

DISCLAIMER = (
    "Desktop CPU measurements only. These are NOT Raspberry Pi 5 measurements. The 4-thread runs limit "
    "torch/OpenCV threads on the desktop CPU as a thread-sensitivity experiment; they do not emulate a Pi 5 "
    "(Cortex-A76 cores are much slower per core) and must not be used to estimate Pi throughput."
)

# Directly timed stages. Their per-frame sum plus the two residuals equals the end-to-end frame time.
TOP_STAGES = [
    ("decode", "Video decode (cap.read)"),
    ("binarize", "Grayscale + threshold (DroneDetector)"),
    ("morphology", "Morphology (ImageProcessor.process)"),
    ("tight_boxes", "Tight candidate boxes (DroneDetector)"),
    ("clean_image", "clean_image (legacy path; unused by DroneDetector)"),
    ("yolo", "YOLO detect (YoloDetector.detect)"),
    ("fusion", "Fusion (FusionEngine.fuse)"),
    ("tracking", "Association + track management"),
    ("localization", "GBDT localization (features + predict)"),
    ("hud", "HUD render"),
    ("telemetry", "Telemetry (disabled in config)"),
    ("video_write", "Video write (VideoWriter.write)"),
    ("loop_other", "Loop residual (progress log, bookkeeping)"),
]
SUB_STAGES = {
    "yolo": [
        ("yolo_preprocess", "ultralytics preprocess (letterbox)"),
        ("yolo_inference", "ultralytics inference"),
        ("yolo_postprocess", "ultralytics postprocess (NMS)"),
        ("yolo_overhead", "predict() setup, logging, result conversion"),
    ],
    "tracking": [
        ("association", "_associate"),
        ("merge", "_merge_colocated_tracks"),
        ("tracker_other", "updates, creation, reaping, logging (residual)"),
    ],
    "localization": [
        ("gbdt_features", "encode_features"),
        ("gbdt_predict", "GeoLocalizer.predict"),
    ],
}
# Residuals that are not directly timed (used for the reconciliation check).
RESIDUAL_STAGES = ["tracker_other", "loop_other"]


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates wall-clock milliseconds per named stage for the current frame."""

    def __init__(self):
        self.current = defaultdict(float)
        self.calls = defaultdict(int)

    def add(self, name: str, ms: float):
        self.current[name] += ms

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.current[name] += (time.perf_counter() - t0) * 1e3
                self.calls[name] += 1
        return timed

    def pop_frame(self) -> dict:
        out = dict(self.current)
        self.current.clear()
        return out


def instrument(timer: StageTimer, tracker, image_processor, detector, fusion_engine, localizer, drone_detector=None):
    """Wrap component methods with timers. Arguments and call order are passed through unchanged."""
    image_processor._to_grayscale = timer.wrap("binarize", image_processor._to_grayscale)
    image_processor._binarize = timer.wrap("binarize", image_processor._binarize)
    if drone_detector is not None:
        drone_detector._tight_box = timer.wrap("tight_boxes", drone_detector._tight_box)
    image_processor.process = timer.wrap("morphology", image_processor.process)
    image_processor.clean_image = timer.wrap("clean_image", image_processor.clean_image)
    fusion_engine.fuse = timer.wrap("fusion", fusion_engine.fuse)
    tracker._associate = timer.wrap("association", tracker._associate)
    tracker._merge_colocated_tracks = timer.wrap("merge", tracker._merge_colocated_tracks)
    tracker_module.encode_features = timer.wrap("gbdt_features", tracker_module.encode_features)
    if localizer is not None:
        localizer.predict = timer.wrap("gbdt_predict", localizer.predict)

    if detector is not None:
        original_detect = detector.detect

        @functools.wraps(original_detect)
        def timed_detect(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original_detect(*args, **kwargs)
            finally:
                timer.add("yolo", (time.perf_counter() - t0) * 1e3)
                timer.calls["yolo"] += 1
                # ultralytics' own profiler split for the call that just finished
                predictor = getattr(detector.model, "predictor", None)
                results = getattr(predictor, "results", None)
                if results:
                    speed = getattr(results[0], "speed", None) or {}
                    timer.add("yolo_preprocess", float(speed.get("preprocess", 0.0)))
                    timer.add("yolo_inference", float(speed.get("inference", 0.0)))
                    timer.add("yolo_postprocess", float(speed.get("postprocess", 0.0)))

        detector.detect = timed_detect


def apply_threads(n: int):
    import torch
    torch.set_num_threads(n)
    cv2.setNumThreads(n)


def observed_threads() -> dict:
    import torch
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cv2_num_threads": cv2.getNumThreads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "std": float(np.std(a)),
    }


def derive_columns(cols: dict) -> dict:
    """Add aggregate and residual columns to raw per-frame columns (all values in ms)."""
    n = len(cols["total"])
    z = np.zeros(n)
    g = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    get = lambda k: g.get(k, z)
    g["yolo_overhead"] = get("yolo") - get("yolo_preprocess") - get("yolo_inference") - get("yolo_postprocess")
    g["localization"] = get("gbdt_features") + get("gbdt_predict")
    timed_inside = (get("binarize") + get("morphology") + get("tight_boxes") + get("clean_image") + get("yolo") + get("fusion")
                    + get("association") + get("merge") + g["localization"])
    g["tracker_other"] = get("process_frame") - timed_inside
    g["tracking"] = get("association") + get("merge") + g["tracker_other"]
    g["loop_other"] = (get("total") - get("decode") - get("process_frame") - get("hud")
                       - get("telemetry") - get("video_write"))
    for name, _ in TOP_STAGES:
        g.setdefault(name, z)
    return g


def stage_table(cols: dict) -> dict:
    g = derive_columns(cols)
    total_mean = float(np.mean(g["total"]))
    table = {}
    names = [n for n, _ in TOP_STAGES] + [s for subs in SUB_STAGES.values() for s, _ in subs]
    for name in names + ["process_frame", "total"]:
        d = describe(g[name])
        d["share_of_total_pct"] = 100.0 * d["mean"] / total_mean if total_mean > 0 else 0.0
        table[name] = d
    residual_mean = sum(table[r]["mean"] for r in RESIDUAL_STAGES)
    table["_reconciliation"] = {
        "end_to_end_mean_ms": total_mean,
        "sum_of_directly_timed_stage_means_ms": total_mean - residual_mean,
        "unattributed_mean_ms": residual_mean,
        "unattributed_pct": 100.0 * residual_mean / total_mean if total_mean > 0 else 0.0,
    }
    return table


# ---------------------------------------------------------------------------
# System information and idle check
# ---------------------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _device_model() -> str:
    """Board name on single-board computers (e.g. "Raspberry Pi 5 Model B Rev 1.0"); empty elsewhere."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(path).read_text(errors="ignore").strip("\x00").strip()
        except OSError:
            continue
    return ""


def thermal_snapshot() -> dict:
    """SoC temperature and (on Raspberry Pi) firmware throttling flags. Empty fields where unavailable."""
    snap = {"cpu_temp_c": None, "throttled": None}
    try:
        snap["cpu_temp_c"] = round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0, 1)
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        snap["throttled"] = out.stdout.strip() or None  # "throttled=0x0" means no throttling ever occurred
    except (OSError, subprocess.SubprocessError):
        pass
    return snap


def _power_plan() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
            return out.stdout.strip()
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if gov.exists():
            return f"scaling_governor={gov.read_text().strip()}"
    except Exception as e:  # informational only
        return f"unavailable ({e})"
    return "unknown"


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_system_info() -> dict:
    import torch
    import sklearn
    import ultralytics

    parallel_framework = ""
    for line in cv2.getBuildInformation().splitlines():
        if "Parallel framework" in line:
            parallel_framework = line.split(":", 1)[1].strip()
    aten_backend = ""
    for line in torch.__config__.parallel_info().splitlines():
        if "ATen parallel backend" in line:
            aten_backend = line.split(":", 1)[1].strip()

    vm = psutil.virtual_memory()
    return {
        "hardware": {
            "cpu_name": _cpu_name(),
            "device_model": _device_model(),
            "thermal_at_start": thermal_snapshot(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(vm.total / 1024 ** 3, 1),
            "ram_available_gb_at_start": round(vm.available / 1024 ** 3, 1),
            "machine": platform.machine(),
            "power_plan": _power_plan(),
            "gpu_used": False,
        },
        "software": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_aten_parallel_backend": aten_backend,
            "torch_mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "opencv_parallel_framework": parallel_framework,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "psutil": psutil.__version__,
            "env_thread_vars_before_import": {
                k: _ORIG_ENV.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS")
            },
        },
        "repo": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "pipeline_paths_modified": _git("status", "--porcelain", "--", "defense", "config",
                                            "scripts/track_video.py").splitlines(),
        },
    }


def sample_cpu_load(seconds: int, top_n: int = 5) -> dict:
    """Sample system-wide CPU utilisation once per second and list the busiest processes."""
    ncpu = psutil.cpu_count(logical=True) or 1
    procs = []
    for p in psutil.process_iter(["name"]):
        try:
            p.cpu_percent(None)
            procs.append(p)
        except (psutil.Error, OSError):
            pass
    psutil.cpu_percent(None)
    samples = [psutil.cpu_percent(interval=1.0) for _ in range(max(1, seconds))]
    busiest = []
    for p in procs:
        try:
            if p.pid == 0:  # System Idle Process
                continue
            share = p.cpu_percent(None) / ncpu
            if share > 0.0:
                busiest.append({"name": p.info.get("name"), "pid": p.pid, "cpu_pct_of_machine": round(share, 2)})
        except (psutil.Error, OSError):
            pass
    busiest.sort(key=lambda x: x["cpu_pct_of_machine"], reverse=True)
    return {
        "seconds": len(samples),
        "samples_pct": samples,
        "mean_pct": float(np.mean(samples)),
        "max_pct": float(np.max(samples)),
        "busiest_processes": busiest[:top_n],
    }


# ---------------------------------------------------------------------------
# Single benchmark run (worker process)
# ---------------------------------------------------------------------------

def benchmark_once(args) -> dict:
    """Run the track_video.py pipeline on one video with stage timers. Mirrors scripts/track_video.py."""
    t_entry = time.perf_counter()
    threads = args.threads[0]

    cfg = load_config(args.config)
    det_cfg = get_detection_config(cfg)
    loc_cfg = get_localization_config(cfg)
    trk_cfg = get_tracking_config(cfg)
    cam_cfg = get_camera_config(cfg)
    out_cfg = cfg.get("output", {})
    tel_cfg = cfg.get("telemetry", {})

    yolo_path = args.yolo_model or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
    gbdt_dir = args.gbdt_models or loc_cfg.get("models_dir", "models/")
    fusion_mode = args.fusion or loc_cfg.get("fusion_mode", "adaptive")
    if args.conf is not None:
        det_cfg["confidence"] = args.conf
    confidence = float(det_cfg.get("confidence", 0.40))

    video_path = args.videos[0]
    cap = open_video(video_path)
    info = get_video_info(cap)
    width = info["width"]
    height = info["height"]
    fps = info["fps"] if info["fps"] > 0 else 30.0
    frame_count = info["frame_count"]

    t0 = time.perf_counter()
    drone_detector = build_drone_detector(det_cfg, yolo_path=yolo_path)
    detector = drone_detector.yolo
    yolo_load_ms = (time.perf_counter() - t0) * 1e3
    # Count parameters before the first predict fuses Conv+BN in place.
    yolo_params = int(sum(p.numel() for p in detector.model.model.parameters())) if detector else 0

    t0 = time.perf_counter()
    localizer = None
    try:
        localizer = GeoLocalizer(cam_cfg, models_dir=gbdt_dir)
        localizer.load()
    except Exception as e:
        log("Defense", f"⚠️ Localizer not loaded: {e}. Localization disabled on HUD.")
        localizer = None
    gbdt_load_ms = (time.perf_counter() - t0) * 1e3

    image_processor = drone_detector.image_processor
    fusion_engine = FusionEngine.from_config(loc_cfg, mode=fusion_mode)
    tracker = VideoTracker(
        detector=drone_detector,
        localizer=localizer,
        fusion_engine=fusion_engine,
        config=trk_cfg,
    )
    hud = HUDRenderer()
    telemetry = TelemetryDispatcher(
        camera_config=cam_cfg,
        api_endpoint=tel_cfg.get("api_endpoint", "") if tel_cfg.get("enabled", False) else ""
    )
    codec = out_cfg.get("codec", "mp4v")
    out_video = Path(args.video_out)
    writer = create_video_writer(str(out_video), fps, width, height, codec=codec)

    timer = StageTimer()
    instrument(timer, tracker, image_processor, detector, fusion_engine, localizer, drone_detector)
    if threads != "default":
        apply_threads(int(threads))
    threads_before = observed_threads()
    setup_ms = (time.perf_counter() - t_entry) * 1e3

    log("Bench", f"Benchmarking {video_path} | threads={threads} | max_frames={args.max_frames} | warmup={args.warmup}")

    records = []
    frame_starts = []
    frame_ends = []
    threads_after_first = None

    # ---- main loop: identical call order to scripts/track_video.py (debug branch omitted, --debug unused) ----
    start_time = time.time()
    frame_idx = 0
    max_simul = 0
    fps_time = time.time()

    while True:
        t_frame = time.perf_counter()
        ret, frame = cap.read()
        t_decoded = time.perf_counter()
        if not ret:
            break

        frame_idx += 1
        tracks = tracker.process_frame(frame)
        t_tracked = time.perf_counter()
        if len(tracks) > max_simul:
            max_simul = len(tracks)

        annotated = hud.render(frame, tracks, frame_idx)
        t_hud = time.perf_counter()
        telemetry.maybe_send_alert(tracks, frame, frame_idx)
        t_tel = time.perf_counter()
        writer.write(annotated)
        t_written = time.perf_counter()

        if frame_idx % 10 == 0 or frame_idx == 1 or frame_idx == frame_count:
            now = time.time()
            instant_fps = 10.0 / max(0.001, now - fps_time) if frame_idx > 1 else fps
            fps_time = now
            lost_tracks = sum(1 for t in tracker.tracks if t.lost_count > 0)
            log("Tracker", f"Frame {frame_idx}/{frame_count} | FPS: {instant_fps:.1f} | Drones: {len(tracks)} | Tracks: {len(tracker.tracks)} ({lost_tracks} lost)")

        t_end = time.perf_counter()

        # ---- bookkeeping (outside the timed region) ----
        rec = timer.pop_frame()
        rec["decode"] = (t_decoded - t_frame) * 1e3
        rec["process_frame"] = (t_tracked - t_decoded) * 1e3
        rec["hud"] = (t_hud - t_tracked) * 1e3
        rec["telemetry"] = (t_tel - t_hud) * 1e3
        rec["video_write"] = (t_written - t_tel) * 1e3
        rec["total"] = (t_end - t_frame) * 1e3
        rec["n_seg_candidates"] = len(tracker.last_seg_candidates)
        rec["n_yolo_detections"] = len(tracker.last_yolo_detections)
        rec["n_confirmed_tracks"] = len(tracks)
        rec["n_gbdt_calls"] = sum(1 for t in tracker.tracks if t.confirmed) if localizer is not None else 0
        records.append(rec)
        frame_starts.append(t_frame)
        frame_ends.append(t_end)

        if frame_idx == 1:
            # ultralytics resets torch threads in select_device() during the first predict; re-apply the limit.
            if threads != "default":
                apply_threads(int(threads))
            threads_after_first = observed_threads()

        if args.max_frames and frame_idx >= args.max_frames:
            log("Tracker", f"Reached max frames limit: {args.max_frames}")
            break

    t0 = time.perf_counter()
    cap.release()
    writer.release()
    release_ms = (time.perf_counter() - t0) * 1e3
    total_time = time.time() - start_time
    threads_end = observed_threads()

    size_mb = out_video.stat().st_size / (1024 * 1024) if out_video.exists() else 0.0
    if not args.keep_videos and out_video.exists():
        out_video.unlink()

    # Stages that never ran in a frame (e.g. GBDT before any track is confirmed) count as 0 ms.
    all_keys = sorted({k for r in records for k in r})
    columns = {k: [r.get(k, 0.0) for r in records] for k in all_keys}
    n = frame_idx
    w = min(args.warmup, max(0, n - 1))
    steady = {k: v[w:] for k, v in columns.items()}
    total_ms = np.asarray(columns["total"])
    steady_total = np.asarray(steady["total"])
    steady_wall_s = (frame_ends[-1] - frame_starts[w]) if n > w else 0.0

    yolo_input = {}
    if detector is not None:
        predictor = getattr(detector.model, "predictor", None)
        yolo_input = {
            "imgsz_arg": getattr(getattr(predictor, "args", None), "imgsz", None),
            "rect": getattr(getattr(predictor, "args", None), "rect", None),
            "input_pixels": getattr(predictor, "pixels", None),
            "sahi": False,
        }

    return {
        "video": str(video_path),
        "video_info": {
            "width": width, "height": height, "fps": fps, "frame_count_reported": frame_count,
            "file_size_mb": round(Path(video_path).stat().st_size / 1024 ** 2, 2),
        },
        "threads_setting": threads,
        "threads_observed": {"before_loop": threads_before, "after_frame_1": threads_after_first,
                             "end": threads_end},
        "pipeline": {
            "yolo_model": yolo_path,
            "yolo_model_sha256": _sha256(Path(yolo_path)),
            "yolo_params": yolo_params,
            "yolo_input": yolo_input,
            "confidence": confidence,
            "device": det_cfg.get("device", "cpu"),
            "fusion_mode": fusion_mode,
            "localization_enabled": localizer is not None,
            "telemetry_enabled": bool(telemetry.api_endpoint),
            "codec": codec,
        },
        "frames_processed": n,
        "warmup_frames": w,
        "load_ms": {
            "setup_total_ms": setup_ms,
            "yolo_load_ms": yolo_load_ms,
            "gbdt_load_ms": gbdt_load_ms,
            "writer_release_ms": release_ms,
        },
        "warmup": {
            "frame_1_ms": float(total_ms[0]) if n else None,
            "frame_1_yolo_ms": float(columns["yolo"][0]) if n and "yolo" in columns else None,
            "frames_1_to_warmup_total_ms": float(total_ms[:w].sum()) if w else 0.0,
            "frames_2_to_warmup_mean_ms": float(total_ms[1:w].mean()) if w > 1 else None,
        },
        "steady_state": {
            "frames": int(steady_total.size),
            "first_frame_index": w + 1,
            "last_frame_index": n,
            "fps": float(steady_total.size / (steady_total.sum() / 1e3)) if steady_total.size else 0.0,
            "fps_wall_clock": float(steady_total.size / steady_wall_s) if steady_wall_s > 0 else 0.0,
            "ms_per_frame": describe(steady_total),
        },
        "track_video_equivalent": {
            "note": "Same formula as scripts/track_video.py: all frames incl. warm-up, plus writer release.",
            "avg_fps": n / max(0.001, total_time),
            "total_time_s": total_time,
        },
        "content": {
            "max_simultaneous_confirmed": max_simul,
            "tracks_created": tracker.next_id - 1,
            "mean_seg_candidates": float(np.mean(steady["n_seg_candidates"])) if n > w else 0.0,
            "mean_yolo_detections": float(np.mean(steady["n_yolo_detections"])) if n > w else 0.0,
            "mean_gbdt_calls_per_frame": float(np.mean(steady["n_gbdt_calls"])) if n > w else 0.0,
        },
        "output_video_mb": round(size_mb, 2),
        "stages_steady_state": stage_table(steady),
        "per_frame_ms": {k: [round(float(x), 4) for x in v] for k, v in columns.items()},
    }


# ---------------------------------------------------------------------------
# Orchestration (parent process)
# ---------------------------------------------------------------------------

def _video_label(path: str) -> str:
    return Path(path).stem


def _run_key(run: dict) -> str:
    return f"{_video_label(run['video'])}|{run['threads_setting']}|b{run.get('batch', 1)}r{run['repeat']}"


def summarize_group(runs: list[dict]) -> dict:
    fps = [r["steady_state"]["fps"] for r in runs]
    med = [r["steady_state"]["ms_per_frame"]["median"] for r in runs]
    mean = [r["steady_state"]["ms_per_frame"]["mean"] for r in runs]
    p95 = [r["steady_state"]["ms_per_frame"]["p95"] for r in runs]
    pooled = defaultdict(list)
    all_keys = {k for r in runs for k in r["per_frame_ms"]}
    for r in runs:
        w = r["warmup_frames"]
        n_steady = len(r["per_frame_ms"]["total"]) - w
        for k in all_keys:
            v = r["per_frame_ms"].get(k)
            pooled[k].extend(v[w:] if v is not None else [0.0] * n_steady)
    fps_mean = float(np.mean(fps))
    return {
        "runs": len(runs),
        "run_keys": [_run_key(r) for r in runs],
        "steady_fps_per_run": fps,
        "steady_fps_median_of_runs": float(np.median(fps)),
        "steady_fps_mean": fps_mean,
        "steady_fps_min": float(np.min(fps)),
        "steady_fps_max": float(np.max(fps)),
        "steady_fps_spread_pct": 100.0 * (max(fps) - min(fps)) / fps_mean if fps_mean else 0.0,
        "ms_median_per_run": med,
        "ms_mean_per_run": mean,
        "ms_p95_per_run": p95,
        "pooled_ms_per_frame": describe(pooled["total"]),
        "pooled_fps": float(len(pooled["total"]) / (np.sum(pooled["total"]) / 1e3)),
        "track_video_equivalent_avg_fps_per_run": [r["track_video_equivalent"]["avg_fps"] for r in runs],
        "frame_1_ms_per_run": [r["warmup"]["frame_1_ms"] for r in runs],
        "warmup_total_ms_per_run": [r["warmup"]["frames_1_to_warmup_total_ms"] for r in runs],
        "yolo_load_ms_per_run": [r["load_ms"]["yolo_load_ms"] for r in runs],
        "setup_ms_per_run": [r["load_ms"]["setup_total_ms"] for r in runs],
        "threads_observed_after_frame_1": runs[0]["threads_observed"]["after_frame_1"],
        "stages_pooled": stage_table(pooled),
    }


def print_summary(summary: dict):
    log("Bench", "=" * 100)
    log("Bench", DISCLAIMER)
    log("Bench", "End-to-end steady state (FPS per run; ms/frame pooled over all steady-state frames of all runs):")
    log("Bench", f"{'video':<14}{'threads':<9}{'runs':>5}{'FPS med':>9}{'FPS mean':>10}{'FPS min-max':>16}"
                 f"{'spread':>8}{'median':>9}{'mean':>9}{'p95':>9}")
    for key, s in summary.items():
        video, threads = key.split("|")
        pm = s["pooled_ms_per_frame"]
        log("Bench", f"{video:<14}{threads:<9}{s['runs']:>5}{s['steady_fps_median_of_runs']:>9.2f}"
                     f"{s['steady_fps_mean']:>10.2f}"
                     f"{s['steady_fps_min']:>8.2f}-{s['steady_fps_max']:<7.2f}{s['steady_fps_spread_pct']:>7.1f}%"
                     f"{pm['median']:>9.2f}{pm['mean']:>9.2f}{pm['p95']:>9.2f}")
    for key, s in summary.items():
        st = s["stages_pooled"]
        log("Bench", "-" * 100)
        log("Bench", f"Per-stage ms ({key.replace('|', ', threads=')}), steady-state frames pooled over {s['runs']} runs")
        log("Bench", f"{'stage':<48}{'median':>9}{'p95':>9}{'mean':>9}{'share':>8}")
        for name, label in TOP_STAGES:
            d = st[name]
            log("Bench", f"{label:<48}{d['median']:>9.2f}{d['p95']:>9.2f}{d['mean']:>9.2f}{d['share_of_total_pct']:>7.1f}%")
            for sub, sub_label in SUB_STAGES.get(name, []):
                d = st[sub]
                log("Bench", f"{'  - ' + sub_label:<48}{d['median']:>9.2f}{d['p95']:>9.2f}{d['mean']:>9.2f}"
                             f"{d['share_of_total_pct']:>7.1f}%")
        d = st["total"]
        log("Bench", f"{'END-TO-END':<48}{d['median']:>9.2f}{d['p95']:>9.2f}{d['mean']:>9.2f}{100.0:>7.1f}%")
        rc = st["_reconciliation"]
        log("Bench", f"Reconciliation: directly timed stages sum to {rc['sum_of_directly_timed_stage_means_ms']:.2f} ms "
                     f"of {rc['end_to_end_mean_ms']:.2f} ms mean ({rc['unattributed_pct']:.2f}% unattributed)")
    log("Bench", "=" * 100)


def orchestrate(args):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="defense_bench_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    videos = [str(Path(v).resolve()) for v in args.videos]
    for v in videos:
        if not Path(v).exists():
            raise FileNotFoundError(f"Video not found: {v}")

    log("Bench", "🧪 Pipeline CPU benchmark")
    log("Bench", DISCLAIMER)
    system = collect_system_info()
    hw = system["hardware"]
    log("Bench", f"CPU: {hw['cpu_name']} ({hw['cpu_physical_cores']} cores / {hw['cpu_logical_cores']} threads), "
                 f"RAM {hw['ram_total_gb']} GB | {system['software']['os']}")
    log("Bench", f"Power plan: {hw['power_plan']}")

    log("Bench", f"Sampling idle CPU load for {args.idle_seconds}s...")
    idle = sample_cpu_load(args.idle_seconds)
    log("Bench", f"Idle CPU load: mean {idle['mean_pct']:.1f}%, max {idle['max_pct']:.1f}% | busiest: "
                 + ", ".join(f"{p['name']} {p['cpu_pct_of_machine']:.1f}%" for p in idle["busiest_processes"]))
    idle["threshold_pct"] = args.max_idle_load
    idle["passed"] = idle["mean_pct"] <= args.max_idle_load
    if not idle["passed"]:
        log("Bench", f"⚠️ Machine is not idle (mean {idle['mean_pct']:.1f}% > {args.max_idle_load}%). "
                     f"Results may be pessimistic; recorded in the JSON.")

    env = dict(_ORIG_ENV)
    env["PYTHONIOENCODING"] = "utf-8"

    runs = []
    plan = [(rep, threads, video) for rep in range(1, args.repeats + 1)
            for threads in args.threads for video in videos]
    for i, (rep, threads, video) in enumerate(plan, 1):
        tag = f"{_video_label(video)}_t{threads}_r{rep}"
        worker_json = work_dir / f"{tag}.json"
        worker_log = work_dir / f"{tag}.log"
        pre_load = sample_cpu_load(args.pre_run_seconds, top_n=3)
        cmd = [sys.executable, str(Path(__file__).resolve()), video, "--worker",
               "--threads", threads, "--warmup", str(args.warmup),
               "--worker-out", str(worker_json), "--video-out", str(work_dir / f"{tag}.mp4")]
        cmd += ["--max-frames", str(args.max_frames or 0)]
        for flag, val in (("--config", args.config), ("--yolo-model", args.yolo_model),
                          ("--gbdt-models", args.gbdt_models), ("--fusion", args.fusion), ("--conf", args.conf)):
            if val is not None:
                cmd += [flag, str(val)]
        if args.keep_videos:
            cmd.append("--keep-videos")
        log("Bench", f"[{i}/{len(plan)}] {tag} (pre-run load {pre_load['mean_pct']:.1f}%) ...")
        t0 = time.perf_counter()
        with open(worker_log, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed for {tag} (exit {proc.returncode}); see {worker_log}")
        run = json.loads(worker_json.read_text(encoding="utf-8"))
        run.update({"batch": 1, "repeat": rep, "order": i, "pre_run_cpu_load": pre_load,
                    "worker_log": str(worker_log), "process_wall_s": time.perf_counter() - t0})
        runs.append(run)
        ss = run["steady_state"]
        log("Bench", f"    steady {ss['fps']:.2f} FPS | median {ss['ms_per_frame']['median']:.2f} ms | "
                     f"p95 {ss['ms_per_frame']['p95']:.2f} ms | frame 1 {run['warmup']['frame_1_ms']:.0f} ms | "
                     f"torch threads {run['threads_observed']['after_frame_1']['torch_num_threads']}, "
                     f"cv2 threads {run['threads_observed']['after_frame_1']['cv2_num_threads']}")

    created = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    command = " ".join([Path(sys.executable).name] + sys.argv)
    protocol = {
        "videos": videos,
        "max_frames": args.max_frames,
        "warmup_frames_excluded": args.warmup,
        "repeats": args.repeats,
        "thread_settings": args.threads,
        "thread_setting_meaning": {
            "default": "As scripts/track_video.py runs: ultralytics sets torch threads to min(8, cpu_count-1) "
                       "on the first predict and disables OpenCV threading (cv2.setNumThreads(0)) on import.",
            "N": "torch.set_num_threads(N) and cv2.setNumThreads(N), applied before the loop and re-applied "
                 "after frame 1 because ultralytics resets torch threads during predictor setup.",
        },
        "isolation": "Each run is a fresh subprocess (cold model load and warm-up). Order interleaves "
                     "repeat -> thread setting -> video.",
        "stdout": "Pipeline stdout/stderr (per-frame ultralytics and tracker logs) redirected to a file.",
        "timing": "time.perf_counter wall-clock per stage; YOLO sub-split from ultralytics Results.speed.",
        "output_video": "Annotated video written as in track_video.py, then deleted unless --keep-videos.",
    }
    batches = [{"batch": 1, "created": created, "command": command, "idle_check": idle}]
    system["hardware"]["thermal_at_end"] = thermal_snapshot()
    write_results(out_path, system, protocol, batches, runs, command)
    log("Bench", f"   Worker logs: {work_dir}")


def write_results(out_path: Path, system: dict, protocol: dict, batches: list[dict], runs: list[dict], command: str):
    """Summarise runs per (video, thread setting) and write the results + per-frame JSON files."""
    frames_path = out_path.with_name(out_path.stem + "_frames.json")
    groups = defaultdict(list)
    for r in runs:
        groups[f"{_video_label(r['video'])}|{r['threads_setting']}"].append(r)
    summary = {k: summarize_group(v) for k, v in groups.items()}

    frames = {_run_key(r): r.pop("per_frame_ms") for r in runs}
    results = {
        "schema": "defense.benchmark_pipeline/v1",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "command": command,
        **system,
        "protocol": protocol,
        "batches": batches,
        "summary": summary,
        "runs": runs,
        "frames_file": str(frames_path),
    }
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    frames_path.write_text(json.dumps({"disclaimer": DISCLAIMER, "per_frame_ms": frames}), encoding="utf-8")
    print_summary(summary)
    log("Bench", f"✅ Results: {out_path}")
    log("Bench", f"   Per-frame timings: {frames_path}")


def merge_results(sources: list[str], out_path: Path):
    """Combine results files from separate invocations (e.g. batches run at different times) into one."""
    runs, batches = [], []
    system = protocol = None
    for src in map(Path, sources):
        d = json.loads(src.read_text(encoding="utf-8"))
        frames = json.loads(src.with_name(src.stem + "_frames.json").read_text(encoding="utf-8"))["per_frame_ms"]
        if protocol is None:
            system = {k: d[k] for k in ("hardware", "software", "repo")}
            protocol = dict(d["protocol"])
        else:
            for k in ("max_frames", "warmup_frames_excluded"):
                if d["protocol"][k] != protocol[k]:
                    raise ValueError(f"Cannot merge {src}: protocol '{k}' differs ({d['protocol'][k]} vs {protocol[k]})")
            if d["hardware"]["cpu_name"] != system["hardware"]["cpu_name"]:
                raise ValueError(f"Cannot merge {src}: recorded on a different CPU")
        src_batches = d.get("batches") or [{"batch": 1, "created": d["created"], "command": d["command"],
                                             "idle_check": d["idle_check"]}]
        offset = len(batches)
        for b in src_batches:
            batches.append({**b, "batch": offset + b.get("batch", 1), "source_file": str(src)})
        for r in d["runs"]:
            key = _run_key(r)
            if key not in frames:  # files written before batch tags were added
                key = f"{_video_label(r['video'])}|{r['threads_setting']}|r{r['repeat']}"
            r["per_frame_ms"] = frames[key]
            r["batch"] = offset + r.get("batch", 1)
            runs.append(r)
    protocol["merged_from"] = [str(s) for s in sources]
    log("Bench", f"Merging {len(runs)} runs from {len(batches)} batches into {out_path}")
    write_results(out_path, system, protocol, batches, runs, " ".join([Path(sys.executable).name] + sys.argv))


def main():
    parser = argparse.ArgumentParser(description="Drone Tracking Pipeline — CPU Throughput & Per-Stage Latency Benchmark")
    parser.add_argument("videos", nargs="*", help="Input video(s) to benchmark")
    parser.add_argument("--max-frames", type=int, default=600, help="Frames to process per run (0 = whole video)")
    parser.add_argument("--warmup", type=int, default=20, help="Leading frames excluded from steady-state stats")
    parser.add_argument("--threads", nargs="+", default=["default", "4"],
                        help="Thread settings: 'default' (as track_video.py) and/or integers for torch+OpenCV")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per (video, thread setting)")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json", help="Results JSON path")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory for worker logs and temporary output videos (default: new temp dir)")
    parser.add_argument("--keep-videos", action="store_true", help="Keep the annotated output videos")
    parser.add_argument("--idle-seconds", type=int, default=10, help="Seconds of idle CPU sampling before runs")
    parser.add_argument("--pre-run-seconds", type=int, default=2, help="Seconds of CPU sampling before each run")
    parser.add_argument("--max-idle-load", type=float, default=10.0, help="Warn if idle CPU load exceeds this %%")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO model (.pt)")
    parser.add_argument("--gbdt-models", type=str, default=None, help="Directory containing GBDT models")
    parser.add_argument("--fusion", type=str, default=None, help="Fusion mode: adaptive, filter, soft, hard, yolo, seg")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--merge", nargs="+", default=None, metavar="RESULTS_JSON",
                        help="Combine existing results files into --output instead of running a benchmark")
    # Internal: single run in a fresh process
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--video-out", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        if len(args.videos) != 1 or len(args.threads) != 1 or not args.worker_out or not args.video_out:
            parser.error("--worker needs exactly one video, one --threads value, --worker-out and --video-out")
        result = benchmark_once(args)
        Path(args.worker_out).write_text(json.dumps(result), encoding="utf-8")
        return

    if args.merge:
        out_path = Path(args.output)
        if out_path.resolve() in {Path(m).resolve() for m in args.merge}:
            parser.error("--output must differ from the files being merged")
        merge_results(args.merge, out_path)
        return

    if not args.videos:
        parser.error("at least one input video is required (or use --merge)")
    for t in args.threads:
        if t != "default" and not t.isdigit():
            parser.error(f"Invalid --threads value: {t}")
    orchestrate(args)


if __name__ == "__main__":
    main()
