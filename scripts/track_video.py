import argparse
from pathlib import Path
import time
import cv2
from defense.config import (
    load_config,
    get_detection_config,
    get_localization_config,
    get_tracking_config,
    get_camera_config,
)
from defense.detection import YoloDetector, ImageProcessor
from defense.localization import GBDTRegressor, FusionEngine
from defense.tracking import VideoTracker, HUDRenderer, TelemetryDispatcher
from defense.utils.video_io import open_video, get_video_info, create_video_writer
from defense.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="Drone Video Tracker — Full Real-Time Pipeline with HUD Overlay")
    parser.add_argument("input_video", type=str, help="Path to input video (.mp4, .avi, etc.)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output annotated video")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO model (.pt)")
    parser.add_argument("--gbdt-models", type=str, default=None, help="Directory containing GBDT models")
    parser.add_argument("--fusion", type=str, default=None, help="Fusion mode: adaptive, filter, soft, hard, yolo, seg")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--debug", type=str, default=None, help="Directory to save 5-stage intermediate debug frames")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    cfg = load_config(args.config)
    det_cfg = get_detection_config(cfg)
    loc_cfg = get_localization_config(cfg)
    trk_cfg = get_tracking_config(cfg)
    cam_cfg = get_camera_config(cfg)
    out_cfg = cfg.get("output", {})
    tel_cfg = cfg.get("telemetry", {})

    yolo_path = args.yolo_model or det_cfg.get("yolo_model_path", "models/yolo11n-obb.pt")
    gbdt_dir = args.gbdt_models or loc_cfg.get("gbdt_models_dir", "models/")
    fusion_mode = args.fusion or loc_cfg.get("fusion_mode", "adaptive")
    confidence = args.conf if args.conf is not None else float(det_cfg.get("confidence", 0.75))

    cap = open_video(args.input_video)
    info = get_video_info(cap)
    width = info["width"]
    height = info["height"]
    fps = info["fps"] if info["fps"] > 0 else 30.0
    frame_count = info["frame_count"]

    # Initialize components
    detector = None
    if Path(yolo_path).exists():
        detector = YoloDetector(model_path=yolo_path, confidence=confidence, device=det_cfg.get("device", "cpu"))
    else:
        log("Defense", f"⚠️ YOLO weights not found at {yolo_path}. Proceeding with morphological segmentation only.")

    localizer = None
    try:
        localizer = GBDTRegressor(models_dir=gbdt_dir)
        localizer.load()
    except Exception as e:
        log("Defense", f"⚠️ GBDT models not loaded: {e}. Localization disabled on HUD.")
        localizer = None

    image_processor = ImageProcessor(det_cfg)
    fusion_engine = FusionEngine(
        mode=fusion_mode,
        iou_weight=loc_cfg.get("iou_weight", 0.60),
        shape_weight=loc_cfg.get("shape_weight", 0.30),
        motion_weight=loc_cfg.get("motion_weight", 0.10),
        soft_threshold=loc_cfg.get("soft_threshold", 0.25),
        area_ratio_min=loc_cfg.get("area_ratio_min", 0.00005),
        area_ratio_max=loc_cfg.get("area_ratio_max", 0.003),
        aspect_ratio_min=loc_cfg.get("aspect_ratio_min", 0.4),
        aspect_ratio_max=loc_cfg.get("aspect_ratio_max", 3.0),
    )

    tracker = VideoTracker(
        detector=detector,
        localizer=localizer,
        fusion_engine=fusion_engine,
        config=trk_cfg,
        image_processor=image_processor
    )

    hud = HUDRenderer()
    telemetry = TelemetryDispatcher(
        camera_config=cam_cfg,
        api_endpoint=tel_cfg.get("api_endpoint", "") if tel_cfg.get("enabled", False) else ""
    )

    # Output writer
    codec = out_cfg.get("codec", "mp4v")
    writer = create_video_writer(args.output, fps, width, height, codec=codec)

    # Debug setup if requested
    debug_dirs = {}
    if args.debug:
        base_dbg = Path(args.debug)
        for stage in ["1_original", "2_binary_mask", "3_detections", "4_tracking", "5_final_annotated"]:
            sd = base_dbg / stage
            sd.mkdir(parents=True, exist_ok=True)
            debug_dirs[stage] = sd

    # Startup Banner
    log("Defense", "🎯 Drone Video Tracker")
    log("Defense", f"Input: {args.input_video} ({width}x{height}, {fps:.1f} FPS, {frame_count} frames)")
    log("Defense", f"Output: {args.output}")
    log("Defense", f"Model: {yolo_path} (conf={confidence})")
    log("Defense", f"Fusion: {fusion_mode} | Localization: {'enabled' if localizer else 'disabled'}")

    start_time = time.time()
    frame_idx = 0
    max_simul = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        tracks = tracker.process_frame(frame)
        if len(tracks) > max_simul:
            max_simul = len(tracks)

        annotated = hud.render(frame, tracks, frame_idx)
        telemetry.maybe_send_alert(tracks, frame, frame_idx)
        writer.write(annotated)

        # Debug frame saving
        if debug_dirs:
            cv2.imwrite(str(debug_dirs["1_original"] / f"frame_{frame_idx:06d}.jpg"), frame)
            # Binary mask
            mask, _ = image_processor.process(frame)
            cv2.imwrite(str(debug_dirs["2_binary_mask"] / f"frame_{frame_idx:06d}.jpg"), mask)
            # Annotations
            cv2.imwrite(str(debug_dirs["5_final_annotated"] / f"frame_{frame_idx:06d}.jpg"), annotated)

        # Print progress regularly
        if frame_idx % 10 == 0 or frame_idx == 1 or frame_idx == frame_count:
            now = time.time()
            instant_fps = 10.0 / max(0.001, now - fps_time) if frame_idx > 1 else fps
            fps_time = now
            lost_tracks = sum(1 for t in tracker.tracks if t.lost_count > 0)
            log("Tracker", f"Frame {frame_idx}/{frame_count} | FPS: {instant_fps:.1f} | Drones: {len(tracks)} | Tracks: {len(tracker.tracks)} ({lost_tracks} lost)")

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / max(0.001, total_time)
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    out_file = Path(args.output)
    size_mb = out_file.stat().st_size / (1024 * 1024) if out_file.exists() else 0.0

    log("Defense", "✅ Video tracking complete!")
    log("Defense", f"Processed {frame_idx} frames in {mins}m {secs}s (avg {avg_fps:.1f} FPS)")
    log("Defense", f"Total tracks created: {tracker.next_id - 1} | Max simultaneous: {max_simul}")
    log("Defense", f"Output: {args.output} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
