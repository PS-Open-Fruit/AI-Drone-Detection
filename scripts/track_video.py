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
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
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
            # Stage 1: Original frame
            cv2.imwrite(str(debug_dirs["1_original"] / f"frame_{frame_idx:06d}.jpg"), frame)

            # Stage 2: Binary mask (white background, black objects)
            mask = tracker.last_binary_mask if tracker.last_binary_mask is not None else image_processor.process(frame)[0]
            cv2.imwrite(str(debug_dirs["2_binary_mask"] / f"frame_{frame_idx:06d}.jpg"), mask)

            # Stage 3: Detections (SEG in blue, YOLO in orange, Fused in green)
            detection_frame = frame.copy()
            for y_det in tracker.last_yolo_detections:
                yx = int(round(y_det["center_x"] - y_det["width"] / 2.0))
                yy = int(round(y_det["center_y"] - y_det["height"] / 2.0))
                yw = int(round(y_det["width"]))
                yh = int(round(y_det["height"]))
                y_conf = y_det.get("confidence", 0.0)
                cv2.rectangle(detection_frame, (yx, yy), (yx + yw, yy + yh), (0, 165, 255), 2)
                cv2.putText(detection_frame, f"YOLO {y_conf:.2f}", (yx, max(12, yy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            for (sx, sy, sw, sh) in tracker.last_seg_candidates:
                cv2.rectangle(detection_frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                cv2.putText(detection_frame, "SEG", (sx, max(12, sy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for i, f_det in enumerate(tracker.last_fused_detections):
                fx = int(round(f_det["center_x"] - f_det["width"] / 2.0))
                fy = int(round(f_det["center_y"] - f_det["height"] / 2.0))
                fw = int(round(f_det["width"]))
                fh = int(round(f_det["height"]))
                f_conf = f_det.get("confidence", 0.0)
                cv2.rectangle(detection_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
                label = f"FUSED-{i+1}"
                if f_conf > 0:
                    label += f" {f_conf:.2f}"
                cv2.putText(detection_frame, label, (fx, max(14, fy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(detection_frame, f"Fusion: {fusion_mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(detection_frame,
                        f"SEG: {len(tracker.last_seg_candidates)} | YOLO: {len(tracker.last_yolo_detections)} | FUSED: {len(tracker.last_fused_detections)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(debug_dirs["3_detections"] / f"frame_{frame_idx:06d}.jpg"), detection_frame)

            # Stage 4: Tracking status
            tracking_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            for trk in tracker.tracks:
                tx, ty, tw, th = trk.bbox
                if trk.confirmed:
                    t_color = trk.color
                    status_lbl = "CONFIRMED"
                else:
                    t_color = (128, 128, 128)
                    status_lbl = "UNCONFIRMED"

                cv2.rectangle(tracking_frame, (tx, ty), (tx + tw, ty + th), t_color, 2)
                cx, cy = int(round(trk.centroid[0])), int(round(trk.centroid[1]))
                cv2.circle(tracking_frame, (cx, cy), 3, t_color, -1)
                info_y = max(15, ty - 10)
                cv2.putText(tracking_frame, f"ID:{trk.track_id} {status_lbl}",
                            (tx, info_y), font, 0.4, t_color, 1)
                cv2.putText(tracking_frame, f"Frames:{trk.frames_tracked} Lost:{trk.lost_count}",
                            (tx, max(28, info_y - 15)), font, 0.3, t_color, 1)

            cv2.putText(tracking_frame, f"Active Tracks: {len(tracker.tracks)}",
                        (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(tracking_frame, f"Confirmed: {len([t for t in tracker.tracks if t.confirmed])}",
                        (10, 60), font, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(debug_dirs["4_tracking"] / f"frame_{frame_idx:06d}.jpg"), tracking_frame)

            # Stage 5: Final annotated frame
            cv2.imwrite(str(debug_dirs["5_final_annotated"] / f"frame_{frame_idx:06d}.jpg"), annotated)

        # Print progress regularly
        if frame_idx % 10 == 0 or frame_idx == 1 or frame_idx == frame_count:
            now = time.time()
            instant_fps = 10.0 / max(0.001, now - fps_time) if frame_idx > 1 else fps
            fps_time = now
            lost_tracks = sum(1 for t in tracker.tracks if t.lost_count > 0)
            log("Tracker", f"Frame {frame_idx}/{frame_count} | FPS: {instant_fps:.1f} | Drones: {len(tracks)} | Tracks: {len(tracker.tracks)} ({lost_tracks} lost)")

        if args.max_frames and frame_idx >= args.max_frames:
            log("Tracker", f"Reached max frames limit: {args.max_frames}")
            break

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
