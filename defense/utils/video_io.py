import cv2
from pathlib import Path
from defense.utils.logger import log

def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    info = get_video_info(cap)
    log("VideoIO", f"Opened: {path}")
    log("VideoIO", f"  Resolution: {info['width']}x{info['height']}, "
                   f"FPS: {info['fps']:.1f}, Frames: {info['frame_count']}")
    return cap

def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Return dict with 'width', 'height', 'fps', 'frame_count'."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

def create_video_writer(path: str, fps: float, width: int, height: int,
                        codec: str = "mp4v") -> cv2.VideoWriter:
    """Create a VideoWriter. Auto-creates parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    log("VideoIO", f"Writer created: {path} ({width}x{height} @ {fps:.1f} FPS, codec={codec})")
    return writer

def extract_frames(video_path: str, output_dir: str, interval: int = 1):
    """Extract every Nth frame from a video to output_dir as JPEG files."""
    cap = open_video(video_path)
    info = get_video_info(cap)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = Path(output_dir) / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if saved % 100 == 0:
                log("VideoIO", f"  Extracted {saved} frames so far (at frame {frame_idx}/{info['frame_count']})")
        frame_idx += 1
    cap.release()
    log("VideoIO", f"✅ Extracted {saved} frames to {output_dir}")
