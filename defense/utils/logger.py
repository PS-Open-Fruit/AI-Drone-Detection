import sys
import time

# Ensure Windows console / stdout handles UTF-8 and unicode emojis properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def log(module: str, msg: str):
    """Print a timestamped, prefixed log message. Always goes to stdout."""
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] [{module}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(line.encode(enc, errors="replace").decode(enc))
