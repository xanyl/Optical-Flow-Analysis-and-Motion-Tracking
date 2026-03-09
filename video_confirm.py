import cv2
import os
from pathlib import Path

VIDEO_PATHS = {
    "sample1": "sample1.mp4",
    "sample2": "sample2.mp4",
}

OUT_DIR = Path("cv_project_output")
OUT_DIR.mkdir(exist_ok=True)

def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": duration
    }

for name, path in VIDEO_PATHS.items():
    info = video_info(path)
    print(f"\n{name}")
    for k, v in info.items():
        print(f"  {k}: {v}")