import cv2
from pathlib import Path

VIDEO_PATHS = {
    "sample1": "sample1.mp4",
    "sample2": "sample2.mp4",
}

OUT_DIR = Path("cv_project_output")
OUT_DIR.mkdir(exist_ok=True)

SEGMENTS = {
    "sample1": {"start_sec": 0, "duration_sec": 30},
    "sample2": {"start_sec": 0, "duration_sec": 30},
}

def save_preview_frames(video_path, name, start_sec, duration_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = min(int((start_sec + duration_sec) * fps), total_frames - 1)

    preview_indices = [
        start_frame,
        (start_frame + end_frame) // 2,
        end_frame
    ]

    saved = []
    for idx_i, frame_idx in enumerate(preview_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            out_path = OUT_DIR / f"{name}_preview_{idx_i+1}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append((frame_idx, str(out_path)))

    cap.release()
    return saved

for name, cfg in SEGMENTS.items():
    saved = save_preview_frames(
        VIDEO_PATHS[name],
        name,
        cfg["start_sec"],
        cfg["duration_sec"]
    )
    print(f"\n{name} preview frames:")
    for frame_idx, path in saved:
        print(f"  frame {frame_idx} -> {path}")