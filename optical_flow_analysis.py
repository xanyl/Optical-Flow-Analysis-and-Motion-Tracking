import cv2
import numpy as np
import pandas as pd
from pathlib import Path

VIDEO_PATHS = {
    "sample1": "sample1.mp4",
    "sample2": "sample2.mp4",
}

SEGMENTS = {
    "sample1": {"start_sec": 0, "duration_sec": 30},
    "sample2": {"start_sec": 0, "duration_sec": 30},
}

OUT_DIR = Path("cv_project_output")
OUT_DIR.mkdir(exist_ok=True)

def resize_keep_aspect(frame, width=960):
    h, w = frame.shape[:2]
    scale = width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def flow_to_color(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)      # hue = direction
    hsv[..., 1] = 255                             # saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value = magnitude

    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_bgr, mag

def draw_sparse_arrows(frame_bgr, flow, step=24, scale=3.0):
    vis = frame_bgr.copy()
    h, w = flow.shape[:2]

    ys, xs = np.mgrid[step//2:h:step, step//2:w:step].astype(int)

    for x0, y0 in zip(xs.ravel(), ys.ravel()):
        dx, dy = flow[y0, x0]
        x1 = int(round(x0 + dx * scale))
        y1 = int(round(y0 + dy * scale))

        cv2.line(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(vis, (x0, y0), 1, (0, 255, 0), -1)

    return vis

def process_video(name, path, start_sec=0, duration_sec=30, out_width=960):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    num_frames = min(int(duration_sec * fps), total_frames - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {path}")

    prev = resize_keep_aspect(prev, out_width)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    h, w = prev.shape[:2]
    out_video_path = OUT_DIR / f"{name}_optical_flow.mp4"

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 3, h)
    )

    stats = []

    # save two evidence frames roughly at 1/3 and 2/3 of the clip
    evidence_targets = {num_frames // 3, 2 * num_frames // 3}

    for i in range(1, num_frames):
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_keep_aspect(frame, out_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_color, mag = flow_to_color(flow)
        arrows = draw_sparse_arrows(frame, flow, step=24, scale=3.0)

        combined = np.hstack([frame, flow_color, arrows])
        writer.write(combined)

        stats.append({
            "frame_idx": start_frame + i,
            "time_sec": (start_frame + i) / fps,
            "mean_magnitude": float(np.mean(mag)),
            "max_magnitude": float(np.max(mag)),
            "moving_pixel_ratio_gt_0.5": float(np.mean(mag > 0.5))
        })

        if i in evidence_targets:
            cv2.imwrite(str(OUT_DIR / f"{name}_evidence_frame_{i}.jpg"), combined)

        prev_gray = gray

    writer.release()
    cap.release()

    df = pd.DataFrame(stats)
    csv_path = OUT_DIR / f"{name}_flow_stats.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{name}")
    print(f"  optical flow video: {out_video_path}")
    print(f"  stats csv:          {csv_path}")
    print(f"  mean of mean flow magnitude: {df['mean_magnitude'].mean():.6f}")
    print(f"  max of max flow magnitude:   {df['max_magnitude'].max():.6f}")
    print(f"  mean moving-pixel ratio:     {df['moving_pixel_ratio_gt_0.5'].mean():.6f}")

for name, cfg in SEGMENTS.items():
    process_video(name, VIDEO_PATHS[name], cfg["start_sec"], cfg["duration_sec"], out_width=960)