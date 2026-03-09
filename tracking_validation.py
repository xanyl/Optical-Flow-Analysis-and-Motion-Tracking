import cv2
import numpy as np
import pandas as pd
import math
from pathlib import Path

VIDEO_PATHS = {
    "sample1": "sample1.mp4",
    "sample2": "sample2.mp4",
}

OUT_DIR = Path("cv_project_output")
OUT_DIR.mkdir(exist_ok=True)

def get_frame(path, frame_idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    return frame

def resize_keep_aspect(frame, width=960):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def bilinear_sample(img, x, y):
    h, w = img.shape
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0

    I00 = img[y0, x0]
    I10 = img[y0, x0 + 1]
    I01 = img[y0 + 1, x0]
    I11 = img[y0 + 1, x0 + 1]

    return (
        (1 - dx) * (1 - dy) * I00 +
        dx * (1 - dy) * I10 +
        (1 - dx) * dy * I01 +
        dx * dy * I11
    )

def lk_track_point(I1, I2, x, y, win=7, max_iters=15, eps=1e-3):
    Ix = cv2.Sobel(I2, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(I2, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    u = 0.0
    v = 0.0

    coords = [(j, i) for i in range(-win, win + 1) for j in range(-win, win + 1)]
    T = np.array([bilinear_sample(I1, x + j, y + i) for j, i in coords], dtype=np.float32)

    for _ in range(max_iters):
        A = []
        b = []

        for (j, i), Tval in zip(coords, T):
            xp = x + j + u
            yp = y + i + v

            Iw = bilinear_sample(I2, xp, yp)
            gx = bilinear_sample(Ix, xp, yp)
            gy = bilinear_sample(Iy, xp, yp)

            A.append([gx, gy])
            b.append(Tval - Iw)

        A = np.asarray(A, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32).reshape(-1, 1)

        H = A.T @ A
        if np.linalg.det(H) < 1e-6:
            return np.nan, np.nan, False

        delta = np.linalg.solve(H, A.T @ b).flatten()
        u += float(delta[0])
        v += float(delta[1])

        if np.linalg.norm(delta) < eps:
            break

    return x + u, y + v, True

def exhaustive_match(I1, I2, x, y, win=7, search=12):
    x = int(round(x))
    y = int(round(y))
    h, w = I1.shape

    if x - win < 0 or y - win < 0 or x + win >= w or y + win >= h:
        return np.nan, np.nan, False

    T = I1[y - win:y + win + 1, x - win:x + win + 1]

    best = None
    for yy in range(max(win, y - search), min(h - win, y + search + 1)):
        for xx in range(max(win, x - search), min(w - win, x + search + 1)):
            P = I2[yy - win:yy + win + 1, xx - win:xx + win + 1]
            ssd = float(np.sum((T - P) ** 2))
            if best is None or ssd < best[0]:
                best = (ssd, xx, yy)

    if best is None:
        return np.nan, np.nan, False

    return float(best[1]), float(best[2]), True

def pick_features(gray, max_corners=8, quality=0.03, min_dist=25, roi=None, margin=20):
    gray_u8 = (gray * 255).astype(np.uint8)

    mask = None
    if roi is not None:
        x1, y1, x2, y2 = roi
        mask = np.zeros_like(gray_u8)
        mask[y1:y2, x1:x2] = 255

    corners = cv2.goodFeaturesToTrack(
        gray_u8,
        maxCorners=max_corners * 5,
        qualityLevel=quality,
        minDistance=min_dist,
        mask=mask
    )

    pts = []
    if corners is None:
        return pts

    h, w = gray.shape
    for c in corners.reshape(-1, 2):
        x, y = map(float, c)
        if margin <= x < w - margin and margin <= y < h - margin:
            pts.append((x, y))
            if len(pts) >= max_corners:
                break

    return pts

def draw_validation(frame1, frame2, df, out_path):
    vis1 = frame1.copy()
    vis2 = frame2.copy()

    for _, row in df.iterrows():
        x0, y0 = int(round(row["x0"])), int(round(row["y0"]))
        xp, yp = int(round(row["x_pred"])), int(round(row["y_pred"]))
        xa, ya = int(round(row["x_actual"])), int(round(row["y_actual"]))

        cv2.circle(vis1, (x0, y0), 5, (0, 255, 255), 2)
        cv2.circle(vis2, (xp, yp), 5, (0, 255, 0), 2)     # predicted
        cv2.circle(vis2, (xa, ya), 5, (0, 0, 255), 2)     # actual

        cv2.line(vis2, (xp, yp), (xa, ya), (255, 255, 0), 1)

    combined = np.hstack([vis1, vis2])
    cv2.imwrite(str(out_path), combined)

def validate_video(name, path, frame_idx, roi=None):
    frame1 = resize_keep_aspect(get_frame(path, frame_idx), width=960)
    frame2 = resize_keep_aspect(get_frame(path, frame_idx + 1), width=960)

    I1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    I2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    points = pick_features(I1, max_corners=8, quality=0.03, min_dist=25, roi=roi, margin=20)

    rows = []
    for i, (x, y) in enumerate(points, start=1):
        x_pred, y_pred, ok_pred = lk_track_point(I1, I2, x, y, win=7, max_iters=15, eps=1e-3)
        x_act, y_act, ok_act = exhaustive_match(I1, I2, x, y, win=7, search=12)

        if ok_pred and ok_act and np.isfinite(x_pred) and np.isfinite(x_act):
            err = math.hypot(x_pred - x_act, y_pred - y_act)
            rows.append({
                "point_id": i,
                "x0": x,
                "y0": y,
                "x_pred": x_pred,
                "y_pred": y_pred,
                "x_actual": x_act,
                "y_actual": y_act,
                "error_pixels": err
            })

    df = pd.DataFrame(rows)

    csv_path = OUT_DIR / f"{name}_tracking_validation.csv"
    img_path = OUT_DIR / f"{name}_tracking_validation.jpg"

    df.to_csv(csv_path, index=False)
    draw_validation(frame1, frame2, df, img_path)

    print(f"\n{name}")
    print(f"  frame pair used: {frame_idx} and {frame_idx + 1}")
    print(f"  validation table: {csv_path}")
    print(f"  validation image: {img_path}")
    print(df.round(4))
    print(f"  mean error = {df['error_pixels'].mean():.6f} pixels")
    print(f"  max error  = {df['error_pixels'].max():.6f} pixels")

# Suggested frame pairs:
# sample1: coherent globe motion
# sample2: stronger water motion region
validate_video("sample1", VIDEO_PATHS["sample1"], frame_idx=150, roi=None)

# ROI chosen to focus on water motion rather than static bird/rock.
# Format = (x1, y1, x2, y2) after resizing to width=960.
validate_video("sample2", VIDEO_PATHS["sample2"], frame_idx=186, roi=(420, 30, 930, 330))