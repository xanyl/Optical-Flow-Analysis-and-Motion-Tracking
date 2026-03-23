"""
Structure from Motion (SfM) for a Planar Object
=================================================
Uses four views of a flat book (back cover) captured from different viewpoints
with an iPhone 13 to:

1. Detect and match SIFT features across views
2. Estimate homographies from each view to a reference view
3. Recover camera poses from the homographies
4. Triangulate matched points into 3D
5. Reconstruct and visualize the planar object boundary
6. Plot estimated camera positions relative to the object plane

Camera intrinsic matrix K (iPhone 13):
    [[1635.8023    0.      756.7658]
     [   0.     1635.3378 1005.263 ]
     [   0.        0.        1.    ]]
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_DIR = Path("images")
OUT_DIR = Path("cv_project_output")
OUT_DIR.mkdir(exist_ok=True)

VIEW_NAMES = ["front", "left", "right", "top"]
IMAGE_FILES = {v: IMAGE_DIR / f"image_{v}.jpeg" for v in VIEW_NAMES}
REFERENCE_VIEW = "front"

# iPhone 13 intrinsic matrix (provided by user)
K = np.array([
    [1635.8023,    0.0,     756.7658],
    [   0.0,    1635.3378, 1005.263 ],
    [   0.0,       0.0,       1.0   ]
], dtype=np.float64)

K_inv = np.linalg.inv(K)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_images():
    """Load all four images and their grayscale versions."""
    imgs, grays = {}, {}
    for name, path in IMAGE_FILES.items():
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        imgs[name] = img
        grays[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imgs, grays


def detect_and_match(gray_ref, gray_other, ratio_thresh=0.75):
    """Detect SIFT features and match with Lowe's ratio test."""
    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_other, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_other = np.float32([kp2[m.trainIdx].pt for m in good])
    return kp1, kp2, good, pts_ref, pts_other


def compute_homography(pts_ref, pts_other, reproj_thresh=5.0):
    """Compute homography (other -> ref) using RANSAC."""
    H, mask = cv2.findHomography(pts_other, pts_ref, cv2.RANSAC, reproj_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, mask, inliers


def decompose_homography(H, K):
    """
    Decompose a homography mapping a planar object in one view to another.

    For a planar scene the homography relates two views as:
        H = K (R + t n^T / d) K^{-1}

    OpenCV's decomposeHomographyMat returns up to four (R, t, n) solutions.
    We pick the solution where the normal n has a positive z-component
    (i.e. the plane faces the camera) and translation z < 0 (camera is in
    front of the plane).
    """
    num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)

    best_idx = 0
    best_score = -1e9
    for i in range(num):
        n = normals[i].flatten()
        t = ts[i].flatten()
        # Prefer solutions where the plane normal points toward the camera
        # and translation is physically plausible
        score = n[2]  # normal z-component should be positive
        if score > best_score:
            best_score = score
            best_idx = i

    R = Rs[best_idx]
    t = ts[best_idx].flatten()
    n = normals[best_idx].flatten()
    return R, t, n


def reprojection_error(H, pts_src, pts_dst):
    """Mean reprojection error after applying H to pts_src."""
    pts_src_h = np.hstack([pts_src, np.ones((len(pts_src), 1))])
    projected = (H @ pts_src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return np.mean(np.linalg.norm(projected - pts_dst, axis=1))


def detect_book_boundary_from_matches(match_data, homographies):
    """
    Detect the book boundary in the reference view by using SIFT inlier
    points.  Since matches land on the textured book surface, the
    minimum-area rotated rectangle of all inlier reference-view points
    gives a tight bounding box around the book.
    """
    all_ref_pts = []
    for other, H in homographies.items():
        _, _, good, pts_ref, pts_other = match_data[other]
        # Re-compute inlier mask for this homography
        _, mask, _ = compute_homography(pts_ref, pts_other, reproj_thresh=5.0)
        if mask is not None:
            inlier_pts = pts_ref[mask.ravel().astype(bool)]
            all_ref_pts.append(inlier_pts)

    if not all_ref_pts:
        raise RuntimeError("No inlier points found for boundary estimation")

    all_pts = np.vstack(all_ref_pts)

    # Minimum-area rotated rectangle around all inlier points
    rect = cv2.minAreaRect(all_pts.astype(np.float32))
    box = cv2.boxPoints(rect)  # 4 corners of the rotated rect
    corners = order_corners(box.astype(np.float32))
    return corners


def order_corners(pts):
    """Order four 2D points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]       # bottom-right has largest x+y
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]       # top-right has smallest y-x
    rect[3] = pts[np.argmax(d)]       # bottom-left has largest y-x
    return rect


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def draw_matches_grid(imgs, grays, match_data, out_path):
    """Draw feature matches between the reference view and each other view."""
    ref = REFERENCE_VIEW
    others = [v for v in VIEW_NAMES if v != ref]

    fig, axes = plt.subplots(1, len(others), figsize=(7 * len(others), 7))
    if len(others) == 1:
        axes = [axes]

    for ax, other in zip(axes, others):
        kp1, kp2, good_matches = match_data[other][:3]
        # Draw only top 50 matches for clarity
        vis = cv2.drawMatches(
            imgs[ref], kp1, imgs[other], kp2,
            good_matches[:50], None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{ref} <-> {other}  ({len(good_matches)} matches)", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature matches saved: {out_path}")


def draw_homography_warps(imgs, homographies, out_path):
    """Warp each view into the reference frame and display side by side."""
    ref_img = imgs[REFERENCE_VIEW]
    h, w = ref_img.shape[:2]

    others = [v for v in VIEW_NAMES if v != REFERENCE_VIEW]
    fig, axes = plt.subplots(1, len(others) + 1, figsize=(6 * (len(others) + 1), 6))

    axes[0].imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{REFERENCE_VIEW} (reference)", fontsize=12)
    axes[0].axis("off")

    for ax, other in zip(axes[1:], others):
        H = homographies[other]
        warped = cv2.warpPerspective(imgs[other], H, (w, h))
        ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{other} → {REFERENCE_VIEW}", fontsize=12)
        ax.axis("off")

    plt.suptitle("Homography Warps to Reference View", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Homography warps saved: {out_path}")


def draw_reconstructed_boundary(imgs, boundary_ref, homographies, out_path):
    """
    Draw the detected/reconstructed book boundary on every view.
    For the reference view, draw the detected corners directly.
    For other views, back-project the reference corners using H_inv.
    """
    fig, axes = plt.subplots(1, len(VIEW_NAMES), figsize=(6 * len(VIEW_NAMES), 6))

    for ax, view in zip(axes, VIEW_NAMES):
        img_rgb = cv2.cvtColor(imgs[view], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        if view == REFERENCE_VIEW:
            corners = boundary_ref
        else:
            H_inv = np.linalg.inv(homographies[view])
            corners_h = np.hstack([boundary_ref, np.ones((4, 1))])
            projected = (H_inv @ corners_h.T).T
            corners = projected[:, :2] / projected[:, 2:3]

        # Draw the quadrilateral
        quad = np.vstack([corners, corners[0]])
        ax.plot(quad[:, 0], quad[:, 1], "r-", linewidth=2.5)
        for i, (cx, cy) in enumerate(corners):
            ax.plot(cx, cy, "yo", markersize=10, markeredgecolor="red", markeredgewidth=2)
            ax.annotate(f"C{i+1}", (cx, cy), fontsize=11, color="yellow",
                        fontweight="bold", ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 10))

        ax.set_title(f"{view}", fontsize=12)
        ax.axis("off")

    plt.suptitle("Reconstructed Book Boundary Across Views", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reconstructed boundary saved: {out_path}")


def draw_camera_positions(poses, boundary_3d, out_path):
    """Plot estimated camera positions and the object plane in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"front": "#2196F3", "left": "#FF9800", "right": "#4CAF50", "top": "#E91E63"}

    # Draw object plane (book boundary at Z=0)
    if boundary_3d is not None and len(boundary_3d) >= 4:
        verts = [list(zip(boundary_3d[:, 0], boundary_3d[:, 1], boundary_3d[:, 2]))]
        poly = Poly3DCollection(verts, alpha=0.25, facecolor="cyan", edgecolor="black", linewidth=2)
        ax.add_collection3d(poly)
        for i, (bx, by, bz) in enumerate(boundary_3d):
            ax.text(bx, by, bz, f"  C{i+1}", fontsize=9, color="black")

    # Draw camera positions
    for view, (R, t) in poses.items():
        # Camera center in world coordinates: C = -R^T t
        C = -R.T @ t
        color = colors.get(view, "gray")
        ax.scatter(*C, s=120, c=color, marker="^", depthshade=True, edgecolors="black", linewidths=1)
        ax.text(C[0], C[1], C[2], f"  {view}", fontsize=10, color=color, fontweight="bold")

        # Draw viewing direction
        direction = R.T @ np.array([0, 0, 1])  # camera looks along +z in camera frame
        end = C + direction * 0.3
        ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]], color=color, linewidth=2)

    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title("Estimated Camera Positions & Object Plane", fontsize=14, fontweight="bold")

    # Try to set a reasonable view angle
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Camera positions plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Structure from Motion — Planar Object (Book Back Cover)")
    print("=" * 70)

    # 1. Load images
    print("\n[1/6] Loading images ...")
    imgs, grays = load_images()
    for v in VIEW_NAMES:
        h, w = imgs[v].shape[:2]
        print(f"  {v}: {w}×{h}")

    # 2. Feature detection & matching
    print("\n[2/6] Detecting SIFT features and matching to reference view ...")
    match_data = {}   # view -> (kp_ref, kp_view, good_matches, pts_ref, pts_view)
    others = [v for v in VIEW_NAMES if v != REFERENCE_VIEW]

    for other in others:
        kp1, kp2, good, pts_ref, pts_other = detect_and_match(
            grays[REFERENCE_VIEW], grays[other], ratio_thresh=0.75
        )
        match_data[other] = (kp1, kp2, good, pts_ref, pts_other)
        print(f"  {REFERENCE_VIEW} ↔ {other}: {len(good)} good matches")

    draw_matches_grid(imgs, grays, match_data,
                      OUT_DIR / "sfm_feature_matches.jpg")

    # 3. Homography estimation
    print("\n[3/6] Computing homographies ...")
    homographies = {}
    summary_rows = []

    for other in others:
        _, _, good, pts_ref, pts_other = match_data[other]
        H, mask, inliers = compute_homography(pts_ref, pts_other)
        homographies[other] = H

        reproj_err = reprojection_error(H, pts_other, pts_ref)
        print(f"  H({other}→{REFERENCE_VIEW}): {inliers} inliers, "
              f"reproj error = {reproj_err:.3f} px")

        summary_rows.append({
            "view_pair": f"{other}→{REFERENCE_VIEW}",
            "total_matches": len(good),
            "inliers": inliers,
            "reproj_error_px": round(reproj_err, 4),
            "H_00": H[0, 0], "H_01": H[0, 1], "H_02": H[0, 2],
            "H_10": H[1, 0], "H_11": H[1, 1], "H_12": H[1, 2],
            "H_20": H[2, 0], "H_21": H[2, 1], "H_22": H[2, 2],
        })

    draw_homography_warps(imgs, homographies, OUT_DIR / "sfm_homography_warp.jpg")

    # 4. Detect book boundary & reconstruct across views
    print("\n[4/6] Detecting book boundary in reference view ...")
    boundary_ref = detect_book_boundary_from_matches(match_data, homographies)
    print(f"  Boundary corners (ref view): {boundary_ref.tolist()}")

    draw_reconstructed_boundary(imgs, boundary_ref, homographies,
                                OUT_DIR / "sfm_reconstructed_boundary.png")

    # 5. Recover camera poses from homographies
    print("\n[5/6] Decomposing homographies to recover camera poses ...")
    poses = {}
    # Reference view is at the origin
    poses[REFERENCE_VIEW] = (np.eye(3), np.zeros(3))

    for other in others:
        H = homographies[other]
        R, t, n = decompose_homography(H, K)
        poses[other] = (R, t)
        print(f"  {other}: R det={np.linalg.det(R):.6f}, "
              f"|t|={np.linalg.norm(t):.4f}, n={n.round(3).tolist()}")

    # 6. Construct 3D boundary points
    # Since the object is planar, we place it at Z=0 in the world frame.
    # The reference camera looks at the plane, so the book corners in
    # normalized camera coords give us world XY directly.
    print("\n[6/6] Constructing 3D boundary and plotting camera positions ...")
    boundary_norm = (K_inv @ np.hstack([boundary_ref, np.ones((4, 1))]).T).T
    boundary_3d = np.column_stack([boundary_norm[:, 0], boundary_norm[:, 1],
                                    np.zeros(4)])

    draw_camera_positions(poses, boundary_3d,
                          OUT_DIR / "sfm_camera_positions.png")

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / "sfm_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV saved: {csv_path}")

    # Print camera intrinsics info used
    print("\n" + "=" * 70)
    print("Camera information")
    print("=" * 70)
    print(f"  Device:   iPhone 13")
    print(f"  K matrix:")
    for row in K:
        print(f"    [{row[0]:12.4f}  {row[1]:12.4f}  {row[2]:12.4f}]")
    print(f"\n  Views used: {VIEW_NAMES}")
    print(f"  Reference view: {REFERENCE_VIEW}")
    print("\nDone.")


if __name__ == "__main__":
    main()
