"""
SfM Mathematical Derivations
=============================
Generates derivation pages (saved as PNG images) covering
the mathematical foundations of Structure from Motion for planar objects.

Topics:
1. Homography for planar scenes
2. Direct Linear Transform (DLT)
3. SVD solution & RANSAC
4. Pose recovery from homography
5. Triangulation
6. Reprojection error & summary
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("cv_project_output")
OUT_DIR.mkdir(exist_ok=True)


def create_derivation_pages():
    """Create all derivation pages and save them."""
    pages = [
        page1_homography_derivation,
        page2_dlt_algorithm,
        page3_svd_and_ransac,
        page4_pose_from_homography,
        page5_triangulation,
        page6_reprojection_and_summary,
    ]

    paths = []
    for i, page_fn in enumerate(pages, 1):
        out_path = OUT_DIR / f"sfm_math_page_{i}.png"
        page_fn(out_path)
        paths.append(out_path)
        print(f"  Page {i} saved: {out_path}")

    return paths


# --- helpers -------------------------------------------------------------

def new_page(title, out_path, figsize=(11, 14)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.text(0.50, 0.97, title,
             fontsize=18, fontweight="bold", ha="center", va="top",
             family="serif",
             bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#1976d2", lw=1.5))
    return fig


def T(fig, y, text, **kw):
    """Add a line of text, return new y."""
    fs = kw.pop("fontsize", 11)
    fig.text(kw.pop("x", 0.06), y, text, fontsize=fs, va="top",
             family="serif", **kw)
    return y - 0.025 * (fs / 11)


def EQ(fig, y, eq, fontsize=13):
    """Add a centered equation (plain-text style, no unsupported LaTeX)."""
    fig.text(0.50, y, eq, fontsize=fontsize, ha="center", va="top",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bdbdbd"))
    return y - 0.045


def SEC(fig, y, title):
    y -= 0.01
    fig.text(0.06, y, title, fontsize=13, fontweight="bold", va="top",
             family="serif", color="#1565c0")
    return y - 0.03


# =========================================================================
# Page 1
# =========================================================================

def page1_homography_derivation(out_path):
    fig = new_page("1. Homography for Planar Scenes", out_path)
    y = 0.91

    y = SEC(fig, y, "1.1  Pinhole Camera Model")
    y = T(fig, y, "A 3D world point  X = (X, Y, Z, 1)T  is projected to pixel  x = (u, v, 1)T  by:")
    y = EQ(fig, y, "s * x  =  K  [ R | t ]  X")
    y = T(fig, y, "where K is the 3x3 intrinsic matrix, R is 3x3 rotation, t is 3x1 translation,")
    y = T(fig, y, "and s is a scale factor (projective depth).")

    y = SEC(fig, y, "1.2  Planar Constraint (Z = 0)")
    y = T(fig, y, "If all object points lie on a plane (e.g. Z = 0), the projection simplifies.")
    y = T(fig, y, "Let R = [r1  r2  r3] be the columns of R.  When Z = 0:")
    y = EQ(fig, y, "s * x  =  K [r1  r2  r3  t] [X, Y, 0, 1]T  =  K [r1  r2  t] [X, Y, 1]T")
    y = T(fig, y, "The third column r3 is multiplied by Z = 0 and drops out.")

    y = SEC(fig, y, "1.3  Homography Definition")
    y = T(fig, y, "Define the 3x3 homography  H  mapping points from the object plane to the image:")
    y = EQ(fig, y, "H  =  K  [ r1   r2   t ]")
    y = T(fig, y, "So every point on the planar object transforms as  x ~ H x'  (equality up to scale).")

    y = SEC(fig, y, "1.4  Inter-Image Homography")
    y = T(fig, y, "Given two views with projection matrices  P1 = K1[R1|t1]  and  P2 = K2[R2|t2],")
    y = T(fig, y, "the inter-image homography (for the Z=0 plane) is:")
    y = EQ(fig, y, "H_12  =  K2 * ( R_12  +  t_12 * nT / d ) * K1_inv")
    y = T(fig, y, "where R12 = R2*R1T, t12 = t2 - R12*t1, n is the plane normal, d is distance to plane.")

    y = SEC(fig, y, "1.5  Degrees of Freedom")
    y = T(fig, y, "H is a 3x3 matrix with 9 entries, defined up to scale => 8 DOF.")
    y = T(fig, y, "Each point correspondence gives 2 equations => minimum 4 correspondences needed.")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Page 2
# =========================================================================

def page2_dlt_algorithm(out_path):
    fig = new_page("2. Direct Linear Transform (DLT) for Homography", out_path)
    y = 0.91

    y = SEC(fig, y, "2.1  Setting Up the Linear System")
    y = T(fig, y, "Given N >= 4 point correspondences  (xi, x'i),  we want H such that  xi ~ H x'i.")
    y = T(fig, y, "The cross-product form gives:   xi  x  H*x'i  =  0")
    y = T(fig, y, "Let  h = vec(HT) = (h1, h2, ..., h9)T  be the 9-vector of H entries (row-major).")
    y = T(fig, y, "Stacking all correspondences:   A * h  =  0")

    y = SEC(fig, y, "2.2  The A Matrix (for one correspondence)")
    y = T(fig, y, "For the i-th correspondence (ui, vi) <-> (u'i, v'i), the two rows of A are:")
    y = EQ(fig, y, "[ -u'i  -v'i  -1    0     0    0   ui*u'i  ui*v'i  ui ]")
    y = EQ(fig, y, "[   0     0    0  -u'i  -v'i  -1   vi*u'i  vi*v'i  vi ]")
    y = T(fig, y, "For N correspondences, A is a 2N x 9 matrix.")

    y = SEC(fig, y, "2.3  Minimum Data")
    y = T(fig, y, "4 correspondences => 8 equations for 8 unknowns (H is up to scale).")
    y = T(fig, y, "With N > 4, the system is over-determined and we use least-squares via SVD.")

    y = SEC(fig, y, "2.4  Normalization (Hartley)")
    y = T(fig, y, "For numerical stability, normalize coordinates before applying DLT:")
    y = T(fig, y, "  1. Translate points so centroid is at the origin.")
    y = T(fig, y, "  2. Scale so mean distance from origin is sqrt(2).")
    y = T(fig, y, "Then compute H on normalized points and de-normalize:  H = T_inv * H_tilde * T'.")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Page 3
# =========================================================================

def page3_svd_and_ransac(out_path):
    fig = new_page("3. SVD Solution & RANSAC Robust Estimation", out_path)
    y = 0.91

    y = SEC(fig, y, "3.1  Solving A*h = 0 via SVD")
    y = T(fig, y, "Compute the Singular Value Decomposition of A:")
    y = EQ(fig, y, "A  =  U  *  Sigma  *  VT")
    y = T(fig, y, "The solution h that minimises ||Ah|| subject to ||h|| = 1 is the LAST column of V,")
    y = T(fig, y, "corresponding to the smallest singular value of A.")
    y = T(fig, y, "Reshape the 9-vector h back into the 3x3 matrix H.")

    y = SEC(fig, y, "3.2  Why SVD Works")
    y = T(fig, y, "The columns of V form an orthonormal basis for R^9.")
    y = T(fig, y, "The last column spans the null space of A (or the closest vector to it")
    y = T(fig, y, "in the least-squares sense when A has full column rank).")

    y = SEC(fig, y, "3.3  RANSAC for Robust Estimation")
    y = T(fig, y, "Real correspondences contain outliers.  RANSAC handles this iteratively:")
    y = T(fig, y, "  1. Randomly sample 4 correspondences (minimal set for H).")
    y = T(fig, y, "  2. Compute H from these 4 points using DLT.")
    y = T(fig, y, "  3. Count inliers: correspondences whose reprojection error < threshold e.")
    y = T(fig, y, "  4. Repeat for k iterations;  keep the H with the most inliers.")
    y = T(fig, y, "  5. Refit H using all inliers for the best model.")

    y = SEC(fig, y, "3.4  Number of Iterations")
    y = T(fig, y, "If the inlier ratio is w and we want success probability p:")
    y = EQ(fig, y, "k  =  log(1 - p) / log(1 - w^4)")
    y = T(fig, y, "For example, with w = 0.5 and p = 0.99:  k ~ 72 iterations.")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Page 4
# =========================================================================

def page4_pose_from_homography(out_path):
    fig = new_page("4. Camera Pose Recovery from Homography", out_path)
    y = 0.91

    y = SEC(fig, y, "4.1  Recall: Homography <-> Camera Pose")
    y = T(fig, y, "For a planar scene the homography between view i and the reference view is:")
    y = EQ(fig, y, "H  =  K  [ r1   r2   t ]")

    y = SEC(fig, y, "4.2  Extracting R and t")
    y = T(fig, y, "Given H and the known intrinsic matrix K, compute:")
    y = EQ(fig, y, "K_inv * H  =  [ r1   r2   t ]   (call this M)")
    y = T(fig, y, "Let m1, m2, m3 be the columns of M.  Then:")
    y = T(fig, y, "  r1 = m1 / ||m1||        (normalize to unit length)")
    y = T(fig, y, "  r2 = m2 / ||m1||        (use same scale factor)")
    y = T(fig, y, "  r3 = r1 x r2            (cross product for right-hand system)")
    y = T(fig, y, "  t  = m3 / ||m1||")

    y = SEC(fig, y, "4.3  Enforcing Rotation Constraints")
    y = T(fig, y, "Due to noise, R = [r1  r2  r3] may not be a valid rotation (det(R) != 1).")
    y = T(fig, y, "Project R onto SO(3) using SVD:")
    y = EQ(fig, y, "R_valid  =  U * diag(1, 1, det(U*VT)) * VT")
    y = T(fig, y, "where  U*Sigma*VT = SVD(R).  This gives the closest valid rotation matrix.")

    y = SEC(fig, y, "4.4  Ambiguity")
    y = T(fig, y, "cv2.decomposeHomographyMat returns up to 4 solutions (R, t, n).")
    y = T(fig, y, "Physical constraints (scene in front of camera, plane normal faces camera)")
    y = T(fig, y, "resolve the ambiguity to select the physically plausible solution.")

    y = SEC(fig, y, "4.5  Camera Center in World Coordinates")
    y = T(fig, y, "The camera center C in world coordinates is recovered as:")
    y = EQ(fig, y, "C  =  -RT * t")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Page 5
# =========================================================================

def page5_triangulation(out_path):
    fig = new_page("5. Triangulation (3D Point Reconstruction)", out_path)
    y = 0.91

    y = SEC(fig, y, "5.1  Problem Statement")
    y = T(fig, y, "Given a point visible in two views with projection matrices P1 and P2,")
    y = T(fig, y, "and corresponding image points x1 and x2, find the 3D point X such that:")
    y = EQ(fig, y, "x1 ~ P1 * X,       x2 ~ P2 * X")

    y = SEC(fig, y, "5.2  Linear Triangulation (DLT)")
    y = T(fig, y, "From  x ~ PX  we get  x x PX = 0.  For  x = (u, v, 1)T  this gives:")
    y = EQ(fig, y, "[ u*p3T - p1T ]")
    y = EQ(fig, y, "[ v*p3T - p2T ]  *  X  =  0")
    y = T(fig, y, "where pi is the i-th row of P.  Stacking two views gives a 4x4 system AX = 0.")

    y = SEC(fig, y, "5.3  SVD Solution")
    y = T(fig, y, "Same approach as for homography: X is the last column of V from SVD(A) = U*Sigma*VT.")
    y = T(fig, y, "Dehomogenize:  X_3D = (X/W, Y/W, Z/W)  where W is the fourth component.")

    y = SEC(fig, y, "5.4  Planar Object Simplification")
    y = T(fig, y, "For a planar object at Z = 0, the 3D reconstruction simplifies significantly.")
    y = T(fig, y, "Given the homography H relating image points to the world plane:")
    y = EQ(fig, y, "[X, Y, 1]T  ~  H_inv * x")
    y = T(fig, y, "The 3D point is simply (X, Y, 0) in the world frame.")
    y = T(fig, y, "This avoids full triangulation and is exact when the planar assumption holds.")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Page 6
# =========================================================================

def page6_reprojection_and_summary(out_path):
    fig = new_page("6. Reprojection Error & Project Summary", out_path)
    y = 0.91

    y = SEC(fig, y, "6.1  Reprojection Error")
    y = T(fig, y, "The reprojection error for a correspondence (xi, x'i) under homography H is:")
    y = EQ(fig, y, "e_i  =  || xi  -  x_hat_i ||_2")
    y = T(fig, y, "where x_hat_i = H * x'i  (after dehomogenization).")
    y = T(fig, y, "The total reprojection error over all N correspondences is:")
    y = EQ(fig, y, "E  =  (1/N) * sum( e_i )   for i = 1..N")
    y = T(fig, y, "A good homography fit typically gives  E < 2-5 pixels.")

    y = SEC(fig, y, "6.2  Geometric Interpretation")
    y = T(fig, y, "The reprojection error measures the Euclidean distance (in pixels) between")
    y = T(fig, y, "where H predicts the point should appear and where it actually appears.")
    y = T(fig, y, "It captures both the quality of feature matching and the accuracy of the model.")

    y = SEC(fig, y, "6.3  Our Experimental Setup")
    y = T(fig, y, "Object:       Book back cover (planar, ~14 cm x 21 cm)")
    y = T(fig, y, "Camera:       iPhone 13 (f ~ 1636 px, cx ~ 757 px, cy ~ 1005 px)")
    y = T(fig, y, "Views:        4 viewpoints -- front, left, right, top")
    y = T(fig, y, "Reference:    front view (camera roughly perpendicular to the book)")
    y = T(fig, y, "Features:     SIFT keypoints with Lowe's ratio test (threshold 0.75)")
    y = T(fig, y, "Estimation:   RANSAC homography (reproj threshold 5 px)")

    y = SEC(fig, y, "6.4  Pipeline Summary")
    y = T(fig, y, "  1.  Detect SIFT features in all four views.")
    y = T(fig, y, "  2.  Match features between each view and the reference (front).")
    y = T(fig, y, "  3.  Estimate homographies using RANSAC + DLT.")
    y = T(fig, y, "  4.  Detect book boundary corners in the reference view.")
    y = T(fig, y, "  5.  Map corners to other views via inverse homography -- validates H.")
    y = T(fig, y, "  6.  Decompose each H into camera rotation R and translation t.")
    y = T(fig, y, "  7.  Plot 3D camera positions relative to the object plane.")
    y = T(fig, y, "  8.  Compute reprojection errors to quantify accuracy.")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("SfM Mathematical Derivations -- Generating Pages")
    print("=" * 70)
    paths = create_derivation_pages()
    print(f"\nAll {len(paths)} derivation pages saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
