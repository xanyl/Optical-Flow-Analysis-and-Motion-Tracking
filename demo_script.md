# Demo Video Script (~5 minutes)

**Recording tip**: Use Win+G (Game Bar) or OBS. Show each file/image on screen as you speak.

---

## 1. Intro (30s)
**[Show: Project folder in VS Code]**

"Hi, I'm Anil Tiwari. This is my Assignment 6 for Computer Vision, covering optical flow analysis, motion tracking, and structure from motion. All code is in Python using OpenCV and NumPy, and is available on my GitHub repo. Let me walk through the project."

---

## 2. Running the Project (30s)
**[Show: Terminal]**

"The project is organized with a main.py entry point. Running `python main.py --part a` executes all Part A scripts — optical flow and tracking. Running `python main.py --part b` runs the Structure from Motion pipeline. All outputs go into the cv_project_output folder."

**[Run: `python main.py --part a` — let it start, then cut to results]**

---

## 3. Optical Flow (60s)
**[Show: preview images, then evidence frames]**

"I selected two videos — sample1 is a rotating Earth globe with smooth rotational motion, and sample2 shows a bird near flowing water with much stronger, irregular motion. Both are over 30 seconds.

I computed dense optical flow using the Farnebäck method. These evidence frames show three panels — the original frame, the HSV flow visualization where color encodes direction and brightness encodes speed, and sparse arrow overlays.

For the Earth video, flow is concentrated on the globe surface while the background stays dark. For the water video, flow is much stronger and spatially irregular, concentrated in the water region.

The statistics confirm this — mean flow magnitude is 0.21 for sample1 versus 1.78 for sample2, and the moving pixel ratio jumps from 18% to 61%."

**[Show: mean_flow_plot and moving_ratio_plot for both]**

---

## 4. Tracking Theory & Validation (90s)
**[Show: Math section in the PDF report]**

"For motion tracking, I derived everything from fundamentals. Starting from brightness constancy — a pixel's intensity stays the same as it moves — I applied a Taylor expansion to get the optical flow constraint equation. This gives one equation per pixel but two unknowns, which is the aperture problem.

The Lucas-Kanade method resolves this by assuming constant flow in a local window, creating an overdetermined system solved by least squares. I also implemented bilinear interpolation for subpixel accuracy — it estimates intensity at non-integer positions using a weighted average of four neighboring pixels.

**[Show: tracking validation tables and images]**

To validate, I picked two consecutive frames from each video, detected 8 feature points, predicted their next positions using my tracker, and compared against ground truth from template matching.

For sample1, the mean error is 0.38 pixels. For sample2, it's 0.44 pixels. All errors are sub-pixel — well below one pixel — confirming that the theoretical Lucas-Kanade result matches actual pixel motion."

---

## 5. Structure from Motion (90s)
**[Show: 4 book images]**

"For Part B, I captured four images of a book's back cover using my iPhone 13 — front, left, right, and top views. The book is a flat planar object with rich texture. The camera intrinsic matrix was calibrated with a focal length of about 1636 pixels.

**[Show: sfm_feature_matches.jpg]**

I used SIFT for feature detection and matched keypoints across all views to the front reference. The front-to-right pair had the most matches at 226 good matches with 165 RANSAC inliers.

**[Show: sfm_homography_warp.jpg]**

Homographies were estimated using RANSAC to map each view into the reference frame. This warp image shows all views aligned — the book region matches correctly.

**[Show: sfm_reconstructed_boundary.png]**

For boundary reconstruction, I used the inlier match points to find the minimum-area bounding rectangle of the book, then projected it to all views via inverse homographies. The red quadrilateral accurately outlines the book in every view.

**[Show: sfm_camera_positions.png]**

Finally, I decomposed the homographies to recover camera poses. This 3D plot shows the four estimated camera positions — front, left, right, and top — matching their actual arrangement.

**[Show: sfm_math_page_1 through 6, scroll quickly]**

All the mathematical derivations — homography, DLT, SVD, RANSAC, pose recovery, and reprojection error — are included in both the report and these generated pages."

---

## 6. Wrap-up (20s)
**[Show: GitHub repo page]**

"To summarize — Part A demonstrated optical flow computation, motion tracking validated with sub-pixel accuracy, and Part B showed structure from motion with accurate boundary reconstruction across four views. All code, documentation, and results are on my GitHub. Thank you."

---

### Submission Checklist
- [ ] Compile `main.tex` to PDF
- [ ] Record screen while reading this script
- [ ] Ensure GitHub repo is public
- [ ] Upload PDF + video to Google Classroom
