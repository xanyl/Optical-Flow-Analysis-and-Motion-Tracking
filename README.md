# Optical Flow and Motion Tracking Analysis

A complete Computer Vision project pipeline for analyzing motion in videos using **dense optical flow** and **feature tracking validation**.  
This project was built to satisfy an academic assignment requiring:

- analysis of **two videos**
- computation and visualization of **optical flow**
- interpretation of motion from flow fields
- derivation and validation of **motion tracking equations**
- practical comparison between **predicted** and **actual** pixel locations

---

## Project Overview

This repository processes two videos and performs the following tasks in sequence:

1. **Verify video properties**
   - frame rate
   - number of frames
   - resolution
   - duration

2. **Extract preview frames**
   - beginning
   - middle
   - end of the selected 30-second clip

3. **Compute dense optical flow**
   - Farneback optical flow
   - color-coded flow visualization
   - sparse motion arrows
   - output optical-flow video

4. **Generate motion statistics**
   - mean optical flow magnitude
   - maximum optical flow magnitude
   - ratio of moving pixels

5. **Plot motion evidence**
   - motion intensity over time
   - moving-pixel ratio over time

6. **Validate tracking between consecutive frames**
   - Lucas–Kanade style tracking
   - bilinear interpolation
   - comparison with exhaustive template matching
   - pixel-wise tracking error

---

## Project Structure

```text
Optical Flow Analysis/
│
├── sample1.mp4
├── sample2.mp4
│
├── video_confirm.py
├── extract_frames.py
├── optical_flow_analysis.py
├── plot_motion_stats.py
├── tracking_validation.py
├── main.py
│
├── report.tex
├── references.bib
├── README.md
│
└── cv_project_output/
    ├── sample1_preview_1.jpg
    ├── sample1_preview_2.jpg
    ├── sample1_preview_3.jpg
    ├── sample2_preview_1.jpg
    ├── sample2_preview_2.jpg
    ├── sample2_preview_3.jpg
    ├── sample1_optical_flow.mp4
    ├── sample2_optical_flow.mp4
    ├── sample1_flow_stats.csv
    ├── sample2_flow_stats.csv
    ├── sample1_mean_flow_plot.png
    ├── sample2_mean_flow_plot.png
    ├── sample1_moving_ratio_plot.png
    ├── sample2_moving_ratio_plot.png
    ├── sample1_tracking_validation.csv
    ├── sample2_tracking_validation.csv
    ├── sample1_tracking_validation.jpg
    └── sample2_tracking_validation.jpg
```

---

## Features

- End-to-end optical flow analysis on two videos
- Automated sequential execution using `main.py`
- Dense optical flow visualization with Farneback method
- Motion statistics saved as CSV files
- Automatic graph generation for report evidence
- Motion tracking validation with subpixel estimation
- LaTeX report support with bibliography
- Clean output organization inside `cv_project_output/`

---

## Requirements

Install the required Python packages before running the project.

```bash
pip install opencv-python numpy matplotlib pandas
```

Recommended:
- Python 3.9+
- Windows / Linux / macOS
- OpenCV with video codec support

---

## Input Videos

Place the two input videos in the project root directory with the following names:

```text
sample1.mp4
sample2.mp4
```

### Expected characteristics

- `sample1.mp4`: about 30 seconds, coherent motion
- `sample2.mp4`: longer than 30 seconds, with visible motion in the selected 30-second segment

The pipeline assumes these filenames unless your individual scripts already use different ones.

---

## How to Run

### Run the full pipeline using `main.py`

This runs all existing scripts **one by one** without modifying them.

```bash
python main.py
```

### Run scripts manually

If you prefer step-by-step execution:

```bash
python video_confirm.py
python extract_frames.py
python optical_flow_analysis.py
python plot_motion_stats.py
python tracking_validation.py
```

### Run from another directory

If your scripts are in a different folder:

```bash
python main.py --dir "E:\\Study\\Second Sem\\CV\\project_5\\Optical Flow Analysis"
```

### Continue even if one script fails

```bash
python main.py --continue-on-error
```

---

## Script Descriptions

### `video_confirm.py`
Reads both videos and prints:

- FPS
- total frame count
- width
- height
- duration

This confirms that the videos are valid and suitable for the project.

### `extract_frames.py`
Extracts three preview frames from each selected 30-second segment:

- first frame
- middle frame
- last frame

These images are useful for:
- report illustrations
- dataset description
- visual evidence

### `optical_flow_analysis.py`
Computes **dense optical flow** using the **Farneback method**.

For each frame pair, it produces:
- original resized frame
- HSV-style dense flow visualization
- sparse arrow-based motion visualization

It also saves:
- optical flow video
- motion statistics CSV
- evidence frames for the report

### `plot_motion_stats.py`
Generates plots from the saved CSV files:

- **Mean Optical Flow Magnitude vs Time**
- **Moving Pixel Ratio vs Time**

These plots provide quantitative evidence of motion behavior in both videos.

### `tracking_validation.py`
Validates motion tracking using two consecutive frames.

It performs:
- feature selection using good corner detection
- Lucas–Kanade style tracking
- bilinear interpolation for subpixel intensity estimation
- exhaustive template matching to find an “actual” location
- error computation in pixels

Outputs include:
- validation table as CSV
- visualization image of tracked points
- mean and maximum tracking error

### `main.py`
A lightweight runner script that executes all existing project scripts in the correct order.

Important:
- it **does not modify** your current scripts
- it only automates sequential execution
- it stops on failure unless `--continue-on-error` is used

---

## Output Files

All generated outputs are saved in:

```text
cv_project_output/
```

### Preview frames
- `sample1_preview_1.jpg`
- `sample1_preview_2.jpg`
- `sample1_preview_3.jpg`
- `sample2_preview_1.jpg`
- `sample2_preview_2.jpg`
- `sample2_preview_3.jpg`

### Optical flow videos
- `sample1_optical_flow.mp4`
- `sample2_optical_flow.mp4`

### Motion statistics
- `sample1_flow_stats.csv`
- `sample2_flow_stats.csv`

### Motion plots
- `sample1_mean_flow_plot.png`
- `sample2_mean_flow_plot.png`
- `sample1_moving_ratio_plot.png`
- `sample2_moving_ratio_plot.png`

### Tracking validation
- `sample1_tracking_validation.csv`
- `sample2_tracking_validation.csv`
- `sample1_tracking_validation.jpg`
- `sample2_tracking_validation.jpg`

---

## Methodology

## Part A — Optical Flow Analysis

The project uses **Farneback dense optical flow** to estimate motion between consecutive frames.

### What is computed
For each frame pair:
- horizontal and vertical motion vectors
- motion magnitude
- motion direction

### How it is visualized
- **Color-coded flow map**
  - hue represents motion direction
  - intensity represents motion magnitude
- **Sparse arrow overlay**
  - arrows indicate approximate displacement direction and size

### What can be inferred
Optical flow helps reveal:
- regions with motion
- regions that remain static
- motion direction
- relative speed
- coherence of object motion
- non-rigid vs rigid motion behavior

---

## Part B — Motion Tracking Validation

Tracking is based on the classical **brightness constancy assumption**:

```math
I(x, y, t) = I(x + u, y + v, t+1)
```

Using first-order Taylor expansion:

```math
I_x u + I_y v + I_t = 0
```

Since one pixel gives one equation with two unknowns, a local window is used and solved in a least-squares sense, producing the Lucas–Kanade formulation.

### Bilinear interpolation
Because tracked points often land at non-integer coordinates, bilinear interpolation is used to estimate intensity values at subpixel locations.

### Validation strategy
The predicted point position from tracking is compared with the best local template match in the next frame, and the error is measured in pixels.

---

## Example Results from This Project

Based on the current experiment:

### Video confirmation
- `sample1.mp4`
  - FPS: 30.0
  - Frames: 901
  - Resolution: 1920×1080
  - Duration: 30.03 s

- `sample2.mp4`
  - FPS: 25.0
  - Frames: 1068
  - Resolution: 1920×1080
  - Duration: 42.72 s

### Optical flow summary
- `sample1`
  - Mean of mean flow magnitude: **0.211598**
  - Maximum observed flow magnitude: **5.672446**
  - Mean moving-pixel ratio: **0.177897**

- `sample2`
  - Mean of mean flow magnitude: **1.778561**
  - Maximum observed flow magnitude: **38.318985**
  - Mean moving-pixel ratio: **0.611745**

### Strongest motion moments
- `sample1`: frame **877**, time **29.233 s**
- `sample2`: frame **344**, time **13.760 s**

### Tracking validation summary
- `sample1`
  - Mean error: **0.382027 pixels**
  - Maximum error: **0.548023 pixels**

- `sample2`
  - Mean error: **0.438999 pixels**
  - Maximum error: **0.648365 pixels**

These results indicate that:
- `sample1` has relatively smooth, coherent motion
- `sample2` contains stronger and more irregular motion
- tracking accuracy remains subpixel to near-subpixel in both videos

---

## Interpretation of the Two Videos

### `sample1.mp4`
This video shows a rotating Earth against a mostly dark background.

Expected observations:
- motion concentrated on the globe
- little motion in the background
- coherent directional flow over the visible Earth surface
- relatively low average motion magnitude

### `sample2.mp4`
This video contains a bird, rock, and moving water.

Expected observations:
- strong motion in the water region
- low motion in more static regions such as the bird or rock
- more irregular and non-rigid motion patterns
- higher motion magnitude and moving-pixel ratio

---

## Report Files

This project also includes:

- `report.tex` — LaTeX report
- `references.bib` — bibliography database

To compile the report:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Make sure the `cv_project_output/` folder exists before compiling, because the report references generated figures and result images.

---

## Troubleshooting

### Video not opening
Check:
- file names are exactly `sample1.mp4` and `sample2.mp4`
- videos are in the correct folder
- OpenCV is installed correctly

### No output video generated
Possible reasons:
- codec issue with your environment
- missing write permission
- invalid video path

### Tracking validation returns empty results
Possible reasons:
- insufficient texture in selected region
- ROI excludes good features
- selected frame pair has poor corners

### Plots do not appear
If running in a terminal-only environment, figures may still be saved even if interactive display is limited. Check the output folder.

---

## Academic Notes

If you are using this project for coursework or submission:

- cite all external sources used in the report
- include both theory and implementation
- attach representative figures and plots
- discuss limitations such as:
  - occlusion
  - brightness changes
  - motion blur
  - textureless regions
  - non-rigid motion

---

## References

Suggested references used in the theory/report:

1. B. D. Lucas and T. Kanade, *An Iterative Image Registration Technique with an Application to Stereo Vision*, 1981.
2. B. K. P. Horn and B. G. Schunck, *Determining Optical Flow*, 1981.
3. Gunnar Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, 2003.
4. Richard Szeliski, *Computer Vision: Algorithms and Applications*.
5. OpenCV Documentation.

---

## Author Notes

This repository is structured so the full workflow can be executed in a clean, reproducible order:
- verify inputs
- extract evidence frames
- compute optical flow
- generate plots
- validate tracking
- include outputs in the final report

If needed, you can further improve the project by adding:
- argument parsing for video paths
- automatic report generation
- GUI/video preview tools
- support for additional optical flow methods

---

## License

This project is intended for educational and academic use.
