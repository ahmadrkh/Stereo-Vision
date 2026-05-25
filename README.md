# Stereo Vision — Depth Estimation

> Estimating scene depth from a pair of stereo images using block matching and disparity map computation.

## What is Stereo Vision?

Human eyes perceive depth because each eye sees the world from a slightly different angle. Your brain computes the **disparity** (horizontal offset) between corresponding points in each eye's image and converts it to a depth estimate — closer objects have larger disparity.

Stereo vision replicates this: given two rectified camera images (left and right), we find matching regions and measure how far they've shifted. This shift is the **disparity**; its inverse is proportional to depth.

## Project Overview

This project implements a stereo matching pipeline from scratch:

1. **Input** — a rectified stereo image pair (left + right)
2. **Block Matching** — for each pixel in the left image, find the best-matching block in the right image along the same horizontal line (epipolar constraint)
3. **Disparity Map** — the horizontal offset for each matched pixel, visualized as a grayscale or color-mapped image
4. **Depth Estimation** — convert disparity to real-world depth using camera baseline and focal length

## Sample Output

```
Left Image      Right Image     Disparity Map
[camera L]  +  [camera R]  →  [depth gradient]
```

> Brighter pixels = closer to the camera · Darker pixels = farther away

*(Add your own output images here: `docs/disparity_output.png`)*

## Tech Stack

| Tool | Role |
|---|---|
| Python 3 | Core language |
| OpenCV | Image I/O, preprocessing, visualization |
| NumPy | Array operations, sliding window computation |
| Matplotlib | Plotting disparity maps |

## Getting Started

### Prerequisites

```bash
pip install opencv-python numpy matplotlib
```

### Run

```bash
# Clone the repo
git clone https://github.com/ahmadrkh/Stereo-Vision.git
cd Stereo-Vision

# Run the main script
python main.py --left images/left.png --right images/right.png
```

The disparity map will be saved to `output/disparity.png`.

## How Block Matching Works

```
For each pixel (x, y) in the left image:
  1. Extract a patch of size (block_size × block_size) centered at (x, y)
  2. Slide that patch along row y in the right image (search range: 0 to max_disparity)
  3. Compute similarity (SAD or SSD) at each offset d
  4. The offset with minimum cost = disparity at pixel (x, y)
```

**SAD** (Sum of Absolute Differences) — fast, simple, works well for texture-rich scenes.

**SSD** (Sum of Squared Differences) — penalizes large mismatches more, slightly more robust.

## Parameters

| Parameter | Description | Typical Value |
|---|---|---|
| `block_size` | Patch size for matching | 9–15 px |
| `max_disparity` | Maximum pixel shift to search | 64–128 px |
| `metric` | Matching cost: `'sad'` or `'ssd'` | `'sad'` |

Larger `block_size` → smoother map, but loses fine detail.  
Smaller `block_size` → more detail, but noisier.

## Project Context

Built as part of the **Computer Vision** course at Sharif University of Technology.  
The goal was to implement the stereo matching pipeline from scratch — without using OpenCV's built-in `StereoBM` or `StereoSGBM` — to deeply understand the underlying algorithm.

## What I Learned

- How the epipolar constraint reduces 2D search to 1D (row-wise)
- Trade-offs between block size and accuracy in dense matching
- Why rectification is a prerequisite for efficient stereo matching
- Visualizing and interpreting disparity maps

## References

- Szeliski, *Computer Vision: Algorithms and Applications* — Chapter 12 (Stereo Correspondence)
- Middlebury Stereo Evaluation Dataset
