# Stereo Vision — Depth Estimation

> Disparity map computation from rectified stereo image pairs using block matching — implemented from scratch without OpenCV's built-in stereo solvers.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy&logoColor=white)

---

## What is Stereo Vision?

Human eyes perceive depth because each eye sees the world from a slightly different angle. Your brain computes the **disparity** — the horizontal offset between corresponding points in each eye — and converts it to a depth estimate. Objects closer to you have a larger disparity; distant objects have a smaller one.

Stereo vision replicates this computationally:

```
Left camera          Right camera
     \                   /
      \       Scene      /
       \                /
        [   disparity  ]
                ↓
           Depth map
```

Given two **rectified** camera images (left and right), we find matching regions and measure how far each region has shifted horizontally. That shift is the **disparity `d`**, and its inverse is proportional to depth `Z`:

```
Z = (f × B) / d
```

where `f` = focal length (px) and `B` = baseline (distance between cameras in meters).

---

## Algorithm: Block Matching

The key insight of block matching is simple: instead of comparing single pixels (which are noisy), compare *patches* — small rectangular windows centred on each pixel. Patches have enough texture to be uniquely matchable.

### Step-by-step

```
For each pixel (x, y) in the left image:
  1. Extract patch P_L of size (block_size × block_size) centred at (x, y)
  2. For each candidate disparity d = 0, 1, …, max_disparity-1:
       Extract patch P_R centred at (x - d, y) in the right image
       cost[d] = SAD(P_L, P_R)   ← sum |P_L[i,j] - P_R[i,j]|
  3. Best disparity = argmin(cost)
```

The **epipolar constraint** (from rectification) means the search is always on the **same row** — this reduces the problem from 2D to 1D, making it tractable.

### Cost functions

| Name | Formula | Notes |
|---|---|---|
| **SAD** | `Σ |L - R|` | Fast, works well in practice |
| **SSD** | `Σ (L - R)²` | Penalises large errors more; slightly more robust |

### Efficient implementation with a cost volume

Rather than running a nested loop per pixel (slow in Python), the implementation builds a **cost volume** of shape `(H, W, max_disparity)`:

```python
cost_volume[y, x, d] = SAD of the block centred at (x, y) at disparity d
```

For each `d`, the right image is shifted by `d` columns and a 2D box filter computes the block sum in `O(H × W)` — far faster than a naive `O(H × W × block² × max_disp)` loop.

The final disparity map is just `argmin(cost_volume, axis=2)`.

---

## Project Structure

```
stereo-vision/
├── main.py          ← CLI entry point (parse args, run pipeline, save results)
├── stereo.py        ← StereoMatcher class (cost volume, block matching, subpixel)
├── utils.py         ← Image I/O, colormap visualisation, Middlebury evaluation
├── requirements.txt
└── images/          ← Put your stereo pairs here
    ├── left.png
    └── right.png
```

---

## Getting Started

### Install

```bash
git clone https://github.com/ahmadrkh/Stereo-Vision.git
cd Stereo-Vision
pip install -r requirements.txt
```

### Run

```bash
# Basic
python main.py --left images/left.png --right images/right.png

# With all options
python main.py \
    --left  images/left.png \
    --right images/right.png \
    --block-size 11 \
    --max-disparity 64 \
    --metric sad \
    --scale 0.5 \
    --output output/disparity.png
```

### Evaluate against ground truth (Middlebury format)

```bash
python main.py \
    --left  data/Tsukuba/left.png \
    --right data/Tsukuba/right.png \
    --gt    data/Tsukuba/disp_left.png \
    --output output/tsukuba.png
```

Output:
```
[1/4] Loading stereo pair …        Left: … Right: … Size: 384×288 px
[2/4] Running block matching …     Done in 4.2s
[3/4] Evaluating against GT …      MAE: 1.83px  RMSE: 3.41px  Bad3%: 12.4%
[4/4] Saving and displaying …      Saved → output/tsukuba.png
```

---

## Parameters

| Flag | Default | Effect |
|---|---|---|
| `--block-size` | 11 | Patch size. Larger → smoother but loses fine detail |
| `--max-disparity` | 64 | Search range. Increase for closer objects or wider baselines |
| `--metric` | `sad` | `sad` or `ssd` — see cost functions above |
| `--scale` | 1.0 | Resize factor. `0.5` halves resolution, runs ~4× faster |
| `--no-show` | False | Skip matplotlib window (useful for scripts) |

### Tuning guidance

- **Texture-rich scene** → smaller `block_size` (9) captures more detail
- **Smooth surfaces** → larger `block_size` (15) reduces noise
- **Close objects** → increase `max_disparity` (128)
- **Slow runtime** → reduce `--scale 0.5` or reduce `--max-disparity`

---

## Output

The pipeline saves two files:

- `disparity.png` — colour-mapped disparity (PLASMA colourmap; brighter = closer)
- `disparity_raw16.png` — 16-bit grayscale for downstream processing (normalised to 0–65535)

---

## Limitations & Future Work

| Limitation | Cause | Possible fix |
|---|---|---|
| Slow on large images | Pure NumPy; no CUDA | Port inner loop to C++ or use cuPy |
| Noisy in low-texture regions | Not enough gradient for matching | Add uniqueness check or semi-global matching (SGM) |
| No occlusion handling | WTA doesn't model occluded pixels | Left-right consistency check |
| Integer disparity only | WTA gives integer output | Subpixel parabolic interpolation (implemented, optional) |

---

## Datasets for testing

| Dataset | Resolution | Notes |
|---|---|---|
| [Middlebury Stereo](https://vision.middlebury.edu/stereo/data/) | Various | Ground truth included, standard benchmark |
| [KITTI Stereo](https://www.cvlibs.net/datasets/kitti/eval_stereo.php) | 1242×375 | Autonomous driving scenes, LiDAR GT |
| [ETH3D](https://www.eth3d.net/stereo_overview) | Various | Indoor + outdoor, high-res GT |

---

## What I Learned

- How the **epipolar constraint** reduces 2D correspondence to 1D search
- Why **cost volumes** are more efficient than per-pixel loops
- The trade-off between **block size and accuracy** in dense matching
- How to measure quality with standard metrics (MAE, RMSE, Bad-3%)
- Why **rectification is a prerequisite** — without it, the epipolar lines aren't horizontal and the 1D search assumption breaks

---

## References

- Szeliski, *Computer Vision: Algorithms and Applications* — Ch. 12 (Stereo Correspondence)
- Hirschmüller (2008). *Stereo Processing by Semi-Global Matching and Mutual Information*. TPAMI.
- [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/)

---

## Context

Implemented as a course project for the Computer Vision class at **Sharif University of Technology**, Faculty of Computer Engineering. The constraint was to build the stereo matching pipeline from scratch — without using `cv2.StereoBM` or `cv2.StereoSGBM` — to deeply understand the algorithm before using library implementations.
