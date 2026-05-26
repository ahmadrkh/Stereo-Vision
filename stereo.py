"""
stereo.py — Core stereo matching algorithm.

Stereo matching answers one question:
  "For a pixel at (x, y) in the left image, where is the same point in the right image?"

Because the cameras are rectified (aligned horizontally), the matching point in the
right image is always on the SAME row — just shifted left by some amount `d` (disparity).
This constraint is called the epipolar constraint and it reduces the search from 2D → 1D.

The disparity `d` at pixel (x, y) is directly related to depth Z:
    Z = (focal_length × baseline) / d
where baseline is the horizontal distance between the two cameras.
"""

import numpy as np
import cv2
from typing import Literal


class StereoMatcher:
    """
    Dense stereo matcher using block matching (SAD or SSD).

    Block matching is the simplest dense stereo algorithm:
    - For every pixel, compare a small patch in the left image
      against candidate patches in the right image along the same row.
    - The offset with the lowest matching cost = disparity.

    Parameters
    ----------
    block_size : int
        Side length of the square matching window (must be odd, e.g. 9, 11, 15).
        Larger → smoother disparity map, less detail.
        Smaller → more detail, but noisier.
    max_disparity : int
        Maximum pixel shift to search. Set this based on how close the nearest
        objects are and how far apart the cameras are.
        Rule of thumb: if the nearest object is ~1m away and baseline is ~0.5m,
        max_disparity ≈ 64–128.
    metric : 'sad' | 'ssd'
        SAD = Sum of Absolute Differences — fast, works well in practice.
        SSD = Sum of Squared Differences — penalises large mismatches more,
              slightly more robust to noise.
    """

    def __init__(
        self,
        block_size: int = 11,
        max_disparity: int = 64,
        metric: Literal['sad', 'ssd'] = 'sad',
    ):
        if block_size % 2 == 0:
            raise ValueError("block_size must be odd (e.g. 9, 11, 15)")
        self.block_size = block_size
        self.max_disparity = max_disparity
        self.metric = metric

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute the disparity map for a rectified stereo pair.

        Parameters
        ----------
        left, right : np.ndarray
            Rectified grayscale images, same shape (H, W), dtype uint8.

        Returns
        -------
        disparity : np.ndarray
            Float32 array of shape (H, W). Value at [y, x] is the horizontal
            shift (in pixels) that maps pixel x in the left image to the right image.
            Zero means no match was found (or object is very far away).
        """
        # Convert to float32 once — avoids repeated conversions inside the loop
        left_f  = left.astype(np.float32)
        right_f = right.astype(np.float32)

        H, W = left_f.shape
        half = self.block_size // 2  # half-width of the patch

        # Output: disparity map, initialised to 0 (no match)
        disparity = np.zeros((H, W), dtype=np.float32)

        # For each pixel (x, y), we search disparities 0 … max_disparity-1.
        # We store the cost for each candidate disparity in a 3D array:
        #   cost_volume[y, x, d] = matching cost at disparity d
        # This is more cache-friendly than computing per-pixel inside a Python loop.
        cost_volume = np.full(
            (H, W, self.max_disparity), fill_value=np.inf, dtype=np.float32
        )

        print(f"  Computing cost volume ({self.metric.upper()}, "
              f"block={self.block_size}×{self.block_size}, "
              f"max_disp={self.max_disparity}) …")

        for d in range(self.max_disparity):
            # Shift the right image d pixels to the right using a column slice.
            # After shifting: right_shifted[:, x] corresponds to right[:, x - d]
            # which is the candidate match for left[:, x] at disparity d.
            if d == 0:
                right_shifted = right_f
            else:
                # Pad left with zeros, crop right — equivalent to shifting right→left
                right_shifted = np.zeros_like(right_f)
                right_shifted[:, d:] = right_f[:, :-d]

            # Compute per-pixel difference image
            diff = left_f - right_shifted        # shape (H, W)

            if self.metric == 'sad':
                pixel_cost = np.abs(diff)        # |L - R|
            else:  # ssd
                pixel_cost = diff ** 2           # (L - R)²

            # Sum over the block_size × block_size neighbourhood using a box filter.
            # cv2.boxFilter with normalize=False sums (doesn't average) the values
            # in the kernel — this is equivalent to a 2D sliding window sum,
            # but runs in O(H*W) time instead of O(H*W*block²).
            block_cost = cv2.boxFilter(
                pixel_cost, ddepth=-1,
                ksize=(self.block_size, self.block_size),
                normalize=False,
            )
            cost_volume[:, :, d] = block_cost

            if d % 16 == 0:
                print(f"    d = {d:3d} / {self.max_disparity}")

        # Winner-Takes-All: the disparity at each pixel is whichever d gave
        # the minimum cost. np.argmin along axis=2 returns the index (= disparity).
        print("  Finding best disparity (argmin over cost volume) …")
        disparity = np.argmin(cost_volume, axis=2).astype(np.float32)

        # Mask the border region (block_size//2 pixels on each side) where
        # the box filter result is unreliable because the patch extends outside
        # the image boundary.
        disparity[:half, :]  = 0
        disparity[-half:, :] = 0
        disparity[:, :half]  = 0
        disparity[:, -half:] = 0

        return disparity

    # ──────────────────────────────────────────────────────────────────────────
    # Optional: subpixel refinement
    # ──────────────────────────────────────────────────────────────────────────

    def refine_subpixel(self, disparity: np.ndarray, cost_volume: np.ndarray) -> np.ndarray:
        """
        Parabolic subpixel interpolation.

        The integer WTA disparity is only accurate to 1 pixel. We can get
        sub-pixel precision by fitting a parabola through the three cost values
        around the minimum: cost[d-1], cost[d], cost[d+1].
        The parabola minimum is at:
            d_sub = d - (cost[d+1] - cost[d-1]) / (2 * (cost[d+1] - 2*cost[d] + cost[d-1]))

        This is optional and adds modest quality improvement.
        """
        H, W = disparity.shape
        disp_int = disparity.astype(np.int32)
        disp_sub = disparity.copy()

        for y in range(H):
            for x in range(W):
                d = disp_int[y, x]
                if 1 <= d < self.max_disparity - 1:
                    c_prev = cost_volume[y, x, d - 1]
                    c_curr = cost_volume[y, x, d]
                    c_next = cost_volume[y, x, d + 1]
                    denom = c_prev - 2 * c_curr + c_next
                    if denom > 1e-6:
                        disp_sub[y, x] = d - (c_next - c_prev) / (2 * denom)
        return disp_sub
