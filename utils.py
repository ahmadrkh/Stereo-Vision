"""
utils.py — Helper functions for loading, preprocessing, and visualising results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Image loading ──────────────────────────────────────────────────────────────

def load_stereo_pair(
    left_path: str,
    right_path: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a rectified stereo pair as grayscale images.

    Parameters
    ----------
    left_path, right_path : str
        Paths to the left and right images. Any format OpenCV supports works.
    scale : float
        Resize factor. 0.5 → half resolution (faster, less accurate).
        Default 1.0 keeps original resolution.

    Returns
    -------
    left, right : np.ndarray
        Grayscale uint8 images of identical shape (H, W).
    """
    left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if left is None:
        raise FileNotFoundError(f"Could not load left image: {left_path}")
    if right is None:
        raise FileNotFoundError(f"Could not load right image: {right_path}")
    if left.shape != right.shape:
        raise ValueError(
            f"Image shapes don't match: left={left.shape}, right={right.shape}. "
            "Make sure the stereo pair is rectified."
        )

    if scale != 1.0:
        new_w = int(left.shape[1] * scale)
        new_h = int(left.shape[0] * scale)
        left  = cv2.resize(left,  (new_w, new_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return left, right


# ── Visualisation ──────────────────────────────────────────────────────────────

def disparity_to_colormap(
    disparity: np.ndarray,
    max_disparity: int,
    colormap: int = cv2.COLORMAP_PLASMA,
) -> np.ndarray:
    """
    Convert a float32 disparity map to a colour image for visualisation.

    Normalises disparity to 0–255, then applies a colourmap:
      - PLASMA  → purple→yellow (perceptually uniform, good default)
      - JET     → blue→red (classic, but can mislead the eye)
      - MAGMA   → black→yellow (good on dark backgrounds)

    Returns BGR uint8 image (OpenCV format).
    """
    # Normalise to 0–255
    disp_norm = np.clip(disparity / max_disparity, 0, 1)
    disp_uint8 = (disp_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(disp_uint8, colormap)


def show_results(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    max_disparity: int,
    save_path: str | None = None,
) -> None:
    """
    Display a 3-panel figure: left image | right image | disparity map.

    Parameters
    ----------
    save_path : str, optional
        If provided, saves the figure to this path instead of (or in addition to) showing it.
    """
    disparity_color = disparity_to_colormap(disparity, max_disparity)
    disparity_rgb   = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Stereo Matching Results", fontsize=14, fontweight='bold')

    axes[0].imshow(left,  cmap='gray'); axes[0].set_title("Left Image");    axes[0].axis('off')
    axes[1].imshow(right, cmap='gray'); axes[1].set_title("Right Image");   axes[1].axis('off')
    axes[2].imshow(disparity_rgb);      axes[2].set_title("Disparity Map"); axes[2].axis('off')

    # Colourbar showing disparity scale
    sm = plt.cm.ScalarMappable(
        cmap='plasma',
        norm=plt.Normalize(vmin=0, vmax=max_disparity)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Disparity (px)', fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure → {save_path}")

    plt.show()


def save_disparity(disparity: np.ndarray, max_disparity: int, path: str) -> None:
    """Save the disparity map as a PNG (colourmap) and as a 16-bit grayscale."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Colour PNG for human viewing
    color = disparity_to_colormap(disparity, max_disparity)
    cv2.imwrite(path, color)

    # 16-bit grayscale for downstream processing (preserves float precision)
    gray_16 = (disparity / max_disparity * 65535).astype(np.uint16)
    gray_path = path.replace('.png', '_raw16.png')
    cv2.imwrite(gray_path, gray_16)

    print(f"  Saved colour disparity → {path}")
    print(f"  Saved raw 16-bit      → {gray_path}")


# ── Evaluation (Middlebury) ────────────────────────────────────────────────────

def compute_error_metrics(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    max_disparity: int,
    bad_thresh: float = 3.0,
) -> dict:
    """
    Compute standard stereo evaluation metrics against a ground truth disparity map.
    Used with Middlebury-style datasets.

    Metrics:
      - MAE  : Mean Absolute Error (average pixel error)
      - RMSE : Root Mean Squared Error
      - Bad-N%: percentage of pixels with |error| > bad_thresh pixels
               (Middlebury uses 3.0 by default)

    Only evaluates pixels where ground_truth > 0 (valid GT pixels).
    """
    valid = ground_truth > 0
    if not valid.any():
        return {}

    est  = estimated[valid].astype(np.float32)
    gt   = ground_truth[valid].astype(np.float32)
    err  = np.abs(est - gt)

    return {
        'mae':     float(err.mean()),
        'rmse':    float(np.sqrt((err ** 2).mean())),
        f'bad_{bad_thresh}': float((err > bad_thresh).mean() * 100),
        'valid_px': int(valid.sum()),
    }
