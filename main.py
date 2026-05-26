"""
main.py — Entry point for the stereo matching pipeline.

Usage
-----
Basic:
    python main.py --left images/left.png --right images/right.png

With options:
    python main.py \\
        --left  images/left.png \\
        --right images/right.png \\
        --block-size 11 \\
        --max-disparity 64 \\
        --metric sad \\
        --scale 0.5 \\
        --output output/disparity.png \\
        --no-show

With ground truth evaluation (Middlebury format):
    python main.py \\
        --left  data/Tsukuba/left.png \\
        --right data/Tsukuba/right.png \\
        --gt    data/Tsukuba/disp_left.png \\
        --output output/tsukuba_disp.png
"""

import argparse
import time
import cv2

from stereo import StereoMatcher
from utils import (
    load_stereo_pair,
    show_results,
    save_disparity,
    compute_error_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stereo depth estimation via block matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--left",           required=True,  help="Path to left rectified image")
    p.add_argument("--right",          required=True,  help="Path to right rectified image")
    p.add_argument("--gt",             default=None,   help="Path to ground-truth disparity (optional)")
    p.add_argument("--output",         default="output/disparity.png", help="Where to save the disparity map")
    p.add_argument("--block-size",     type=int,   default=11,    help="Matching window size (odd number)")
    p.add_argument("--max-disparity",  type=int,   default=64,    help="Maximum disparity to search")
    p.add_argument("--metric",         choices=["sad", "ssd"], default="sad", help="Matching cost function")
    p.add_argument("--scale",          type=float, default=1.0,   help="Image resize factor (0.5 = half size)")
    p.add_argument("--no-show",        action="store_true",       help="Skip the matplotlib display window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load images ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading stereo pair …")
    print(f"      Left : {args.left}")
    print(f"      Right: {args.right}")
    if args.scale != 1.0:
        print(f"      Scale: {args.scale}×")

    left, right = load_stereo_pair(args.left, args.right, scale=args.scale)
    H, W = left.shape
    print(f"      Image size: {W}×{H} px")

    # ── 2. Run stereo matching ────────────────────────────────────────────────
    print(f"\n[2/4] Running block matching …")
    print(f"      Metric       : {args.metric.upper()}")
    print(f"      Block size   : {args.block_size}×{args.block_size}")
    print(f"      Max disparity: {args.max_disparity} px")

    matcher = StereoMatcher(
        block_size=args.block_size,
        max_disparity=args.max_disparity,
        metric=args.metric,
    )

    t0 = time.perf_counter()
    disparity = matcher.compute(left, right)
    elapsed = time.perf_counter() - t0

    print(f"      Done in {elapsed:.1f}s")
    print(f"      Disparity range: {disparity.min():.1f} – {disparity.max():.1f} px")

    # ── 3. Optional: evaluate against ground truth ────────────────────────────
    if args.gt:
        print(f"\n[3/4] Evaluating against ground truth …")
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE).astype(float)

        # Middlebury disparity maps are stored at 4× scale by convention
        gt = gt / 4.0

        metrics = compute_error_metrics(disparity, gt, args.max_disparity)
        print(f"      MAE  : {metrics['mae']:.3f} px")
        print(f"      RMSE : {metrics['rmse']:.3f} px")
        print(f"      Bad3%: {metrics['bad_3.0']:.2f}%  (pixels with |error| > 3px)")
        print(f"      Valid pixels evaluated: {metrics['valid_px']:,}")
    else:
        print(f"\n[3/4] Skipping evaluation (no --gt provided)")

    # ── 4. Save and show results ──────────────────────────────────────────────
    print(f"\n[4/4] Saving and displaying results …")
    save_disparity(disparity, args.max_disparity, args.output)

    if not args.no_show:
        show_results(
            left, right, disparity,
            max_disparity=args.max_disparity,
            save_path=args.output.replace('.png', '_figure.png'),
        )

    print("\nDone ✓\n")


if __name__ == "__main__":
    main()
