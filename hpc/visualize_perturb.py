#!/usr/bin/env python3
"""
visualize_perturb.py
--------------------
Produce a summary figure showing n_samples example frames for every
perturbation type stored in test_labeled.npz.  Each subplot shows the
centre camera extracted from the concatenated multi-camera strip.

Called by gather_test_task.sh after the profile gather, so the figure
is ready for collect_results.sh to copy back into the repo.

Usage (standalone):
    python hpc/visualize_perturb.py \\
        --labeled-file /ptmp/$USER/atoms_test/test_labeled.npz \\
        --output       /ptmp/$USER/atoms_test/perturb_samples.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled-file", required=True, type=Path,
                   help="Path to test_labeled.npz.")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output PNG path.")
    p.add_argument("--n-samples",    default=3, type=int,
                   help="Sample images per perturbation type (default 3).")
    p.add_argument("--n-cameras",    default=6, type=int,
                   help="Cameras concatenated in the wide RGB strip (default 6 for TFV6).")
    p.add_argument("--camera-idx",   default=None, type=int,
                   help="Which camera to display (0-indexed). Default: middle camera.")
    p.add_argument("--seed",         default=42, type=int)
    return p.parse_args()


def _extract_camera(img_chw: np.ndarray, cam_idx: int, n_cams: int) -> np.ndarray:
    """[3, H, W_total] → [H, W_per_cam, 3] uint8 for one camera."""
    w_cam = img_chw.shape[2] // n_cams
    crop  = img_chw[:, :, cam_idx * w_cam : (cam_idx + 1) * w_cam]
    return np.transpose(crop, (1, 2, 0))   # HWC


def _row_label(pert: str, eps: float) -> str:
    if pert == "clean":
        return "clean"
    if pert == "camera_loss":
        return f"camera_loss\n(cam 0 dropped)"
    if pert == "pgd":
        return f"pgd\n(ε = {eps:.1f}, deferred)"
    return f"{pert}\n(ε = {eps:.1f})"


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    data        = np.load(args.labeled_file, allow_pickle=True)
    wide_rgb    = data["wide_rgb"]       # [N, 3, H, W_total]
    pert_labels = data["perturbation"]   # [N] object array of strings
    intensities = (data["intensity"]
                   if "intensity" in data
                   else np.zeros(len(pert_labels), dtype=np.float32))

    cam_idx = args.camera_idx if args.camera_idx is not None else args.n_cameras // 2
    print(f"[visualize_perturb] camera {cam_idx}/{args.n_cameras - 1}  "
          f"  labeled file: {args.labeled_file}")

    # Ordered: clean first, then alphabetical
    all_types  = sorted(np.unique(pert_labels).tolist())
    pert_types = (["clean"] if "clean" in all_types else []) + [
        t for t in all_types if t != "clean"
    ]
    n_rows = len(pert_types)
    n_cols = args.n_samples

    cell_sz = 2.2   # inches per square subplot
    pad_l   = 1.6   # inches for row labels
    fig_w   = pad_l + n_cols * cell_sz
    fig_h   = 0.55  + n_rows * (cell_sz + 0.50)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.60, "wspace": 0.04},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, pert in enumerate(pert_types):
        mask   = np.where(pert_labels == pert)[0]
        n_draw = min(n_cols, len(mask))
        chosen = sorted(rng.choice(mask, size=n_draw, replace=False).tolist())
        eps    = float(intensities[chosen[0]]) if chosen else 0.0

        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(chosen):
                continue

            img = _extract_camera(wide_rgb[chosen[col]], cam_idx, args.n_cameras)
            ax.imshow(img, interpolation="bilinear")

            subtitle = f"#{chosen[col]}"
            if pert == "pgd":
                subtitle += "\n(clean pixels)"
            ax.set_title(subtitle, fontsize=6, pad=2)

        # Row label: annotated outside the left edge of column 0
        axes[row, 0].annotate(
            _row_label(pert, eps),
            xy=(0, 0.5),
            xycoords=axes[row, 0].get_yaxis_transform(),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            linespacing=1.4,
        )

    fig.suptitle(
        f"Perturbation samples  ·  camera {cam_idx} of {args.n_cameras}"
        f"  ·  n = {n_cols} per type",
        fontsize=9, y=1.01,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_perturb] Saved {n_rows}×{n_cols} grid → {args.output}")


if __name__ == "__main__":
    main()
