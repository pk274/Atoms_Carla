#!/usr/bin/env python3
"""
gather_baseline.py
------------------
Combine partial .npz files produced by compute_baseline_chunk.py into a
single baseline.npz compatible with run_analysis.py, and a separate
mdx_features.npz used by run_analysis.py to fit MDXDetector locally.

Writes baseline.npz:
    series      : [N_total, num_classes]     float32
    mean        : [num_classes]              float32
    cov         : [num_classes, num_classes] float32
    class_ids   : [num_classes]              int32
    class_names : [num_classes]              object
    cmd_filter  : [1]                        int32  (always -1, no filter)
    n_frames    : [1]                        int32

Writes mdx_features.npz (alongside baseline.npz unless --mdx-output is set):
    features    : [N_total, 512]  float32  — globally-pooled backbone activations
    actions     : [N_total, 3]    float32  — [steer=0, throttle, brake] proxy

Usage:
    python hpc/gather_baseline.py \
        --partials-dir /ptmp/$USER/atoms_baseline/partials \
        --output       /ptmp/$USER/atoms_baseline/baseline.npz
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--partials-dir", required=True, type=Path,
                   help="Directory containing partial_*.npz files from array tasks.")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output path for baseline.npz.")
    p.add_argument("--mdx-output",   default=None,  type=Path,
                   help="Output path for mdx_features.npz. "
                        "Defaults to <output-dir>/mdx_features.npz.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mdx_out = args.mdx_output or (args.output.parent / "mdx_features.npz")

    partial_files = sorted(args.partials_dir.glob("partial_*.npz"))
    if not partial_files:
        raise FileNotFoundError(
            f"No partial_*.npz files found in {args.partials_dir}\n"
            f"Check that all array tasks completed successfully (squeue / sacct)."
        )
    print(f"[gather] Found {len(partial_files)} partial files.")

    series_parts   = []
    backbone_parts = []
    actions_parts  = []
    class_ids      = None
    class_names    = None
    has_mdx        = True   # set False if any partial is missing backbone_features

    for f in partial_files:
        part = np.load(f, allow_pickle=True)
        series_parts.append(part["series"])
        if class_ids is None:
            class_ids   = part["class_ids"]
            class_names = part["class_names"]
        if "backbone_features" in part and "mdx_actions" in part:
            backbone_parts.append(part["backbone_features"])
            actions_parts.append(part["mdx_actions"])
        else:
            has_mdx = False
            print(f"  WARNING: {f.name} missing backbone_features/mdx_actions — "
                  f"mdx_features.npz will not be written.")
        print(f"  {f.name}: {part['series'].shape}")

    series = np.concatenate(series_parts, axis=0)   # [N_total, num_classes]
    print(f"[gather] Total series: {series.shape}")

    mean = series.mean(axis=0)
    cov  = np.cov(series.T)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        series      = series.astype(np.float32),
        mean        = mean.astype(np.float32),
        cov         = cov.astype(np.float32),
        class_ids   = class_ids,
        class_names = class_names,
        cmd_filter  = np.array([-1], dtype=np.int32),
        n_frames    = np.array([len(series)], dtype=np.int32),
    )
    print(f"[gather] baseline.npz saved → {args.output}")
    print(f"  series : {series.shape}")
    print(f"  mean   : {mean.shape}")
    print(f"  cov    : {cov.shape}")
    print(f"  classes: {list(class_names)}")

    if has_mdx:
        features = np.concatenate(backbone_parts, axis=0)  # [N_total, 512]
        actions  = np.concatenate(actions_parts,  axis=0)  # [N_total, 3]
        mdx_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mdx_out,
            features = features,
            actions  = actions,
        )
        print(f"[gather] mdx_features.npz saved → {mdx_out}")
        print(f"  features : {features.shape}")
        print(f"  actions  : {actions.shape}")


if __name__ == "__main__":
    main()
