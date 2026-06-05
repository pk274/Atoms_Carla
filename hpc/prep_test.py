#!/usr/bin/env python3
"""
prep_test.py
------------
Apply perturbations to clean test frames → test_labeled.npz.
No model required: the image-space perturbations are applied here, and pgd
frames are only *recorded* (clean pixels + label=1 + perturbation="pgd") — the
adversarial attack itself is crafted downstream in compute_test_chunk.py, which
has the TFV6 model loaded.

Called by prep_test_task.sh as a single-node job before the array job.

Usage (standalone):
    python hpc/prep_test.py \
        --frames-dir  /ptmp/$USER/atoms_test/frames \
        --output      /ptmp/$USER/atoms_test/test_labeled.npz \
        --seed        42 \
        --noise-intensity 21 \
        --brightness-intensity 4
"""

import argparse
from pathlib import Path

import numpy as np
import torch


# TFV6 perturbation spec — 5-way 20 % split (matches the WoR mix).
#
# PGD is an adversarial attack that needs a model forward/backward pass, but this
# prep step is deliberately model-free (CPU node).  So pgd frames are *recorded*
# here with clean pixels + label=1 + perturbation="pgd"; the actual attack is
# crafted later in compute_test_chunk.py, which has the TFV6 model on GPU.
_SPEC = [
    # (perturbation_or_None, intensity_default, fraction)
    (None,                0.0, 0.20),
    ("gaussian_noise",   21.0, 0.20),   # overridden by --noise-intensity
    ("brightness_scale",  4.0, 0.20),   # overridden by --brightness-intensity
    ("camera_loss",       0.0, 0.20),
    ("pgd",              12.0, 0.20),   # intensity = ℓ∞ ε budget (0–255); overridden by --pgd-epsilon
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir",           required=True, type=Path,
                   help="Directory containing clean run_*.npz test frame files.")
    p.add_argument("--output",               required=True, type=Path,
                   help="Output path for test_labeled.npz.")
    p.add_argument("--seed",                 default=42,   type=int)
    p.add_argument("--noise-intensity",      default=21.0, type=float)
    p.add_argument("--brightness-intensity", default=4.0,  type=float)
    p.add_argument("--pgd-epsilon",          default=12.0, type=float,
                   help="ℓ∞ ε budget recorded for pgd frames; the attack itself "
                        "is crafted in compute_test_chunk.py using this value.")
    return p.parse_args()


def load_all_runs(frames_dir: Path) -> dict:
    files = sorted(frames_dir.glob("run_*.npz"))
    if not files:
        raise FileNotFoundError(f"No run_*.npz files found in {frames_dir}")
    print(f"[prep] Found {len(files)} run files.")

    parts = []
    for run_id, fp in enumerate(files):
        d = np.load(fp, allow_pickle=False)
        n = d["wide_rgb"].shape[0]
        parts.append({
            "wide_rgb":     d["wide_rgb"],
            "seg_red_wide": d["seg_red_wide"],
            "cmd":          d["cmd"],
            "speed":        d["speed"],
            "is_brake":     d["is_brake"] if "is_brake" in d
                            else np.zeros(n, dtype=np.int8),
            "frame_idx":    d["frame_idx"],
            "run_id":       np.full(n, run_id, dtype=np.int32),
        })
        print(f"  {fp.name}: {n} frames")

    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def assign_frames(n: int, spec: list, seed: int) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    counts = [int(round(frac * n)) for _, _, frac in spec]
    counts[-1] += n - sum(counts)
    assignments = np.concatenate([
        np.full(cnt, i, dtype=np.int32) for i, cnt in enumerate(counts)
    ])
    rng.shuffle(assignments)
    return assignments


def to_chw_uint8(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    args = parse_args()

    spec = list(_SPEC)
    spec[1] = ("gaussian_noise",   args.noise_intensity,      0.20)
    spec[2] = ("brightness_scale", args.brightness_intensity, 0.20)
    spec[4] = ("pgd",              args.pgd_epsilon,          0.20)

    raw = load_all_runs(args.frames_dir)
    n   = raw["wide_rgb"].shape[0]
    print(f"[prep] {n} total frames.")

    from ATOMs_Analysis.perturbation_manager import PerturbationManager
    pm = PerturbationManager()

    n_cameras   = raw["wide_rgb"].shape[-1] // raw["wide_rgb"].shape[-2]
    assignments = assign_frames(n, spec, args.seed)

    out_wide    = np.empty_like(raw["wide_rgb"])
    labels      = np.zeros(n, dtype=np.int32)
    pert_names  = np.empty(n, dtype=object)
    intensities = np.zeros(n, dtype=np.float32)

    for entry_idx, (pert_name, intensity, _) in enumerate(spec):
        frame_idxs = np.where(assignments == entry_idx)[0]
        is_clean   = pert_name is None
        # PGD pixels are crafted later (needs the model); keep them clean here but
        # still record the label so the array job knows which frames to attack.
        is_pgd     = pert_name == "pgd"

        for fi in frame_idxs:
            if is_clean or is_pgd:
                out_wide[fi] = raw["wide_rgb"][fi]
            else:
                perturbed    = pm.perturb_tfv6_image(
                    raw["wide_rgb"][fi],
                    perturbation = pert_name,
                    intensity    = intensity,
                    n_cameras    = n_cameras,
                )
                out_wide[fi] = to_chw_uint8(perturbed)

            labels[fi]      = 0 if is_clean else 1
            pert_names[fi]  = "clean" if is_clean else pert_name
            intensities[fi] = 0.0    if is_clean else intensity

        tag = "clean" if is_clean else f"{pert_name}@{intensity}"
        tag += " (pixels deferred to array job)" if is_pgd else ""
        print(f"  '{tag}': {len(frame_idxs)} frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wide_rgb     = out_wide,
        seg_red_wide = raw["seg_red_wide"],
        cmd          = raw["cmd"],
        speed        = raw["speed"],
        is_brake     = raw["is_brake"],
        frame_idx    = raw["frame_idx"],
        run_id       = raw["run_id"],
        label        = labels,
        perturbation = pert_names,
        intensity    = intensities,
    )

    # Write frame count so submit_test.sh can size the array job if needed
    (args.output.parent / "test_meta.txt").write_text(str(n))

    n_pert = int(labels.sum())
    print(f"[prep] Saved {n} frames ({n - n_pert} clean, {n_pert} perturbed) → {args.output}")


if __name__ == "__main__":
    main()
