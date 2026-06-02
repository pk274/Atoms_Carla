#!/usr/bin/env python3
"""
prep_live_pert_wor.py
---------------------
Concatenate WoR live-perturbation frame files into a single NPZ so the
SLURM array job can address frames by index.

WoR-specific: preserves both wide_rgb and narr_rgb (both cameras are
required for WoR LRP). TFV6's prep_live_pert.py only saves wide_rgb.

Input files:  <frames-dir>/run_<perturbation>_live_pert_*.npz
Output file:  <output>  (live_pert_concat.npz)
Side-effect:  writes <output-dir>/live_pert_meta.txt with the total frame
              count (read by submit_live_pert_wor.sh to size the array job).

Usage (standalone):
    python hpc/prep_live_pert_wor.py \
        --frames-dir   /ptmp/$USER/atoms_wor_live_pert/frames \
        --perturbation pgd \
        --output       /ptmp/$USER/atoms_wor_live_pert/live_pert_concat.npz
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir",   required=True, type=Path,
                   help="Directory containing run_{perturbation}_live_pert_*.npz files.")
    p.add_argument("--perturbation", required=True, type=str,
                   help="Perturbation name, e.g. 'pgd'. Used to match filenames.")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output path for live_pert_concat.npz.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pattern = f"run_{args.perturbation}_live_pert_*.npz"
    files   = sorted(args.frames_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {args.frames_dir}\n"
            f"Check that CARLA live-perturbation recording has been run for "
            f"perturbation='{args.perturbation}'."
        )
    print(f"[prep_live_pert_wor] Found {len(files)} run file(s).")

    parts = []
    for run_id, fp in enumerate(files):
        d = np.load(fp, allow_pickle=False)
        n = d["wide_rgb"].shape[0]

        if "narr_rgb" not in d:
            raise KeyError(
                f"{fp.name} is missing 'narr_rgb'. "
                "WoR live-pert recordings must save both cameras."
            )

        entry = {
            "wide_rgb":     d["wide_rgb"],
            "narr_rgb":     d["narr_rgb"],
            "seg_red_wide": d["seg_red_wide"],
            "seg_red_narr": d["seg_red_narr"] if "seg_red_narr" in d
                            else np.zeros((n, *d["seg_red_wide"].shape[1:]), dtype=np.uint8),
            "cmd":          d["cmd"],
            "speed":        d["speed"],
            "is_brake":     d["is_brake"] if "is_brake" in d
                            else np.zeros(n, dtype=np.int8),
            "frame_idx":    d["frame_idx"],
            "run_id":       np.full(n, run_id, dtype=np.int32),
        }
        parts.append(entry)
        print(f"  {fp.name}: {n} frames")

    combined = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    n_total  = int(combined["wide_rgb"].shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **combined)

    (args.output.parent / "live_pert_meta.txt").write_text(str(n_total))

    print(f"[prep_live_pert_wor] {n_total} frames saved → {args.output}")


if __name__ == "__main__":
    main()
