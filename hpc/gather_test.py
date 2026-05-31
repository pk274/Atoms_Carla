#!/usr/bin/env python3
"""
gather_test.py
--------------
Combine partial profile files produced by compute_test_chunk.py into a
single test_profiles.npy compatible with run_analysis.py.

Partials are sorted by chunk_start before concatenation, so the final
array matches the original frame ordering in test_labeled.npz.

Usage:
    python hpc/gather_test.py \
        --partials-dir /ptmp/$USER/atoms_test/partials \
        --output       /ptmp/$USER/atoms_test/test_profiles.npy
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--partials-dir", required=True, type=Path,
                   help="Directory containing partial_test_*.npz files.")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output path for test_profiles.npy.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    partial_files = sorted(args.partials_dir.glob("partial_test_*.npz"))
    if not partial_files:
        raise FileNotFoundError(
            f"No partial_test_*.npz files found in {args.partials_dir}\n"
            "Check that all array tasks completed (squeue / sacct)."
        )
    print(f"[gather_test] Found {len(partial_files)} partial files.")

    parts = []
    for f in partial_files:
        p           = np.load(f, allow_pickle=True)
        chunk_start = int(p["chunk_start"][0])
        profiles    = p["profiles"]
        parts.append((chunk_start, profiles))
        print(f"  {f.name}: chunk_start={chunk_start}, shape={profiles.shape}")

    # Sort by chunk_start to preserve original frame ordering
    parts.sort(key=lambda x: x[0])

    # Drop empty chunks (tasks that were past the end of the dataset)
    non_empty = [(cs, prof) for cs, prof in parts if prof.shape[0] > 0]
    if len(non_empty) < len(parts):
        print(f"  ({len(parts) - len(non_empty)} empty chunks skipped)")

    profiles = np.concatenate([p for _, p in non_empty], axis=0)
    print(f"[gather_test] Total profiles: {profiles.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, profiles)
    print(f"[gather_test] test_profiles.npy saved → {args.output}")
    print(f"\nNext step — copy to your local repo and commit:")
    print(f"  cp {args.output} <CODE_DIR>/data/TFV6/test_data/attention/test_profiles.npy")
    print(f"  cd <CODE_DIR>")
    print(f"  git add -f data/TFV6/test_data/attention/test_profiles.npy")
    print(f"  git commit -m 'add TFV6 test_profiles.npy from HPC'")
    print(f"  git push")


if __name__ == "__main__":
    main()
