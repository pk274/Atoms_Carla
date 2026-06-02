#!/usr/bin/env python3
"""
gather_test.py
--------------
Combine partial profile files produced by compute_test_chunk.py into:
  - test_profiles.npy      [N, num_classes]  ATOMs attention profiles
  - test_speed_logits.npy  [N, 8]            raw speed logits for PEOC scoring

Partials are sorted by chunk_start before concatenation so the final arrays
match the original frame ordering in test_labeled.npz.

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
    p.add_argument("--partials-dir",       required=True, type=Path,
                   help="Directory containing partial_test_*.npz files.")
    p.add_argument("--output",             required=True, type=Path,
                   help="Output path for test_profiles.npy.")
    p.add_argument("--speed-logits-output", default=None, type=Path,
                   help="Output path for test_speed_logits.npy (TFV6) or test_logits.npy (WOR). "
                        "Default: auto-named next to --output.")
    p.add_argument("--agent", default="TFV6", choices=["TFV6", "WOR"],
                   help="Agent architecture — controls output filename and logit key.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # WOR uses action_logits key → test_logits.npy (matches run_analysis.py loader)
    # TFV6 uses speed_logits key → test_speed_logits.npy
    logit_key   = "action_logits" if args.agent == "WOR" else "speed_logits"
    default_out = "test_logits.npy" if args.agent == "WOR" else "test_speed_logits.npy"
    speed_logits_out = args.speed_logits_output or (args.output.parent / default_out)

    partial_files = sorted(args.partials_dir.glob("partial_test_*.npz"))
    if not partial_files:
        raise FileNotFoundError(
            f"No partial_test_*.npz files found in {args.partials_dir}\n"
            "Check that all array tasks completed (squeue / sacct)."
        )
    print(f"[gather_test] Found {len(partial_files)} partial files.")

    parts      = []
    has_logits = True

    for f in partial_files:
        p           = np.load(f, allow_pickle=True)
        chunk_start = int(p["chunk_start"][0])
        profiles    = p["profiles"]
        logits      = p[logit_key] if logit_key in p else None
        if logits is None:
            has_logits = False
            print(f"  WARNING: {f.name} missing {logit_key} — "
                  f"{default_out} will not be written.")
        parts.append((chunk_start, profiles, logits))
        print(f"  {f.name}: chunk_start={chunk_start}, shape={profiles.shape}")

    # Sort by chunk_start to preserve original frame ordering
    parts.sort(key=lambda x: x[0])

    # Drop empty chunks (tasks that were past the end of the dataset)
    non_empty = [(cs, prof, lg) for cs, prof, lg in parts if prof.shape[0] > 0]
    if len(non_empty) < len(parts):
        print(f"  ({len(parts) - len(non_empty)} empty chunks skipped)")

    profiles = np.concatenate([prof for _, prof, _ in non_empty], axis=0)
    print(f"[gather_test] Total profiles: {profiles.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, profiles)
    print(f"[gather_test] test_profiles.npy saved → {args.output}")

    if has_logits:
        speed_logits = np.concatenate([lg for _, _, lg in non_empty], axis=0)
        print(f"[gather_test] Total speed_logits: {speed_logits.shape}")
        speed_logits_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(speed_logits_out, speed_logits)
        print(f"[gather_test] test_speed_logits.npy saved → {speed_logits_out}")

    agent   = args.agent
    att_dir = f"data/{agent}/test_data/attention"
    logits_fname = default_out
    print(f"\nNext step — copy into the repo and push (run on Viper):")
    print(f"  cp {args.output} /u/$USER/pcla/{att_dir}/test_profiles.npy")
    if has_logits:
        print(f"  cp {speed_logits_out} /u/$USER/pcla/{att_dir}/{logits_fname}")
    print(f"  cd /u/$USER/pcla")
    print(f"  git add -f {att_dir}/test_profiles.npy")
    if has_logits:
        print(f"  git add -f {att_dir}/{logits_fname}")
    print(f"  git commit -m 'add {agent} test_profiles.npy"
          + (f" and {logits_fname}" if has_logits else "")
          + " from HPC'")
    print(f"  git push")
    print(f"Then locally: git pull, set RECOMPUTE_TEST_ATOMS=False")


if __name__ == "__main__":
    main()
