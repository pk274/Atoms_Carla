#!/usr/bin/env python3
"""
cleanup_live_pert_attention.py
------------------------------
Identify (and optionally delete) stale files in the live_pert attention
folders.  A file is "needed" only if its variant has a matching perturbed
frame file in live_pert_frames/.

Dry-run by default.  Pass --execute to delete.
"""

import argparse
import re
from pathlib import Path

DATA_ROOT = Path("C:/Users/paulk/Desktop/Unistuff/Masterarbeit/Code/PCLA/data/TFV6")
PERTURBATIONS = ["pgd", "gaussian_noise", "brightness_scale", "phantom_obstacle"]


def valid_variants(frames_dir: Path, pert: str) -> set[str]:
    """Variant names that have a real (non-clean) frame file for this perturbation."""
    prefix = f"run_{pert}_live_pert_"
    return {
        fp.stem[len(prefix):]
        for fp in frames_dir.glob(f"{prefix}*.npz")
        if not fp.stem.endswith("_clean_rgb")
    }


def extract_variant(filename: str) -> str | None:
    """
    Pull the variant out of a profile/logit filename.
    E.g. "live_pert_profiles_brake_205328_000_2.npy"       -> "brake_205328_000"
         "live_pert_profiles_brake_205328_000_clean_2.npy" -> "brake_205328_000"
    Returns None if the name doesn't match the expected pattern.
    """
    for pfx in ("live_pert_profiles_", "live_pert_speed_logits_"):
        if filename.startswith(pfx):
            rest = filename[len(pfx):]          # "brake_205328_000_clean_2.npy"
            rest = rest.removesuffix(".npy")     # "brake_205328_000_clean_2"
            rest = re.sub(r'_\d+$', '', rest)    # strip mode number first -> "brake_205328_000_clean"
            rest = re.sub(r'_clean$', '', rest)  # then strip _clean -> "brake_205328_000"
            return rest
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="Actually delete stale files. Without this, prints a dry run.")
    p.add_argument("--experiment-variant", default="alternative",
                   choices=["alternative", "original"])
    args = p.parse_args()

    suffix = "_alt" if args.experiment_variant == "alternative" else ""
    test_data_dir  = DATA_ROOT / f"test_data{suffix}"
    frames_dir     = test_data_dir / "live_pert_frames"
    attention_root = test_data_dir / "attention" / "live_pert"

    if not args.execute:
        print("DRY RUN — pass --execute to delete.\n")

    total_keep = total_delete = total_missing = 0

    for pert in PERTURBATIONS:
        att_dir = attention_root / pert
        if not att_dir.exists():
            continue

        good_variants = valid_variants(frames_dir, pert)
        print(f"\n{'='*60}")
        print(f"  {pert}  —  valid variants: {sorted(good_variants) or '(none found!)'}")
        print(f"{'='*60}")

        # Check that every valid variant has its profile
        for v in sorted(good_variants):
            for mode in (2,):   # extend to [1, 2] if both modes are used
                needed = att_dir / f"live_pert_profiles_{v}_{mode}.npy"
                if not needed.exists():
                    print(f"  MISSING  {needed.name}")
                    total_missing += 1

        # Audit existing files
        for fp in sorted(att_dir.iterdir()):
            if not fp.suffix == ".npy":
                continue

            variant = extract_variant(fp.name)
            if variant is None:
                print(f"  [DELETE] {fp.name}  <- unrecognised name pattern")
                if args.execute:
                    fp.unlink()
                total_delete += 1
                continue

            # Old flat files (no variant in name, e.g. live_pert_profiles_2.npy)
            # These have no variant after stripping — variant would be empty or just a digit.
            if not variant or re.fullmatch(r'\d+', variant):
                print(f"  [DELETE] {fp.name}  <- old flat format")
                if args.execute:
                    fp.unlink()
                total_delete += 1
                continue

            if variant in good_variants:
                print(f"  [KEEP]   {fp.name}")
                total_keep += 1
            else:
                print(f"  [DELETE] {fp.name}  <- variant '{variant}' has no frame file for {pert}")
                if args.execute:
                    fp.unlink()
                total_delete += 1

    print(f"\n{'='*60}")
    print(f"Keep: {total_keep}  |  {'Deleted' if args.execute else 'Would delete'}: {total_delete}  |  Missing: {total_missing}")
    if not args.execute and total_delete > 0:
        print("Run with --execute to apply deletions.")
    if total_missing > 0:
        print("MISSING files need to be (re-)computed on the HPC.")


if __name__ == "__main__":
    main()
