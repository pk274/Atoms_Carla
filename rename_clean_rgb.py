#!/usr/bin/env python3
"""
rename_clean_rgb.py
-------------------
Rename _clean_rgb.npz frame files so their variant prefix matches the
corresponding perturbed recording in the same directory.

Clean and perturbed files are matched by stripping the trailing timestamp
(_HHMMSS_NNN) and run-index (_NNN) from each variant name, then pairing
on the remaining base name.

Example:
  run_pgd_live_pert_brake_205453_000_clean_rgb.npz  →  strip suffix → "brake"
  run_pgd_live_pert_brake_205328_000.npz             →  strip suffix → "brake"
  → rename clean file to run_pgd_live_pert_brake_205328_000_clean_rgb.npz

Dry-run by default. Add --execute to apply renames.

Usage:
  python rename_clean_rgb.py                        # dry run, default dirs
  python rename_clean_rgb.py --execute              # apply renames
  python rename_clean_rgb.py --frames-dirs path/a path/b --perturbations pgd phantom_obstacle
"""

import argparse
import re
from pathlib import Path

DATA_ROOT = Path("C:/Users/paulk/Desktop/Unistuff/Masterarbeit/Code/PCLA/data/TFV6")

DEFAULT_FRAMES_DIRS = [
    DATA_ROOT / "test_data"     / "live_pert_frames",
    DATA_ROOT / "test_data_alt" / "live_pert_frames",
]

DEFAULT_PERTURBATIONS = ["pgd", "gaussian_noise", "brightness_scale", "phantom_obstacle"]


def strip_suffix(variant: str) -> str:
    """Strip trailing _HHMMSS_NNN or bare _NNN to get the scene base name."""
    m = re.sub(r'_\d{6}_\d{3}$', '', variant)
    if m != variant:
        return m
    return re.sub(r'_\d{3}$', '', variant)


def process_dir(frames_dir: Path, perturbation: str, execute: bool) -> int:
    prefix = f"run_{perturbation}_live_pert_"
    all_files = sorted(frames_dir.glob(f"{prefix}*.npz"))

    clean_files = [f for f in all_files if f.stem.endswith("_clean_rgb")]
    perturbed_files = [f for f in all_files if not f.stem.endswith("_clean_rgb")]

    if not clean_files:
        return 0

    # Build base → perturbed_file map
    perturbed_by_base: dict[str, Path] = {}
    for fp in perturbed_files:
        variant = fp.stem[len(prefix):]
        base = strip_suffix(variant)
        if base in perturbed_by_base:
            print(f"  WARNING: duplicate base '{base}': "
                  f"{perturbed_by_base[base].name} and {fp.name} — skipping both")
            perturbed_by_base[base] = None  # mark ambiguous
        else:
            perturbed_by_base[base] = fp

    n = 0
    for clean_fp in clean_files:
        variant = clean_fp.stem[len(prefix):]           # e.g. "brake_205453_000_clean_rgb"
        core    = variant.removesuffix("_clean_rgb")    # e.g. "brake_205453_000"
        base    = strip_suffix(core)                    # e.g. "brake"

        match = perturbed_by_base.get(base)
        if match is None:
            print(f"  [SKIP] {clean_fp.name}: no unique perturbed match for base '{base}'")
            continue

        perturbed_variant = match.stem[len(prefix):]    # e.g. "brake_205328_000"
        new_name = f"{prefix}{perturbed_variant}_clean_rgb.npz"
        new_fp   = clean_fp.parent / new_name

        if new_fp == clean_fp:
            print(f"  [OK]     {clean_fp.name}")
            continue

        print(f"  [RENAME] {clean_fp.name}")
        print(f"        -> {new_name}")

        if execute:
            if new_fp.exists():
                print(f"  [ERROR]  target already exists, skipping")
            else:
                clean_fp.rename(new_fp)
                n += 1
        else:
            n += 1

    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dirs", nargs="+", type=Path, default=DEFAULT_FRAMES_DIRS)
    p.add_argument("--perturbations", nargs="+", default=DEFAULT_PERTURBATIONS)
    p.add_argument("--execute", action="store_true",
                   help="Apply renames. Without this flag the script is a dry run.")
    args = p.parse_args()

    if not args.execute:
        print("DRY RUN — pass --execute to apply renames.\n")

    total = 0
    for frames_dir in args.frames_dirs:
        if not frames_dir.exists():
            print(f"[SKIP] {frames_dir} does not exist")
            continue
        print(f"\n{frames_dir}")
        for pert in args.perturbations:
            n = process_dir(frames_dir, pert, args.execute)
            total += n

    verb = "Renamed" if args.execute else "Would rename"
    print(f"\n{verb} {total} file(s) total.")
    if not args.execute and total > 0:
        print("Run with --execute to apply.")


if __name__ == "__main__":
    main()
