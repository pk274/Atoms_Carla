"""
add_shrinkage.py — one-shot migration script

Upgrades an existing baseline_N.npz file from a raw empirical covariance
(old ridge-at-score-time approach) to a shrinkage-regularised covariance
(new shrinkage-at-fit-time approach).

The baseline file schema is:
    series      [N, C] float32  — per-frame attention profiles
    mean        [C]    float32  — mean profile
    cov         [C, C] float32  — covariance (this is the field we modify)
    class_ids   [C]    int32
    class_names [C]    object
    cmd_filter  [1]
    n_frames    [1]
    reference_narr (optional, WoR only)

Only `cov` is changed; all other keys are copied verbatim.

Usage
-----
    # Use defaults from atoms_config.py  (BASELINE_DATA_DIR / baseline_MODE_ANALYSIS.npz)
    python ATOMs_Analysis/add_shrinkage.py

    # Explicit file and alpha
    python ATOMs_Analysis/add_shrinkage.py --baseline-file data/TFV6/baseline_data_alt/baseline_2.npz
    python ATOMs_Analysis/add_shrinkage.py --alpha 0.05
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ATOMs_Analysis.utils.distance_computer import DistanceComputer
from ATOMs_Analysis.atoms_config import ExperimentConfig as conf


def migrate_baseline(path: Path, alpha: float) -> None:
    """Apply shrinkage to the `cov` key of a baseline .npz file."""
    path = path.resolve()
    print(f"\n[baseline] {path}")

    if not path.exists():
        print("  File not found — skipping")
        return

    # Backup original
    backup = path.with_name(path.stem + "_empirical.npz")
    if backup.exists():
        print(f"  Backup already exists: {backup.name} — skipping rename, loading from backup")
    else:
        shutil.copy2(str(path), str(backup))
        print(f"  Backed up original → {backup.name}")

    # Load from the backup (= original data)
    data = np.load(backup, allow_pickle=True)
    keys = list(data.keys())
    print(f"  Keys: {keys}")

    cov_raw = data["cov"].astype(np.float64)
    cov_shrunk = DistanceComputer.apply_shrinkage(cov_raw, alpha)
    print(f"  Diagonal before shrinkage (first 5): {np.diag(cov_raw)[:5].round(6)}")
    print(f"  Diagonal after  shrinkage (first 5): {np.diag(cov_shrunk)[:5].round(6)}")

    # Rebuild save dict: copy all keys, replace cov.
    # Some object-dtype keys (e.g. class_names) may fail to unpickle when the
    # file was written with a different numpy version — skip those gracefully.
    save_kwargs = {}
    for k in keys:
        if k == "cov":
            save_kwargs["cov"] = cov_shrunk.astype(np.float32)
        else:
            try:
                save_kwargs[k] = data[k]
            except Exception as e:
                print(f"  Warning: could not copy key '{k}' ({e}) — skipping")

    np.savez_compressed(path, **save_kwargs)
    print(f"  Saved shrinkage-regularised file → {path.name}  (alpha={alpha})")


def main() -> None:
    default_path = Path(conf.BASELINE_DATA_DIR) / f"baseline_{conf.MODE_ANALYSIS}.npz"

    parser = argparse.ArgumentParser(
        description="Apply shrinkage regularisation to the covariance in a baseline .npz file"
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=default_path,
        help=f"Path to the baseline .npz file (default: {default_path})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=conf.SHRINKAGE_ALPHA,
        help=f"Shrinkage intensity (default: {conf.SHRINKAGE_ALPHA})",
    )
    args = parser.parse_args()

    print("Migration settings")
    print(f"  baseline file : {args.baseline_file}")
    print(f"  alpha         : {args.alpha}")

    migrate_baseline(args.baseline_file, args.alpha)

    print("\nDone.  Original preserved as *_empirical.npz.")


if __name__ == "__main__":
    main()
