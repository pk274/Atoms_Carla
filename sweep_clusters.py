#!/usr/bin/env python
"""
sweep_clusters.py
-----------------
Run run_analysis.py for a range of GMM cluster counts (K) and save each
run's output under:

    data/<AGENT>/results/<K> clusters/atoms_analysis_mode_<N>/

This is the folder layout that summarize_results.py treats as authoritative
snapshots — run that script afterwards to compile the cross-K report.

Usage:
    python sweep_clusters.py
    python sweep_clusters.py --k-values 2 4 6 8
    python sweep_clusters.py --dry-run

Prerequisite: the expensive parts (baseline profiles, test profiles) must
already have been computed once.  Make sure atoms_config.py has:
    RECOMPUTE_BASELINE      = False
    RECOMPUTE_TEST_ATOMS    = False
    RECOMPUTE_MDX_BASELINE  = False
Only the GMM fit and everything downstream re-runs per K, which is fast.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

K_VALUES_DEFAULT = [3, 7, 11, 15, 19]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--k-values", nargs="+", type=int, default=K_VALUES_DEFAULT,
        metavar="K",
        help="GMM cluster counts to sweep (default: 2 4 6 8 … 20)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing anything",
    )
    ap.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the RECOMPUTE warning prompt",
    )
    args = ap.parse_args()

    # Import config for paths (we never modify it here — run_analysis.py gets K via --gmm-k)
    sys.path.insert(0, str(Path(__file__).parent))
    # Add transfuserv6 to path the same way run_analysis.py does, so the import succeeds
    sys.path.insert(0, str(Path(__file__).parent / "pcla_agents" / "transfuserv6"))
    from ATOMs_Analysis.atoms_config import ExperimentConfig as conf

    results_dir = Path(conf.RESULTS_DIR)
    mode        = conf.MODE_ANALYSIS
    src_dir     = results_dir / f"atoms_analysis_mode_{mode}"

    print(f"Agent:    {conf.AGENT}")
    print(f"Mode:     {mode}")
    print(f"Results:  {results_dir}")
    print(f"K sweep:  {args.k_values}")
    if args.dry_run:
        print("(dry-run — nothing will be executed)")
    print()

    # Warn if the user left expensive recompute flags on
    expensive = {
        "RECOMPUTE_BASELINE":     conf.RECOMPUTE_BASELINE,
        "RECOMPUTE_TEST_ATOMS":   conf.RECOMPUTE_TEST_ATOMS,
        "RECOMPUTE_MDX_BASELINE": conf.RECOMPUTE_MDX_BASELINE,
    }
    active = [k for k, v in expensive.items() if v]
    if active and not args.dry_run:
        print("WARNING: the following flags are True in atoms_config.py:")
        for flag in active:
            print(f"  {flag} = True")
        print("These cause expensive re-computation every run, multiplied by",
              len(args.k_values), "K-values.")
        print("Set them to False to skip re-computation and only re-fit the GMM.")
        if not args.yes:
            ans = input("Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit(0)
        print()

    failed: list[int] = []

    for K in args.k_values:
        dst_dir = results_dir / f"{K} clusters" / f"atoms_analysis_mode_{mode}"
        tag     = f"[K={K}]"

        if args.dry_run:
            print(f"{tag} would run: python run_analysis.py --gmm-k {K}")
            print(f"       copy: {src_dir.name}  →  {dst_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"{tag} Running run_analysis.py --gmm-k {K}")
        print(f"{'='*60}")

        ret = subprocess.run(
            [sys.executable, "run_analysis.py", "--gmm-k", str(K)],
            check=False,
        )

        if ret.returncode != 0:
            print(f"{tag} ERROR: run_analysis.py exited with code {ret.returncode} — skipping copy.")
            failed.append(K)
            continue

        if not src_dir.exists():
            print(f"{tag} WARNING: expected output dir {src_dir} not found — skipping copy.")
            failed.append(K)
            continue

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir)
        print(f"{tag} Saved snapshot → {dst_dir}")

    print(f"\n{'='*60}")
    if args.dry_run:
        print("Dry-run complete — no files changed.")
    else:
        done = [K for K in args.k_values if K not in failed]
        print(f"Sweep complete.  Success: {done}  Failed: {failed}")
        if done:
            print("\nNext step:  python summarize_results.py")


if __name__ == "__main__":
    main()
