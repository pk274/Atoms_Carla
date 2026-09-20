#!/usr/bin/env python
"""
fix_baseline_leak.py — remove the four stale routes from the TFV6 alt-split baseline.

Four route .npz files left behind by an aborted earlier migration run were never
cleared from baseline_data_alt/frames/, and BaselineDataLoader globs whatever is
in that directory.  They carry 124 frames that the split assigns to the test and
validation sets, so the training baseline overlaps both evaluation splits.
Removing them restores the intended 5000-frame baseline exactly and leaves the
evaluation splits untouched at 1000 frames each (200 per perturbation), which is
what keeps the pooled-AUROC identity intact.

Full diagnosis and rationale: ATOMs_SOLID/thesis/carla_leak_fix_plan.md.

The stale files are identified by *measurement*, not by a hardcoded list: they are
the ones whose frame count differs from the modal count, since frames_per_route is
a single value per migration call and a route can only ever yield fewer.  The
hardcoded names below are asserted against that, so the script fails loudly if the
data on disk is not what this fix was written for.

Run order matters: the cached arrays are concatenated in *name-sorted* file order
(see the platform-independent sort in BaselineDataLoader.load_all_runs).

    python fix_baseline_leak.py            # dry run, prints what it would do
    python fix_baseline_leak.py --apply    # writes backups, then applies
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "TFV6" / "baseline_data_alt"
FRAMES = BASE / "frames"
QUARANTINE = BASE / "quarantine_stale_routes"

EXPECTED_STALE = {
    "run_Town03_Town03_Rep0_route_000395_route0_04_25_22_55_56.npz",
    "run_Town04_Town04_Rep0_Town04_Scenario9_72_route0_04_24_05_13_10.npz",
    "run_Town06_Town06_Rep0_route_000656_route0_04_25_11_35_31.npz",
    "run_Town06_Town06_Rep0_route_000710_route0_04_24_16_44_22.npz",
}
EXPECTED_TOTAL_BEFORE = 5124
EXPECTED_TOTAL_AFTER = 5000

# (file, keys to mask along axis 0)
ARRAYS = [
    (BASE / "baseline_2.npz", ["series"]),
    (BASE / "mdx_features.npz", ["features", "actions"]),
    (BASE / "mdx_fc_features.npz", ["features", "actions"]),
]

# Rare classes make good fingerprints: few frames carry any relevance on them, so
# a row->frame mapping that is off by even one file fails immediately.
FINGERPRINT_CLASSES = [7, 6, 4]  # SpecialVehicle, Obstacle, Pedestrian


def frame_counts(frames_dir: Path) -> dict[str, int]:
    out = {}
    for f in frames_dir.glob("run_*.npz"):
        with np.load(f, allow_pickle=True) as z:
            out[f.name] = len(z["frame_idx"])
    return out


def ordered_names(counts: dict[str, int]) -> list[str]:
    """Name-sorted order — the order the cached arrays are concatenated in."""
    return sorted(counts)


def detect_stale(counts: dict[str, int]) -> list[str]:
    """Files whose frame count exceeds the modal count (see module docstring)."""
    modal = max(set(counts.values()), key=list(counts.values()).count)
    return sorted(n for n, c in counts.items() if c > modal)


def drop_mask(counts: dict[str, int], stale: set[str]) -> np.ndarray:
    total = sum(counts.values())
    mask = np.zeros(total, dtype=bool)
    off = 0
    for name in ordered_names(counts):
        n = counts[name]
        if name in stale:
            mask[off : off + n] = True
        off += n
    return mask


def fingerprint_violations(series: np.ndarray, counts: dict[str, int],
                           frames_dir: Path) -> tuple[int, int]:
    """(consistent, tested) — does each rare-class row map to a frame containing it?"""
    order = ordered_names(counts)
    bounds, off = [], 0
    for name in order:
        bounds.append((off, off + counts[name], name))
        off += counts[name]

    targets = []
    for c in FINGERPRINT_CLASSES:
        targets += [(int(r), c) for r in np.flatnonzero(series[:, c] > 0)]

    ok = 0
    seg_cache: tuple[str, np.ndarray] | None = None
    for row, c in sorted(targets):
        for a, b, name in bounds:
            if a <= row < b:
                local = row - a
                break
        else:
            continue
        if seg_cache is None or seg_cache[0] != name:
            with np.load(frames_dir / name, allow_pickle=True) as z:
                seg_cache = (name, z["seg_red_wide"])
        if bool((seg_cache[1][local] == c).any()):
            ok += 1
    return ok, len(targets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually move files and rewrite arrays (default: dry run)")
    args = ap.parse_args()

    counts = frame_counts(FRAMES)
    total = sum(counts.values())
    print(f"frames dir: {len(counts)} files, {total} frames")

    stale = detect_stale(counts)
    print(f"stale (frame count above the modal count): {len(stale)}")
    for n in stale:
        print(f"   {counts[n]:>3} frames  {n}")

    if not stale and total == EXPECTED_TOTAL_AFTER and len(counts) == 186:
        print("\nAlready applied — baseline is 5000 frames from 186 routes, nothing to do.")
        series = np.load(BASE / "baseline_2.npz", allow_pickle=True)["series"]
        ok, n = fingerprint_violations(series, counts, FRAMES)
        print(f"alignment check: {ok}/{n} rare-class rows consistent")
        return 0 if ok == n else 1

    if set(stale) != EXPECTED_STALE:
        print("\nABORT: the measured stale set does not match the expected one.")
        print(f"  measured: {sorted(set(stale))}")
        print(f"  expected: {sorted(EXPECTED_STALE)}")
        return 1
    if total != EXPECTED_TOTAL_BEFORE:
        print(f"\nABORT: expected {EXPECTED_TOTAL_BEFORE} frames before the fix, found {total}.")
        return 1

    mask = drop_mask(counts, set(stale))
    keep = ~mask
    print(f"\nrows to drop: {int(mask.sum())}  ->  {int(keep.sum())} remain")
    off, ranges = 0, []
    for name in ordered_names(counts):
        n = counts[name]
        if name in set(stale):
            ranges.append(f"[{off}:{off + n})")
        off += n
    print(f"row ranges (name-sorted order): {', '.join(ranges)}")

    if int(keep.sum()) != EXPECTED_TOTAL_AFTER:
        print(f"\nABORT: would leave {int(keep.sum())} rows, expected {EXPECTED_TOTAL_AFTER}.")
        return 1

    series = np.load(BASE / "baseline_2.npz", allow_pickle=True)["series"]
    if len(series) != total:
        print(f"\nABORT: baseline_2.npz has {len(series)} rows but frames dir yields {total}.")
        return 1

    ok, n = fingerprint_violations(series, counts, FRAMES)
    print(f"\nalignment check BEFORE: {ok}/{n} rare-class rows map to a frame containing that class")
    if ok != n:
        print("ABORT: the cached series is not aligned with this frames dir; do not mask rows.")
        return 1

    for path, keys in ARRAYS:
        if not path.exists():
            print(f"\nABORT: missing {path.name}")
            return 1
        with np.load(path, allow_pickle=True) as z:
            for k in keys:
                if len(z[k]) != total:
                    print(f"\nABORT: {path.name}[{k}] has {len(z[k])} rows, expected {total}.")
                    return 1

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    # ---- apply -----------------------------------------------------------
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    for path, keys in ARRAYS:
        bak = path.with_suffix(path.suffix + ".preleak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"backup -> {bak.name}")

    for path, keys in ARRAYS:
        with np.load(path, allow_pickle=True) as z:
            out = {k: z[k] for k in z.files}
        for k in keys:
            out[k] = out[k][keep]
        if path.name == "baseline_2.npz":
            s = out["series"].astype(np.float64)
            out["mean"] = s.mean(axis=0).astype(np.float32)
            out["cov"] = np.cov(s.T).astype(np.float32)
            out["n_frames"] = np.array([len(s)])
        np.savez_compressed(path, **out)
        print(f"rewrote {path.name}: {', '.join(f'{k}={len(out[k])}' for k in keys)}")

    for name in stale:
        shutil.move(str(FRAMES / name), str(QUARANTINE / name))
    print(f"moved {len(stale)} stale frame files -> {QUARANTINE.name}/")

    # ---- verify ----------------------------------------------------------
    counts2 = frame_counts(FRAMES)
    total2 = sum(counts2.values())
    series2 = np.load(BASE / "baseline_2.npz", allow_pickle=True)["series"]
    print(f"\nAFTER: {len(counts2)} files, {total2} frames, series {series2.shape}")
    if total2 != EXPECTED_TOTAL_AFTER or len(series2) != EXPECTED_TOTAL_AFTER:
        print("FAIL: post-fix counts are wrong.")
        return 1
    ok2, n2 = fingerprint_violations(series2, counts2, FRAMES)
    print(f"alignment check AFTER: {ok2}/{n2} rare-class rows consistent")
    if ok2 != n2:
        print("FAIL: series and frames dir are no longer aligned.")
        return 1
    print("\nOK — baseline restored to 5000 frames and still aligned with frames/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
