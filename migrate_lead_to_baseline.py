"""
migrate_lead_to_baseline.py
----------------------------
Convert LEAD CARLA dataset routes into the npz format used by
BaselineDataLoader / BaselineComputer, without needing a live CARLA session.

Usage
-----
    python migrate_lead_to_baseline.py \
        --lead_dir data/carla_leaderboard2/noScenarios \
        --n_frames 3000 \
        --exclude_towns Town05

The script discovers all route subdirectories that contain rgb/, semantics/,
and metas/ folders, groups them by CARLA town, and samples ~n_frames / n_towns
frames from each town (Town05 excluded by default for the test set).

Output:  conf.BASELINE_DATA_DIR / "frames" / run_<town>_<route>.npz

Each npz contains (all shape [N, ...]):
    wide_rgb     : [N, 3, H, W]  uint8
    seg_red_wide : [N, H, W]     uint8   (CARLA semantic class IDs)
    cmd          : [N]           int32
    speed        : [N]           float32
    is_brake     : [N]           int8
    frame_idx    : [N]           int32
    target_point          : [N, 2]  float32   ego-frame current TP (meters)
    target_point_previous : [N, 2]  float32   ego-frame previous TP
    target_point_next     : [N, 2]  float32   ego-frame next TP

Target points replicate the TRAINING dataloader recipe exactly
(carla_dataset.py: pos_global + theta, next/previous_target_points_3.25
keys with duplicate-merge, no augmentation — the clean rgb/ stream has
perturbation = 0).  When TP extraction succeeds, cmd is likewise taken from
next_commands_3.25 (the filtered list's first entry) to match what the model
saw during training; the unsuffixed next_commands list reflects a different
route-planner pop state and can genuinely differ.  All-zero TPs mean the
meta lacked the suffixed keys (legacy fallback).

narr_rgb / seg_red_narr are intentionally omitted — TFV6 is wide-only
(WIDE_ONLY_PROFILE = True).  BaselineDataLoader handles missing narr keys
by returning None; BaselineComputer passes None to atoms.process_frame().

LEAD meta format (confirmed from real sample):
- Files are XZ-compressed pickle (magic bytes: fd 37 7a 58 5a 00)
- Command:  meta['next_commands'][0]  — CARLA RoadOption int (1-6, 1-based)
- Speed:    meta['speed']             — float64
- Brake:    meta['brake']             — bool
- Town:     meta['town']              — str, e.g. 'Town03'
- RGB shape: (384, W, 3) where W = conf.N_CAMERAS * 384
    lead360 (6-cam):  W = 2304  — 6 cameras x 384px wide  [current]
    legacy  (3-cam):  W = 1152  — 3 cameras x 384px wide  [archival]
"""

from __future__ import annotations

import argparse
import lzma
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_pcla_root = Path(__file__).resolve().parent
if str(_pcla_root) not in sys.path:
    sys.path.insert(0, str(_pcla_root))

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOG = logging.getLogger(__name__)


# CARLA RoadOption integer values (1-based) → our 0-based index:
#   LEFT=1 → 0,  RIGHT=2 → 1,  STRAIGHT=3 → 2,
#   LANEFOLLOW=4 → 3,  CHANGELANELEFT=5 → 4,  CHANGELANERIGHT=6 → 5
_ROAD_OPTION_TO_IDX: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def _load_meta(meta_path: Path) -> dict:
    """Load a LEAD meta pickle (XZ-compressed pickle, magic bytes fd 37 7a 58 5a 00)."""
    with open(meta_path, "rb") as fh:
        return pickle.loads(lzma.decompress(fh.read()))

def _convert_command(raw) -> int:
    if hasattr(raw, "value"):        # RoadOption enum
        raw = raw.value
    v = int(raw)
    if v in _ROAD_OPTION_TO_IDX:
        return _ROAD_OPTION_TO_IDX[v]
    if 0 <= v <= 5:                  # already 0-based
        return v
    LOG.warning("Unknown command value %d — defaulting to LANEFOLLOW (3)", v)
    return 3


# ---------------------------------------------------------------------------
# Target-point extraction (mirrors the LEAD training dataloader)
# ---------------------------------------------------------------------------

# Must equal TrainingConfig.tp_pop_distance — selects which precomputed
# route-planner pop state the meta keys are read from.
TP_POP_DISTANCE = 3.25

_ZERO_TP = np.zeros(2, dtype=np.float32)


def _inverse_conversion_2d(point: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    """World → ego-frame 2D transform (copy of lead common_utils.inverse_conversion_2d)."""
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return rot.T @ (point - translation)


def _extract_target_points(meta: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Reproduce the training dataloader's target-point construction
    (carla_dataset.py, use_noisy_tp=False path, augmentation zeroed):
    duplicate-merge the next-TP list at TP_POP_DISTANCE, pick
    previous/current/next, transform into the ego frame via pos_global + theta.

    Returns (tp, tp_previous, tp_next, raw_cmd) — raw_cmd is the 1-based
    RoadOption of the filtered list's first entry (what training used as
    data["command"]) — or None when the meta lacks the required keys.
    """
    try:
        ego_yaw = float(meta["theta"])
        ego_pos = np.array(meta["pos_global"][:2], dtype=np.float64)
        next_tp_list  = meta[f"next_target_points_{TP_POP_DISTANCE}"]
        next_cmd_list = meta[f"next_commands_{TP_POP_DISTANCE}"]
        prev_tp_list  = meta[f"previous_target_points_{TP_POP_DISTANCE}"]
    except (KeyError, TypeError, ValueError):
        return None

    # Merge consecutive duplicate target points (training dataloader behavior)
    filtered_tp: list = []
    filtered_cmd: list = []
    for pt, c in zip(next_tp_list, next_cmd_list):
        if len(next_tp_list) == 2 or not filtered_tp or not np.allclose(pt[:2], filtered_tp[-1][:2]):
            filtered_tp.append(pt)
            filtered_cmd.append(c)
    if len(filtered_tp) < 2:
        return None

    def to_ego(point) -> np.ndarray:
        p = _inverse_conversion_2d(np.array(point[:2], dtype=np.float64), ego_pos, ego_yaw)
        return p.astype(np.float32)

    if len(filtered_tp) > 2:
        tp_prev = to_ego(filtered_tp[0])
        tp      = to_ego(filtered_tp[1])
        tp_next = to_ego(filtered_tp[2])
    else:
        tp      = to_ego(filtered_tp[1])
        tp_next = tp.copy()
        tp_prev = to_ego(prev_tp_list[-1]) if len(prev_tp_list) > 0 else to_ego(filtered_tp[0])

    return tp, tp_prev, tp_next, int(filtered_cmd[0])


# ---------------------------------------------------------------------------
# Town detection
# ---------------------------------------------------------------------------
_KNOWN_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05",
                "Town06", "Town07", "Town10", "Town15"]

def _detect_town(meta: dict, route_dir: Path) -> Optional[str]:
    """Extract the CARLA town name from a meta dict or the route path."""
    for key in ("town", "map", "world", "map_name", "carla_map"):
        val = str(meta.get(key, ""))
        for t in _KNOWN_TOWNS:
            if t.lower() in val.lower():
                return t

    # Fallback: scan the directory path itself
    for part in route_dir.parts:
        for t in _KNOWN_TOWNS:
            if t.lower() in part.lower():
                return t
    return None


# ---------------------------------------------------------------------------
# Route / frame discovery
# ---------------------------------------------------------------------------

def discover_routes(root: Path) -> List[Path]:
    """
    Return every subdirectory that contains rgb/, semantics/, and metas/.
    Works whether routes are stored flat or nested under scenario folders.
    """
    routes: List[Path] = []
    for rgb_dir in sorted(root.rglob("rgb")):
        route = rgb_dir.parent
        if (route / "semantics").is_dir() and (route / "metas").is_dir():
            routes.append(route)
    LOG.info("Discovered %d routes under %s", len(routes), root)
    return routes


def list_frame_indices(route_dir: Path) -> List[int]:
    """Return sorted frame indices found in rgb/ (based on filename stems)."""
    indices = []
    for f in sorted((route_dir / "rgb").glob("*.jpg")):
        try:
            indices.append(int(f.stem))
        except ValueError:
            pass
    return indices


# ---------------------------------------------------------------------------
# Single-frame loading
# ---------------------------------------------------------------------------

def load_frame(
    route_dir: Path, frame_idx: int
) -> Optional[Tuple[np.ndarray, np.ndarray, int, float, bool,
                    np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load one frame.  Returns
    (wide_rgb, seg_red_wide, cmd, speed, is_brake, tp, tp_previous, tp_next)
    or None if any file is missing or unreadable.  Target points are ego-frame
    float32 [2]; all-zero when the meta lacks the pop-distance-suffixed keys.

    RGB shape: (384, conf.N_CAMERAS * 384, 3).
    """
    rgb_path  = route_dir / "rgb"       / f"{frame_idx:04d}.jpg"
    seg_path  = route_dir / "semantics" / f"{frame_idx:04d}.png"
    meta_path = route_dir / "metas"     / f"{frame_idx:04d}.pkl"

    if not (rgb_path.exists() and seg_path.exists() and meta_path.exists()):
        return None

    # --- RGB ----------------------------------------------------------------
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        LOG.warning("Could not read %s", rgb_path)
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    wide_rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.uint8)  # [3, H, W]

    expected_w = conf.N_CAMERAS * 384
    if wide_rgb.shape[-1] != expected_w:
        LOG.warning(
            "Frame %s: width %d does not match conf.N_CAMERAS=%d × 384 = %d. "
            "Set conf.N_CAMERAS = 3 for legacy 3-camera data.",
            rgb_path, wide_rgb.shape[-1], conf.N_CAMERAS, expected_w,
        )

    # --- Semantics ----------------------------------------------------------
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg is None:
        LOG.warning("Could not read %s", seg_path)
        return None
    if seg.ndim == 3:
        seg = seg[:, :, 0]      # channel 0 carries CARLA class IDs
    seg_red_wide = seg.astype(np.uint8)  # [H, W]

    # --- Meta ---------------------------------------------------------------
    meta = _load_meta(meta_path)

    speed    = float(meta.get("speed", 0.0))
    is_brake = bool(meta.get("brake", False))

    tp_result = _extract_target_points(meta)
    if tp_result is not None:
        tp, tp_prev, tp_next, raw_cmd = tp_result
    else:
        tp, tp_prev, tp_next = _ZERO_TP.copy(), _ZERO_TP.copy(), _ZERO_TP.copy()
        raw_cmd = meta.get("next_commands", [4])[0]   # legacy fallback
    cmd = _convert_command(raw_cmd)

    return wide_rgb, seg_red_wide, cmd, speed, is_brake, tp, tp_prev, tp_next


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------

def build_sampling_plan(
    routes: List[Path],
    n_frames: int,
    exclude_towns: List[str],
    include_towns: Optional[List[str]] = None,
    exclude_routes: Optional[set] = None,
) -> Dict[str, List[Tuple[Path, List[int]]]]:
    """
    Group routes by town, filter, then pick evenly-spaced frames so that each
    retained town contributes ~n_frames / n_towns frames.

    Parameters
    ----------
    exclude_towns  : towns to drop (applied first).
    include_towns  : if given, keep ONLY these towns (applied after exclude).
                     Pass ["Town05"] to build a Town05-only test-set plan.
    exclude_routes : set of route directory *names* to skip within retained towns.
                     Used by migrate_valset() to omit routes already in the test set.

    Returns: town → [(route_dir, [frame_indices]), ...]
    """
    # Peek at the first meta of each route to identify its town
    town_to_routes: Dict[str, List[Path]] = defaultdict(list)
    for route in routes:
        indices = list_frame_indices(route)
        if not indices:
            continue
        town = None
        meta_path = route / "metas" / f"{indices[0]:04d}.pkl"
        if meta_path.exists():
            try:
                town = _detect_town(_load_meta(meta_path), route)
            except Exception:
                pass
        if town is None:
            town = _detect_town({}, route)   # path-based fallback
        if town is None:
            town = "unknown"
        town_to_routes[town].append(route)

    # Report and filter
    for t, rs in sorted(town_to_routes.items()):
        LOG.info("  %-10s  %d routes", t, len(rs))

    for t in exclude_towns:
        if t in town_to_routes:
            LOG.info("Excluding town %s (%d routes)", t, len(town_to_routes.pop(t)))

    if include_towns is not None:
        drop = [t for t in list(town_to_routes) if t not in include_towns]
        for t in drop:
            LOG.info("Dropping town %s (not in include list)", t)
            town_to_routes.pop(t)

    # Route-level exclusion — remove routes whose directory name is in the exclusion set.
    # Used to prevent val set routes from overlapping with the test set.
    if exclude_routes:
        for t in list(town_to_routes):
            before = len(town_to_routes[t])
            town_to_routes[t] = [r for r in town_to_routes[t] if r.name not in exclude_routes]
            dropped = before - len(town_to_routes[t])
            if dropped:
                LOG.info("  %s: excluded %d route(s) already in test set", t, dropped)

    active_towns = sorted(town_to_routes)
    if not active_towns:
        raise ValueError("No routes remaining after town filtering.")

    frames_per_town = max(1, n_frames // len(active_towns))
    LOG.info("Target: %d frames × %d towns = %d total",
             frames_per_town, len(active_towns), frames_per_town * len(active_towns))

    plan: Dict[str, List[Tuple[Path, List[int]]]] = defaultdict(list)

    for town in active_towns:
        town_routes = town_to_routes[town]
        frames_per_route = max(1, round(frames_per_town / len(town_routes)))
        LOG.info("  %-10s  %d routes → ~%d frames/route",
                 town, len(town_routes), frames_per_route)

        for route in town_routes:
            indices = list_frame_indices(route)
            if not indices:
                continue
            step = max(1, len(indices) // frames_per_route)
            selected = indices[::step][:frames_per_route]
            plan[town].append((route, selected))

    return plan


# ---------------------------------------------------------------------------
# Shared writer
# ---------------------------------------------------------------------------

def _write_plan(
    plan: Dict[str, List[Tuple[Path, List[int]]]],
    out_dir: Path,
) -> int:
    """
    Execute a sampling plan produced by build_sampling_plan: load frames and
    write one npz per route.  Returns the total number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Output directory: %s", out_dir)

    total_frames = 0
    for town, route_frame_pairs in sorted(plan.items()):
        for route_dir, frame_indices in route_frame_pairs:

            wide_rgbs, segs, cmds, speeds, brakes, fidxs = [], [], [], [], [], []
            tps, tps_prev, tps_next = [], [], []

            for fidx in frame_indices:
                result = load_frame(route_dir, fidx)
                if result is None:
                    continue
                w, s, c, sp, b, tp, tp_p, tp_n = result
                wide_rgbs.append(w)
                segs.append(s)
                cmds.append(c)
                speeds.append(sp)
                brakes.append(b)
                fidxs.append(fidx)
                tps.append(tp)
                tps_prev.append(tp_p)
                tps_next.append(tp_n)

            if not wide_rgbs:
                LOG.warning("No frames loaded from %s — skipping", route_dir)
                continue

            run_name  = f"run_{town}_{route_dir.name}"
            save_path = out_dir / f"{run_name}.npz"

            np.savez_compressed(
                save_path,
                wide_rgb     = np.stack(wide_rgbs,  axis=0),
                seg_red_wide = np.stack(segs,        axis=0),
                cmd          = np.array(cmds,        dtype=np.int32),
                speed        = np.array(speeds,      dtype=np.float32),
                is_brake     = np.array(brakes,      dtype=np.int8),
                frame_idx    = np.array(fidxs,       dtype=np.int32),
                target_point          = np.stack(tps,      axis=0).astype(np.float32),
                target_point_previous = np.stack(tps_prev, axis=0).astype(np.float32),
                target_point_next     = np.stack(tps_next, axis=0).astype(np.float32),
                # narr_rgb / seg_red_narr intentionally absent (TFV6 wide-only)
            )

            n = len(wide_rgbs)
            total_frames += n
            LOG.info("  %-10s  %s  →  %d frames  (%s)", town, route_dir.name, n, save_path.name)

    return total_frames


# ---------------------------------------------------------------------------
# Baseline conversion
# ---------------------------------------------------------------------------

def migrate(
    lead_dir: Path,
    n_frames: int = 3000,
    exclude_towns: Optional[List[str]] = None,
) -> None:
    """Convert LEAD routes to ATOMs baseline npz files (Town05 excluded by default)."""
    if exclude_towns is None:
        exclude_towns = ["Town05"]

    routes = discover_routes(lead_dir)
    if not routes:
        raise FileNotFoundError(f"No valid routes found under {lead_dir}")

    plan      = build_sampling_plan(routes, n_frames, exclude_towns)
    out_dir   = Path(conf.BASELINE_DATA_DIR) / "frames"
    total     = _write_plan(plan, out_dir)
    LOG.info("Done — %d frames written to %s", total, out_dir)


# ---------------------------------------------------------------------------
# Test-set conversion
# ---------------------------------------------------------------------------

def migrate_testset(
    lead_dir: Path,
    n_frames: int = 500,
    include_towns: Optional[List[str]] = None,
) -> None:
    """
    Convert LEAD routes to clean test-set npz files.

    By default samples from Town05 only (the town reserved for testing).
    Output goes to conf.TEST_DATA_DIR / "frames", matching the layout
    expected by LabeledTestLoader / PerturbationApplier.

    Parameters
    ----------
    lead_dir      : root of the LEAD dataset (same as for migrate()).
    n_frames      : target frame count across all included towns (default 500).
    include_towns : towns to sample from (default: ["Town05"]).
    """
    if include_towns is None:
        include_towns = ["Town05"]

    routes = discover_routes(lead_dir)
    if not routes:
        raise FileNotFoundError(f"No valid routes found under {lead_dir}")

    plan    = build_sampling_plan(routes, n_frames, exclude_towns=[], include_towns=include_towns)
    out_dir = Path(conf.TEST_DATA_DIR) / "frames"
    total   = _write_plan(plan, out_dir)
    LOG.info("Done — %d test frames written to %s", total, out_dir)


# ---------------------------------------------------------------------------
# Helpers for alternative (same-distribution) split
# ---------------------------------------------------------------------------

def _build_plan_from_routes(
    routes: List[Path], n_frames: int
) -> Dict[str, List[Tuple[Path, List[int]]]]:
    """
    Build a sampling plan from a pre-assigned list of routes, sampling an equal
    number of frames from every route (route-balanced).  Used by migrate_alt_split.

    Each route receives max(1, round(n_frames / n_routes)) frames, sampled at
    a uniform stride within that route.  This guarantees that all routes
    contribute equally regardless of their individual frame counts or town of
    origin.
    """
    if not routes:
        return {}

    # Ceiling division so we always collect ≥ n_frames before trimming.
    frames_per_route = max(1, -(-n_frames // len(routes)))
    LOG.info(
        "Route-balanced sampling: %d routes × %d frames/route (ceiling) → trim to %d",
        len(routes), frames_per_route, n_frames,
    )

    # Collect per-route, keeping insertion order so the trim is deterministic.
    all_pairs: List[Tuple[Path, int]] = []
    for route_dir in routes:
        indices = list_frame_indices(route_dir)
        if not indices:
            continue
        step = max(1, len(indices) // frames_per_route)
        for idx in indices[::step][:frames_per_route]:
            all_pairs.append((route_dir, idx))

    # Trim to exactly n_frames (drops at most frames_per_route - 1 frames from the tail).
    all_pairs = all_pairs[:n_frames]
    LOG.info("After trim: %d frames from %d routes", len(all_pairs),
             len({r for r, _ in all_pairs}))

    by_route: Dict[Path, List[int]] = defaultdict(list)
    for route_dir, idx in all_pairs:
        by_route[route_dir].append(idx)

    plan: Dict[str, List[Tuple[Path, List[int]]]] = defaultdict(list)
    for route_dir, indices in by_route.items():
        town = _detect_town({}, route_dir) or "unknown"
        plan[town].append((route_dir, sorted(indices)))
    return plan


# ---------------------------------------------------------------------------
# Validation-set conversion
# ---------------------------------------------------------------------------

def migrate_valset(
    lead_dir: Path,
    n_frames: int = 500,
    include_towns: Optional[List[str]] = None,
) -> None:
    """
    Convert LEAD routes to val-set npz files.

    Automatically excludes routes that are already present in
    conf.TEST_DATA_DIR / "frames", so the val set never overlaps with the
    test set.  Both sets must use the same town (default: Town05).

    Output goes to conf.VAL_DATA_DIR / "frames".
    """
    if include_towns is None:
        include_towns = ["Town05"]

    # Identify routes already migrated for the test set so they are not
    # reused for the val set.  The npz stem follows: run_<town>_<route_dir_name>
    # e.g. "run_Town05_Town05_Rep0_Town05_ll_6_...".
    test_frames_dir = Path(conf.TEST_DATA_DIR) / "frames"
    exclude_routes: set = set()
    if test_frames_dir.exists():
        for p in test_frames_dir.glob("run_*.npz"):
            stem = p.stem   # "run_Town05_Town05_Rep0_..."
            for town in include_towns:
                prefix = f"run_{town}_"
                if stem.startswith(prefix):
                    exclude_routes.add(stem[len(prefix):])   # route_dir.name
                    break
        LOG.info(
            "Excluding %d route(s) already in test_data/frames/: %s",
            len(exclude_routes), sorted(exclude_routes),
        )

    routes = discover_routes(lead_dir)
    if not routes:
        raise FileNotFoundError(f"No valid routes found under {lead_dir}")

    plan = build_sampling_plan(
        routes,
        n_frames,
        exclude_towns=[],
        include_towns=include_towns,
        exclude_routes=exclude_routes,
    )
    out_dir = Path(conf.VAL_DATA_DIR) / "frames"
    total = _write_plan(plan, out_dir)
    LOG.info("Done — %d val frames written to %s", total, out_dir)


# ---------------------------------------------------------------------------
# Alternative split: all towns, random route-level 5k/1k/1k partition
# ---------------------------------------------------------------------------

def migrate_alt_split(
    lead_dir: Path,
    baseline_n: int = 5000,
    test_n: int     = 1000,
    val_n: int      = 1000,
    exclude_towns: Optional[List[str]] = None,
    seed: int = conf.RANDOM_SEED,
) -> None:
    """
    Discover all routes from all towns (minus exclude_towns), shuffle with a
    fixed seed, split at the route level into disjoint baseline/test/val sets,
    and write ~baseline_n / test_n / val_n frames to the corresponding
    conf.*_DATA_DIR/frames/ directories.

    All three sets come from the same town distribution so OOD signal comes
    exclusively from perturbations, not domain shift.
    """
    if exclude_towns is None:
        exclude_towns = []

    all_routes = discover_routes(lead_dir)

    # Filter by town
    filtered: List[Path] = []
    for route in all_routes:
        indices = list_frame_indices(route)
        if not indices:
            continue
        meta_path = route / "metas" / f"{indices[0]:04d}.pkl"
        town = None
        if meta_path.exists():
            try:
                town = _detect_town(_load_meta(meta_path), route)
            except Exception:
                pass
        if town is None:
            town = _detect_town({}, route)
        if town in exclude_towns:
            continue
        filtered.append(route)

    LOG.info("%d routes after town filtering (excluded: %s)", len(filtered), exclude_towns)
    if not filtered:
        raise ValueError("No routes remaining after town filtering.")

    # Deterministic shuffle, then proportional route-level split
    rng   = np.random.default_rng(seed)
    order = rng.permutation(len(filtered)).tolist()
    shuffled = [filtered[i] for i in order]

    total_frac = baseline_n + test_n + val_n
    i_test = round(len(shuffled) * baseline_n / total_frac)
    i_val  = round(len(shuffled) * (baseline_n + test_n) / total_frac)

    baseline_routes = shuffled[:i_test]
    test_routes     = shuffled[i_test:i_val]
    val_routes      = shuffled[i_val:]

    LOG.info(
        "Route split — baseline: %d  test: %d  val: %d",
        len(baseline_routes), len(test_routes), len(val_routes),
    )

    # Always write to the _alt directories regardless of EXPERIMENT_VARIANT —
    # alt_split is the alternative split by definition.
    _alt_root = Path(conf._DATA_ROOT)
    for routes_subset, n_target, out_dir_path in [
        (baseline_routes, baseline_n, _alt_root / "baseline_data_alt" / "frames"),
        (test_routes,     test_n,     _alt_root / "test_data_alt"     / "frames"),
        (val_routes,      val_n,      _alt_root / "val_data_alt"      / "frames"),
    ]:
        plan  = _build_plan_from_routes(routes_subset, n_target)
        total = _write_plan(plan, out_dir_path)
        LOG.info("Written %d frames → %s", total, out_dir_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert LEAD CARLA routes to ATOMs npz format.\n"
            "\n"
            "Modes:\n"
            "  baseline   — sample from all towns except Town05 → conf.BASELINE_DATA_DIR/frames/\n"
            "  testset    — sample from Town05 only             → conf.TEST_DATA_DIR/frames/\n"
            "  valset     — sample from Town05, auto-excluding test routes → conf.VAL_DATA_DIR/frames/\n"
            "  both       — run baseline then testset\n"
            "  alt_split  — all towns, random route-level split into\n"
            "               baseline_data_alt / test_data_alt / val_data_alt\n"
            "               (requires EXPERIMENT_VARIANT='alternative' in atoms_config.py)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lead_dir", type=Path, required=True,
        help="Path to unzipped noScenarios directory (or any root containing routes)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "testset", "valset", "both", "alt_split"],
        default="baseline",
        help="What to generate (default: baseline). Use alt_split for the same-distribution split.",
    )
    parser.add_argument(
        "--n_frames", type=int, default=3000,
        help="Target frame count for baseline (default: 3000)",
    )
    parser.add_argument(
        "--exclude_towns", nargs="*", default=["Town05"],
        help="Towns to exclude from baseline (default: Town05)",
    )
    parser.add_argument(
        "--testset_n_frames", type=int, default=500,
        help="Target frame count for test/val set (default: 500)",
    )
    parser.add_argument(
        "--testset_towns", nargs="*", default=["Town05"],
        help="Towns to include in test/val set (default: Town05)",
    )
    # alt_split-specific args
    parser.add_argument(
        "--baseline_n", type=int, default=5000,
        help="[alt_split] Target baseline frame count (default: 5000)",
    )
    parser.add_argument(
        "--test_n", type=int, default=1000,
        help="[alt_split] Target test frame count (default: 1000)",
    )
    parser.add_argument(
        "--val_n", type=int, default=1000,
        help="[alt_split] Target val frame count (default: 1000)",
    )
    args = parser.parse_args()

    if args.mode in ("baseline", "both"):
        migrate(args.lead_dir, args.n_frames, args.exclude_towns)
    if args.mode in ("testset", "both"):
        migrate_testset(args.lead_dir, args.testset_n_frames, args.testset_towns)
    if args.mode == "valset":
        migrate_valset(args.lead_dir, args.testset_n_frames, args.testset_towns)
    if args.mode == "alt_split":
        migrate_alt_split(
            args.lead_dir,
            baseline_n    = args.baseline_n,
            test_n        = args.test_n,
            val_n         = args.val_n,
            # alt_split pools ALL towns by design — do not inherit the
            # --exclude_towns default ("Town05") meant for baseline mode.
            exclude_towns = [],
        )
