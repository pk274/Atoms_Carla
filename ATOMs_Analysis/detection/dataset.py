"""
detection/dataset.py
--------------------
Test dataset collection and post-processing perturbation pipeline.

Workflow
--------
Phase 1 — Collect a clean test set (same interface as BaselineDataCollector):

    collector = TestDataCollector(config, sample_every=1)
    # every step (typically want denser sampling for test set):
    collector.add_frame(wide_rgb, narr_rgb, seg_red, cmd, speed)
    collector.save_run()          # → config.test_data_dir/frames/run_xxx.npz
    collector.clear()

Phase 2 — Apply perturbations offline to create a labeled dataset:

    spec = PerturbationSpec([
        PerturbationEntry(perturbation="gaussian_noise", intensity=0.3, fraction=0.30),
        PerturbationEntry(perturbation="brightness",     intensity=4.0, fraction=0.30),
        PerturbationEntry(perturbation=None,                            fraction=0.40),
    ])
    applier = PerturbationApplier(config, perturbation_manager)
    applier.apply(spec, seed=42)
    # → config.test_data_dir/test_labeled.npz

Phase 3 — Load labeled dataset for detector evaluation:

    data = LabeledTestLoader.load(config)
    # data["wide_rgb"], data["label"], data["perturbation"], data["intensity"], …

File layout
-----------
config.test_data_dir/
    frames/
        run_xxx.npz    (clean frames, same schema as baseline frames/
    test_labeled.npz   (output of PerturbationApplier)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf


# ---------------------------------------------------------------------------
# Perturbation specification
# ---------------------------------------------------------------------------

@dataclass
class PerturbationEntry:
    """
    One component of a perturbation mix.

    Parameters
    ----------
    fraction    : float  proportion of frames to assign to this entry (must sum to 1.0 across entries)
    perturbation: str or None  perturbation type recognised by your PerturbationManager.
                               None = clean frames.
    intensity   : float  passed to pm.perturb_wide_image(..., intensity=intensity)
    """
    fraction:     float
    perturbation: Optional[str] = None
    intensity:    float         = 0.0
    fgsm_target:  str           = "steer_right"


@dataclass
class PerturbationSpec:
    """
    Full mixing specification for one labeled test dataset.

    Example — 40% clean, 30% gaussian noise, 30% brightness:
        spec = PerturbationSpec([
            PerturbationEntry(fraction=0.40, perturbation=None),
            PerturbationEntry(fraction=0.30, perturbation="gaussian_noise", intensity=0.3),
            PerturbationEntry(fraction=0.30, perturbation="brightness",     intensity=4.0),
        ])
    """
    entries: List[PerturbationEntry]

    def __post_init__(self):
        total = sum(e.fraction for e in self.entries)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"PerturbationSpec fractions must sum to 1.0, got {total:.4f}"
            )

    @property
    def n_entries(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Test data collector  (mirrors BaselineDataCollector from baseline.py)
# ---------------------------------------------------------------------------

class TestDataCollector:
    """
    Collects clean test frames during CARLA episode(s).

    Identical interface to BaselineDataCollector; kept as a separate class
    so that baseline and test data are saved to separate directories and
    are never mixed up.

    Parameters
    ----------
    sample_every      : save every Nth frame (default 1 = every frame, since
                        test sets are typically smaller and denser than baseline).
    perturbation_name : name of the live perturbation being recorded (e.g.
                        "phantom_obstacle").  Used as part of the filename when
                        save_run(live_perturbation=True) is called.
    """

    def __init__(self, sample_every: int = 1, perturbation_name: str = ""):
        self._data_dir            = conf.TEST_DATA_DIR
        self._frames_dir          = self._data_dir / "frames"
        self._live_pert_frames_dir = self._data_dir / "live_pert_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._live_pert_frames_dir.mkdir(parents=True, exist_ok=True)
        self._sample_every        = sample_every
        self._perturbation_name   = perturbation_name

        self._step:      int         = 0
        self._run_count: int         = 0
        self._buf_wide:  List        = []
        self._buf_narr:  List        = []
        self._buf_seg_wide:   List        = []
        self._buf_seg_narr:   List        = []
        self._buf_cmd:       List[int]   = []
        self._buf_speed:     List[float] = []
        self._buf_brake:     List[bool]  = []
        self._buf_idx:       List[int]   = []
        self._buf_perturbed: List[bool]  = []

    def add_frame(self, wide_rgb, narr_rgb, seg_red_wide, seg_red_narr, cmd: int, speed: float = 0.0,
                  is_brake: bool = False, live_perturbation: bool = False,
                  is_perturbed: bool = False) -> bool:
        sampled = (self._step % self._sample_every == 0)
        self._step += 1
        if not sampled:
            return False
        self._buf_wide.append(_to_chw_uint8(wide_rgb))
        if narr_rgb is not None:
            self._buf_narr.append(_to_chw_uint8(narr_rgb))
        self._buf_seg_wide.append(_to_hw_uint8(seg_red_wide))
        if seg_red_narr is not None:
            self._buf_seg_narr.append(_to_hw_uint8(seg_red_narr))
        self._buf_cmd.append(int(cmd))
        self._buf_speed.append(float(speed))
        self._buf_brake.append(bool(is_brake))
        self._buf_idx.append(self._step - 1)
        self._buf_perturbed.append(bool(is_perturbed))
        if len(self._buf_idx) >= conf.MAX_TEST_SIZE or (live_perturbation and len(self._buf_idx) >= conf.MAX_LIVE_PERT_SIZE):
            print("Buffer full, saving run!")
            self.save_run(live_perturbation = live_perturbation)
            self.clear()
        return True

    def save_run(self, run_name: Optional[str] = None, live_perturbation: bool = False) -> Optional[Path]:
        if not self._buf_wide:
            print("[TestDataCollector] Buffer is empty — nothing saved.")
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if live_perturbation:
            pert_slug = self._perturbation_name if self._perturbation_name else "unknown"
            if run_name is None:
                run_name = f"run_{pert_slug}_live_pert_{ts}_{self._run_count:03d}"
            save_path = self._live_pert_frames_dir / f"{run_name}.npz"
        else:
            if run_name is None:
                run_name = f"run_{ts}_{self._run_count:03d}"
            save_path = self._frames_dir / f"{run_name}.npz"
        save_kwargs = dict(
            wide_rgb     = np.stack(self._buf_wide),
            seg_red_wide = np.stack(self._buf_seg_wide),
            cmd          = np.array(self._buf_cmd,       dtype=np.int32),
            speed        = np.array(self._buf_speed,     dtype=np.float32),
            is_brake     = np.array(self._buf_brake,     dtype=np.int8),
            frame_idx    = np.array(self._buf_idx,       dtype=np.int32),
            is_perturbed = np.array(self._buf_perturbed, dtype=np.int8),
        )
        if self._buf_narr:
            save_kwargs['narr_rgb']     = np.stack(self._buf_narr)
        if self._buf_seg_narr:
            save_kwargs['seg_red_narr'] = np.stack(self._buf_seg_narr)
        np.savez_compressed(save_path, **save_kwargs)
        n = len(self._buf_wide)
        self._run_count += 1
        print(f"[TestDataCollector] Saved {n} frames → {save_path}")
        return save_path

    def clear(self):
        self._buf_wide.clear(); self._buf_narr.clear(); self._buf_seg_wide.clear(); self._buf_seg_narr.clear()
        self._buf_cmd.clear();  self._buf_speed.clear(); self._buf_idx.clear(); self._buf_brake.clear()
        self._buf_perturbed.clear()
        self._step = 0


# ---------------------------------------------------------------------------
# Post-processing perturbation applier
# ---------------------------------------------------------------------------

class PerturbationApplier:
    """
    Loads a clean test dataset and applies perturbations offline to produce
    a labeled dataset ready for detector evaluation.

    Each frame is assigned to exactly one perturbation entry according to
    the spec fractions. Assignment is done by random shuffle so that
    perturbed and clean frames are randomly distributed along the temporal
    axis — important for avoiding any temporal autocorrelation artifacts.

    Parameters
    ----------
    config              : config with test_data_dir attribute.
    perturbation_manager: your PerturbationManager instance.
                          Must expose:
                            pm.perturb_wide_image(tensor, perturbation, intensity) -> tensor
                            pm.perturb_narrow_image(tensor, perturbation, intensity) -> tensor
    """

    def __init__(self, perturbation_manager, model=None):
        self._data_dir  = conf.TEST_DATA_DIR
        self._pm        = perturbation_manager
        self._model    = model
        self._out_path  = self._data_dir / "test_labeled.npz"

    def apply(
        self,
        spec:       PerturbationSpec,
        seed:       int = 42,
        max_runs:   Optional[int] = None,
        output_name: str = "test_labeled",
    ) -> Path:
        """
        Build and save a labeled test dataset.

        Parameters
        ----------
        spec        : PerturbationSpec describing the mixing fractions.
        seed        : random seed for reproducible frame-to-entry assignment.
        max_runs    : if set, load at most this many run files (for quick tests).
        output_name : stem of the output .npz file.

        Returns
        -------
        Path to the saved labeled file.

        Output .npz arrays (all length N):
            wide_rgb    : [N, 3, H, W] uint8  — perturbed where applicable
            narr_rgb    : [N, 3, H, W] uint8  — perturbed where applicable
            seg_red_wide     : [N, H, W]    uint8  — always clean (seg is ground truth)
            seg_red_narr     : [N, H, W]    uint8  — always clean (seg is ground truth)
            cmd         : [N]          int32
            speed       : [N]          float32
            frame_idx   : [N]          int32
            run_id      : [N]          int32
            label       : [N]          int32   0 = clean, 1 = perturbed
            perturbation: [N]          object  perturbation name or "clean"
            intensity   : [N]          float32
        """
        print("[PerturbationApplier] Loading clean test frames...")
        raw = _load_all_runs(self._data_dir / "frames", max_runs=max_runs)
        n   = raw["wide_rgb"].shape[0]
        print(f"  {n} frames loaded.")

        has_narr = raw["narr_rgb"] is not None

        # Assign each frame to a spec entry
        assignments = self._assign_frames(n, spec, seed)  # [N] int indices into spec.entries

        # Build output arrays
        out_wide   = np.empty_like(raw["wide_rgb"])
        out_narr   = np.empty_like(raw["narr_rgb"]) if has_narr else None
        labels     = np.zeros(n, dtype=np.int32)
        pert_names = np.empty(n, dtype=object)
        intensities = np.zeros(n, dtype=np.float32)

        for entry_idx, entry in enumerate(spec.entries):
            frame_idxs = np.where(assignments == entry_idx)[0]
            is_clean   = entry.perturbation is None

            is_fgsm = (not is_clean) and entry.perturbation == "fgsm"
            is_pgd  = (not is_clean) and entry.perturbation == "pgd"

            if (is_fgsm or is_pgd) and self._model is None:
                raise ValueError(
                    "Adversarial attack perturbation requires a model. "
                    "Pass model= to PerturbationApplier()."
                )
            if (is_fgsm or is_pgd) and not has_narr:
                raise ValueError(
                    f"Perturbation '{entry.perturbation}' requires a narrow camera "
                    "(WoR dual-camera model).  TFV6 single-stream data is not supported "
                    "for adversarial attacks."
                )

            for fi in frame_idxs:
                wide = torch.from_numpy(raw["wide_rgb"][fi:fi+1]).float()
                narr = torch.from_numpy(raw["narr_rgb"][fi:fi+1]).float() if has_narr else None
                cmd  = torch.tensor(int(raw["cmd"][fi]))

                if is_fgsm:
                    # FGSM needs both images and the model simultaneously.
                    # epsilon reuses the intensity field — set it to your
                    # pixel-budget value (e.g. 8.0, 16.0).
                    wide, narr = self._pm.fgsm_attack(
                        model          = self._model,
                        wide_rgbs_     = wide,
                        narr_rgb_      = narr,
                        cmd_value      = cmd,
                        target         = entry.fgsm_target,   # <<< "steer_right" etc.
                        epsilon        = entry.intensity,
                        apply_to_wide  = True,
                        apply_to_narrow= True,
                    )

                elif is_pgd:
                    wide, narr = self._pm.pgd_attack(
                        model          = self._model,
                        wide_rgbs_     = wide,
                        narr_rgb_      = narr,
                        cmd_value      = cmd,
                        target         = entry.fgsm_target,   # <<< "steer_right" etc.
                        epsilon        = entry.intensity,
                        apply_to_wide  = True,
                        apply_to_narrow= True,
                    )

                elif not is_clean:
                    if has_narr:
                        wide = self._pm.perturb_wide_image(
                            wide, perturbation=entry.perturbation, intensity=entry.intensity
                        )
                        narr = self._pm.perturb_narrow_image(
                            narr, perturbation=entry.perturbation, intensity=entry.intensity
                        )
                    else:
                        # TFV6: single concatenated [3, H, W_total] image — use dedicated API
                        perturbed = self._pm.perturb_tfv6_image(
                            raw["wide_rgb"][fi],
                            perturbation = entry.perturbation,
                            intensity    = entry.intensity,
                            n_cameras    = raw["wide_rgb"].shape[-1] // raw["wide_rgb"].shape[-2],
                        )
                        wide = torch.from_numpy(perturbed).unsqueeze(0).float()

                out_wide[fi] = _to_chw_uint8(wide)
                if has_narr:
                    out_narr[fi] = _to_chw_uint8(narr)
                labels[fi]      = 0 if is_clean else 1
                pert_names[fi]  = "clean" if is_clean else entry.perturbation
                intensities[fi] = 0.0 if is_clean else entry.intensity

            tag = "clean" if is_clean else f"{entry.perturbation}@{entry.intensity}"
            print(f"  Entry '{tag}': {len(frame_idxs)} frames.")

        self._out_path = self._data_dir / f"{output_name}.npz"
        save_kwargs = dict(
            wide_rgb     = out_wide,
            seg_red_wide = raw["seg_red_wide"],          # always clean
            cmd          = raw["cmd"],
            speed        = raw["speed"],
            is_brake     = raw["is_brake"],
            frame_idx    = raw["frame_idx"],
            run_id       = raw["run_id"],
            label        = labels,
            perturbation = pert_names,
            intensity    = intensities,
        )
        if has_narr:
            save_kwargs["narr_rgb"]     = out_narr
            save_kwargs["seg_red_narr"] = raw["seg_red_narr"]
        np.savez_compressed(self._out_path, **save_kwargs)

        n_perturbed = int(labels.sum())
        print(
            f"[PerturbationApplier] Saved {n} frames "
            f"({n - n_perturbed} clean, {n_perturbed} perturbed) "
            f"→ {self._out_path}"
        )
        return self._out_path

    @staticmethod
    def _assign_frames(n: int, spec: PerturbationSpec, seed: int) -> np.ndarray:
        """
        Assign each of the N frames to a spec entry index.
        Fractions are converted to exact counts (last entry absorbs rounding).
        Assignment is random-shuffled for temporal decorrelation.
        """
        rng    = np.random.default_rng(seed)
        counts = [int(round(e.fraction * n)) for e in spec.entries]
        # Correct rounding so counts sum exactly to n
        counts[-1] += n - sum(counts)

        assignments = np.concatenate([
            np.full(cnt, i, dtype=np.int32)
            for i, cnt in enumerate(counts)
        ])
        rng.shuffle(assignments)
        return assignments


# ---------------------------------------------------------------------------
# Labeled test loader
# ---------------------------------------------------------------------------

class LabeledTestLoader:
    """
    Loads a labeled test dataset produced by PerturbationApplier.

    Usage
    -----
    data = LabeledTestLoader.load(config)
    # or
    data = LabeledTestLoader.load_path("test_data/test_labeled.npz")

    Available keys:
        wide_rgb, narr_rgb, seg_red, cmd, speed, frame_idx, run_id,
        label, perturbation, intensity
    """

    @staticmethod
    def load(name: str = "test_labeled") -> Dict[str, np.ndarray]:
        path = conf.TEST_DATA_DIR / f"{name}.npz"
        return LabeledTestLoader.load_path(path)

    @staticmethod
    def load_live_pert(perturbation_name: str) -> Dict[str, np.ndarray]:
        """
        Load all live-perturbation frames for a given perturbation type.

        Looks for files matching  live_pert_frames/run_{perturbation_name}_live_pert_*.npz
        and concatenates them in sorted order.
        """
        frames_dir = conf.TEST_DATA_DIR / "live_pert_frames"
        pattern    = f"run_{perturbation_name}_live_pert_*.npz"
        return _load_all_runs(frames_dir, pattern=pattern)

    @staticmethod
    def load_path(path: str | Path) -> Dict[str, np.ndarray]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Labeled test file not found: {path}\n"
                "Run PerturbationApplier.apply() first."
            )
        data = np.load(path, allow_pickle=True)
        d = dict(data)
        # Normalise optional keys — absent for TFV6 (wide-only) data.
        d.setdefault("narr_rgb", None)
        d.setdefault("seg_red_narr", None)
        return d

    @staticmethod
    def summary(data: Dict) -> str:
        n    = data["wide_rgb"].shape[0]
        labs = data["label"]
        perts = data["perturbation"]
        unique_perts = {str(p): int((perts == p).sum()) for p in np.unique(perts)}
        lines = [
            f"Total frames : {n}",
            f"Clean        : {int((labs == 0).sum())}",
            f"Perturbed    : {int((labs == 1).sum())}",
            f"By type      : {unique_perts}",
        ]
        return "\n".join(lines)

    @staticmethod
    def split_by_perturbation(
        data: Dict,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split a labeled dataset into per-perturbation-type sub-dicts.
        Useful for evaluating detector performance on each perturbation type
        independently.

        Returns
        -------
        dict mapping perturbation name (str) → sub-dict with same keys as data.
        """
        result = {}
        for p in np.unique(data["perturbation"]):
            mask = data["perturbation"] == p
            result[str(p)] = {k: v[mask] if v is not None else None
                               for k, v in data.items()}
        return result


# ---------------------------------------------------------------------------
# Shared helpers (mirrors baseline.py)
# ---------------------------------------------------------------------------

def _to_chw_uint8(img) -> np.ndarray:
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_hw_uint8(seg) -> np.ndarray:
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    seg = np.squeeze(seg)
    if seg.ndim == 3:
        seg = seg[:, :, 2]
    if seg.dtype != np.uint8:
        seg = seg.astype(np.uint8)
    return seg


def _load_all_runs(
    directory: Path,
    pattern: str = "run_*.npz",
    max_runs: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    if max_runs is not None:
        files = files[:max_runs]
    parts = []
    for run_id, fp in enumerate(files):
        d  = np.load(fp)
        n  = d["wide_rgb"].shape[0]
        parts.append({
            "wide_rgb":     d["wide_rgb"],
            "narr_rgb":     d["narr_rgb"]     if "narr_rgb"     in d else None,
            "seg_red_wide": d["seg_red_wide"],
            "seg_red_narr": d["seg_red_narr"] if "seg_red_narr" in d else None,
            "cmd":          d["cmd"],
            "speed":        d["speed"],
            "is_brake":     d["is_brake"],
            "frame_idx":    d["frame_idx"],
            "run_id":       np.full(n, run_id, dtype=np.int32),
        })

    def _concat(key):
        arrays = [p[key] for p in parts]
        if all(a is None for a in arrays):
            return None
        return np.concatenate(arrays, axis=0)

    return {k: _concat(k) for k in parts[0]}

