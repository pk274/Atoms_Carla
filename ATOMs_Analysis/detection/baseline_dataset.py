"""
baseline_dataset.py
-------------------
Data collection, persistence, and loading for the ATOMs-based anomaly
detection pipeline in CARLA.

Workflow
--------
Phase 1 — Collection (online, inside image_agent.run_step):

    collector = BaselineDataCollector(config, sample_every=10)

    # every step:
    collector.add_frame(wide_rgb, narr_rgb, seg_red, cmd, speed)

    # end of episode:
    collector.save_run()        # writes one .npz file to config.baseline_data_dir
    collector.clear()           # ready for next episode

Phase 2 — Baseline computation (offline, after all collection runs):

    computer = BaselineComputer(config, lrp_model, atoms)
    computer.compute_and_save()   # reads all .npz files, runs ATOMsCarla,
                                  # stores mean + covariance

Phase 3 — Inference monitoring:

    monitor = MahalanobisMonitor(config)
    distance = monitor.score(attention_profile)   # float

File layout on disk
-------------------
config.baseline_data_dir/
    frames/
        run_20240101_120000_000.npz
        run_20240101_120500_001.npz
        ...
    baseline.npz      ← written by BaselineComputer

Each run .npz contains arrays of shape [N, ...] where N = number of
sampled frames from that episode.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla
from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel
from ATOMs_Analysis.utils.visualization_carla import visualize_relevance, visualize_segmentation, visualize_comparative_relevance

from ATOMs_Analysis.utils.lrp_test_suite import LRPTestSuite


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_chw_uint8(img) -> np.ndarray:
    """
    Normalise an image to [3, H, W] uint8 regardless of input layout.

    Accepts:
      - torch.Tensor  [1, 3, H, W], [3, H, W], [H, W, 3]  (float or uint8)
      - np.ndarray    [3, H, W], [H, W, 3]                 (float or uint8)
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.squeeze(img)                     # remove batch dim if present
    if img.ndim == 3 and img.shape[-1] == 3:  # HWC → CHW
        img = img.transpose(2, 0, 1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_hw_uint8(seg) -> np.ndarray:
    """
    Normalise segmentation to [H, W] uint8 regardless of input layout.

    Accepts CARLA BGRA image arrays [H, W, 4] or already-extracted
    red channels [H, W].
    """
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    seg = np.squeeze(seg)
    if seg.ndim == 3:
        # All channels are identical in CARLA semantic images (R=G=B=class_id),
        # so take channel 0 regardless of whether layout is BGRA, BGR, or anything else.
        # Channel 2 is only correct for BGRA — avoid hardcoding it.
        seg = seg[:, :, 0]
    if seg.ndim != 2:
        raise ValueError(
            f"_to_hw_uint8: could not reduce seg to [H, W], "
            f"shape after squeeze+slice is {seg.shape}"
        )
    if seg.dtype != np.uint8:
        seg = seg.astype(np.uint8)
    return seg


# ---------------------------------------------------------------------------
# Phase 1 — Online frame collection
# ---------------------------------------------------------------------------

class BaselineDataCollector:
    """
    Buffers frames during a CARLA episode and saves them as a single .npz
    file at the end of the run.

    Parameters
    ----------
    config      : experiment config object.
                  Required attribute: baseline_data_dir (str or Path).
                  Optional attribute: sample_every (int, default 10).
    sample_every: Save every Nth frame. Overrides config.sample_every if given.

    Saved arrays per run (all shape [N, ...]):
      wide_rgb  : [N, 3, H, W]  uint8   wide-camera RGB
      narr_rgb  : [N, 3, H, W]  uint8   narrow-camera RGB (for fix_context)
      seg_red_wide   : [N, H, W]     uint8   semantic seg red channel (class IDs)
      seg_red_narr   : [N, H, W]     uint8   semantic seg red channel (class IDs)
      cmd       : [N]           int32   navigation command at each frame
      speed     : [N]           float32 ego vehicle speed (m/s)
      frame_idx : [N]           int32   original step index within the episode
    """

    def __init__(self, sample_every: Optional[int] = None):
        self._data_dir = Path(conf.BASELINE_DATA_DIR)
        self._frames_dir = self._data_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

        self._sample_every: int = (
            sample_every
            if sample_every is not None
            else int(conf.IMAGE_SAMPLE_INTERVAL)
        )

        # Internal counters
        self._step: int = 0          # total steps this episode
        self._run_count: int = 0     # number of completed runs

        # Frame buffers
        self._buf_wide:  List[np.ndarray] = []
        self._buf_narr:  List[np.ndarray] = []
        self._buf_seg_wide:   List[np.ndarray] = []
        self._buf_seg_narr:    List[np.ndarray] = []
        self._buf_cmd:   List[int]        = []
        self._buf_speed: List[float]      = []
        self._buf_idx:   List[int]        = []
        self._buf_brake: List[bool]       = []

    # ------------------------------------------------------------------
    # Online interface
    # ------------------------------------------------------------------

    def add_frame(
        self,
        wide_rgb,
        narr_rgb,
        seg_red_wide,
        seg_red_narrow,
        cmd:   int,
        speed: float = 0.0,
        is_brake: bool = False,
    ) -> bool:
        """
        Offer one frame to the collector.

        Only every sample_every-th frame is actually buffered; the rest are
        silently dropped. This keeps memory usage predictable during long runs.

        Parameters
        ----------
        wide_rgb : [1,3,H,W] or [3,H,W] or [H,W,3] tensor/array, uint8 [0-255]
        narr_rgb : same format as wide_rgb
        seg_red  : [H,W] uint8 red channel of CARLA semantic seg, OR
                   [H,W,4] full BGRA semantic seg image (red channel extracted automatically)
        cmd      : int  navigation command (0-5)
        speed    : float  ego speed in m/s

        Returns
        -------
        bool : True if this frame was buffered, False if it was skipped.
        """
        sampled = (self._step % self._sample_every == 0)
        self._step += 1

        if not sampled:
            return False

        self._buf_wide.append(_to_chw_uint8(wide_rgb))
        if narr_rgb is not None:
            self._buf_narr.append(_to_chw_uint8(narr_rgb))
        self._buf_seg_wide.append(_to_hw_uint8(seg_red_wide))
        if seg_red_narrow is not None:
            self._buf_seg_narr.append(_to_hw_uint8(seg_red_narrow))
        self._buf_cmd.append(int(cmd))
        self._buf_speed.append(float(speed))
        self._buf_brake.append(bool(is_brake))
        self._buf_idx.append(self._step - 1)


        if len(self._buf_idx) >= conf.MAX_BASELINE_SIZE:
            print("Buffer full, saving run!")
            self.save_run()
            self.clear()

        return True

    def save_run(self, run_name: Optional[str] = None) -> Path:
        """
        Flush the current buffer to a .npz file and return its path.

        Call this at the end of each CARLA episode. The buffer is NOT
        cleared automatically — call clear() afterwards if you want to
        start fresh for the next episode.

        Parameters
        ----------
        run_name : Optional custom filename stem. Defaults to a timestamp
                   plus a run counter, e.g. 'run_20240101_120000_003'.

        Returns
        -------
        Path to the saved file, or None if the buffer is empty.
        """
        if not self._buf_wide:
            print("[BaselineDataCollector] Buffer is empty — nothing saved.")
            return None

        if run_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{ts}_{self._run_count:03d}"

        save_path = self._frames_dir / f"{run_name}.npz"

        save_kwargs = dict(
            wide_rgb     = np.stack(self._buf_wide,     axis=0),
            seg_red_wide = np.stack(self._buf_seg_wide, axis=0),
            cmd          = np.array(self._buf_cmd,   dtype=np.int32),
            speed        = np.array(self._buf_speed, dtype=np.float32),
            is_brake     = np.array(self._buf_brake, dtype=np.int8),
            frame_idx    = np.array(self._buf_idx,   dtype=np.int32),
        )
        if self._buf_narr:
            save_kwargs['narr_rgb']     = np.stack(self._buf_narr,     axis=0)
        if self._buf_seg_narr:
            save_kwargs['seg_red_narr'] = np.stack(self._buf_seg_narr, axis=0)
        np.savez_compressed(save_path, **save_kwargs)

        n = len(self._buf_wide)
        self._run_count += 1
        print(f"[BaselineDataCollector] Saved {n} frames → {save_path}")
        return save_path

    def clear(self):
        """Clear the frame buffer. Call after save_run() to start a new episode."""
        self._buf_wide.clear()
        self._buf_narr.clear()
        self._buf_seg_wide.clear()
        self._buf_seg_narr.clear()
        self._buf_cmd.clear()
        self._buf_speed.clear()
        self._buf_brake.clear()
        self._buf_idx.clear()
        self._step = 0

    @property
    def n_buffered(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buf_wide)

    @property
    def n_runs_saved(self) -> int:
        """Number of runs saved so far in this session."""
        return self._run_count


# ---------------------------------------------------------------------------
# Phase 2 — Loading
# ---------------------------------------------------------------------------

class BaselineDataLoader:
    """
    Loads saved run files from disk.

    Usage
    -----
    # Load a single run:
    data = BaselineDataLoader.load_run("baseline_data/frames/run_xxx.npz")

    # Load and concatenate all runs in a folder:
    data = BaselineDataLoader.load_all_runs("baseline_data/frames/")

    Returned dict keys:
        wide_rgb  : np.ndarray [N, 3, H, W] uint8
        narr_rgb  : np.ndarray [N, 3, H, W] uint8
        seg_red_wide   : np.ndarray [N, H, W]    uint8
        seg_red_narr   : np.ndarray [N, H, W]    uint8
        cmd       : np.ndarray [N]          int32
        speed     : np.ndarray [N]          float32
        frame_idx : np.ndarray [N]          int32
        run_id    : np.ndarray [N]          int32  (which run each frame came from)
    """

    @staticmethod
    def load_run(filepath: str | Path) -> Dict[str, np.ndarray]:
        """Load a single .npz run file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Run file not found: {filepath}")
        data = np.load(filepath)
        n = data["wide_rgb"].shape[0]
        return {
            "wide_rgb":  data["wide_rgb"],
            "narr_rgb":  data["narr_rgb"],
            "seg_red_wide":   data["seg_red_wide"],
            "seg_red_narr":   data["seg_red_narr"],
            "cmd":       data["cmd"],
            "speed":     data["speed"],
            "is_brake":  data["is_brake"],
            "frame_idx": data["frame_idx"],
            "run_id":    np.zeros(n, dtype=np.int32),
        }

    @staticmethod
    def load_all_runs(
        directory: str | Path,
        pattern: str = "run_*.npz",
        max_runs: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load and concatenate all run files from a directory.

        Parameters
        ----------
        directory : Path to the folder containing run .npz files.
        pattern   : Glob pattern for run files (default 'run_*.npz').
        max_runs  : If set, load at most this many runs (useful for quick tests).

        Returns
        -------
        Concatenated dict with an extra 'run_id' field indicating which run
        each frame came from (0-indexed, ordered by filename).
        """
        directory = Path(directory)
        files = sorted(directory.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {directory}"
            )
        if max_runs is not None:
            files = files[:max_runs]

        parts: List[Dict[str, np.ndarray]] = []
        for run_id, fpath in enumerate(files):
            d = np.load(fpath)
            n = d["wide_rgb"].shape[0]
            parts.append({
                "wide_rgb":  d["wide_rgb"],
                "narr_rgb":  d["narr_rgb"],
                "seg_red_wide":   d["seg_red_wide"],
                "seg_red_narr":   d["seg_red_narr"],
                "cmd":       d["cmd"],
                "speed":     d["speed"],
                "is_brake":  d["is_brake"],
                "frame_idx": d["frame_idx"],
                "run_id":    np.full(n, run_id, dtype=np.int32),
            })
            print(f"  Loaded run {run_id:03d}: {n} frames ← {fpath.name}")

        total = sum(d["wide_rgb"].shape[0] for d in parts)
        print(f"[BaselineDataLoader] Total: {total} frames from {len(parts)} runs.")

        return {k: np.concatenate([d[k] for d in parts], axis=0) for k in parts[0]}

    @staticmethod
    def get_run_files(directory: str | Path, pattern: str = "run_*.npz") -> List[Path]:
        """Return sorted list of run file paths in a directory."""
        return sorted(Path(directory).glob(pattern))

    @staticmethod
    def summary(data: Dict[str, np.ndarray]) -> str:
        """Print a brief summary of a loaded dataset dict."""
        n = data["wide_rgb"].shape[0]
        runs = np.unique(data["run_id"])
        cmd_counts = {int(c): int((data["cmd"] == c).sum()) for c in np.unique(data["cmd"])}
        lines = [
            f"Frames     : {n}",
            f"Runs       : {len(runs)}",
            f"Wide res   : {data['wide_rgb'].shape[2:]}",
            f"Narr res   : {data['narr_rgb'].shape[2:]}",
            f"Commands   : {cmd_counts}",
            f"Speed range: [{data['speed'].min():.1f}, {data['speed'].max():.1f}] m/s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2 — Baseline computation (offline)
# ---------------------------------------------------------------------------

class BaselineComputer:
    """
    Runs ATOMsCarla on a loaded dataset, then computes and saves the
    attention mean and covariance matrix used for Mahalanobis scoring.

    Parameters
    ----------
    config    : config object with baseline_data_dir attribute.
    lrp_model : LRPCameraModel (fix_context already called).
    atoms     : ATOMsCarla instance (will be reset between frames).

    Usage
    -----
    computer = BaselineComputer(config, lrp_model, atoms)
    computer.compute_and_save()      # reads all run files, processes, saves
    """

    def __init__(self, lrp_model: LRPCameraModel, atoms: ATOMsCarla):
        self.data_dir  = conf.BASELINE_DATA_DIR
        self.lrp       = lrp_model
        self.atoms     = atoms

        self._frames_dir  = self.data_dir / "frames"
        self._output_path = self.data_dir / "baseline.npz"
        self.narr_tester = None

    def compute_and_save(
        self,
        cmd_filter: Optional[int] = None,
        max_runs:   Optional[int] = None,
    ) -> Path:
        """
        Process all saved frames through ATOMsCarla and store the
        per-frame attention series, mean, and covariance.

        Parameters
        ----------
        cmd_filter : If set, only process frames with this navigation command.
                     Useful for building command-specific baselines.
        max_runs   : Load at most this many run files (for quick tests).

        Returns
        -------
        Attention series.
        """
        print("[BaselineComputer] Loading dataset...")
        data = BaselineDataLoader.load_all_runs(self._frames_dir, max_runs=max_runs)
        print(BaselineDataLoader.summary(data))

        # Apply command filter
        if cmd_filter is not None:
            mask = data["cmd"] == cmd_filter
            data = {k: v[mask] for k, v in data.items()}
            print(f"[BaselineComputer] After cmd={cmd_filter} filter: {mask.sum()} frames.")

        n_frames = data["wide_rgb"].shape[0]
        if n_frames == 0:
            raise ValueError("No frames remaining after filtering.")
        
        # TESTING ----------------------------------------------------------------------
        #self.narr_tester = LRPTestSuite(atoms=self.atoms, lrp=self.lrp)
        #self.narr_tester.run_all_tests(data)
        #from ATOMs_Analysis.utils.atoms_test_suite import ATOMsTestSuite
        #self.atoms_tester = ATOMsTestSuite(atoms=self.atoms)
        #self.atoms_tester.run_all_tests(data)
        # END OF TESTING ---------------------------------------------------------------
        
        # Process each frame
        attention_series: List[np.ndarray] = []
        t0 = time.time()

        for i in range(n_frames):
            if i % 20 == 0:
                print("Processing frame no.", i)
            wide = torch.from_numpy(data["wide_rgb"][i:i+1]).float()   # [1, 3, H, W]
            narr = torch.from_numpy(data["narr_rgb"][i:i+1]).float()   # [1, 3, H, W])
            seg_wide  = data["seg_red_wide"][i]                                   # [H, W]
            seg_narr = data["seg_red_narr"][i]
            cmd  = int(data["cmd"][i])
            spd = float(data["speed"][i])

            frame_att = self.atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd)   # [num_classes]
            attention_series.append(frame_att)

            if i % conf.PLOT_INTERVAL == 0 and conf.PLOT_SEG_AND_REL:
                savepath_seg_w = conf.BASELINE_DATA_DIR / "segmentation_examples" / f"seg_wide{conf.image_counter}"
                savepath_seg_n = conf.BASELINE_DATA_DIR / "segmentation_examples" / f"seg_narr{conf.image_counter}"
                savepath_rel_w = conf.BASELINE_DATA_DIR / "relevance_examples" / f"rel_wide{conf.image_counter}"
                savepath_rel_n = conf.BASELINE_DATA_DIR / "relevance_examples" / f"rel_narr{conf.image_counter}"
                visualize_segmentation(seg_wide, f"Segmentation Frame {i}", save_path=savepath_seg_w)
                visualize_segmentation(seg_narr, f"Segmentation Frame {i}", save_path=savepath_seg_n)
                if conf.PLOT_COMPARATIVE_REL:
                    comp_map_wide = self.atoms.saliency_data_wide_drive - self.atoms.saliency_data_wide_brake
                    comp_map_narr = self.atoms.saliency_data_narr_drive - self.atoms.saliency_data_narr_brake

                    global_max = max(comp_map_wide.abs().max().item(), comp_map_narr.abs().max().item()) + 1e-12
                    comp_map_wide = comp_map_wide / global_max
                    comp_map_narr = comp_map_narr / global_max

                    rgb_wide = wide[0].permute(1, 2, 0).cpu().detach().numpy()
                    rgb_narr = narr[0].permute(1, 2, 0).cpu().detach().numpy()

                    visualize_comparative_relevance(comp_map_wide, rgb_image=rgb_wide, save_path=f"{savepath_rel_w}_comparative",
                                                    is_brake=self.atoms._last_is_brake)
                    visualize_comparative_relevance(comp_map_narr, rgb_image=rgb_narr, save_path=f"{savepath_rel_n}_comparative",
                                                    is_brake=self.atoms._last_is_brake)
                
                if self.atoms._last_is_brake:
                    visualize_relevance(self.atoms.saliency_data_wide_brake, save_path=savepath_rel_w, is_brake=True)
                    visualize_relevance(self.atoms.saliency_data_narr_brake, save_path=savepath_rel_n, is_brake=True)
                else:
                    visualize_relevance(self.atoms.saliency_data_wide_drive, save_path=savepath_rel_w, is_brake=False)
                    visualize_relevance(self.atoms.saliency_data_narr_drive, save_path=savepath_rel_n, is_brake=False)
                conf.image_counter += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                eta = (n_frames - i - 1) / fps
                print(f"  {i+1}/{n_frames}  ({fps:.1f} fr/s, ETA {eta:.0f}s)")

        self.atoms.reset()

        # Stack and compute statistics
        series = np.stack(attention_series, axis=0)    # [N, num_classes]
        mean   = series.mean(axis=0)                   # [num_classes]
        cov    = np.cov(series.T)                      # [num_classes, num_classes]

        # Save the first narr_rgb frame as a reference for fix_context.
        # Any typical frame works — we just need a representative narrow image
        # to initialize the frozen context embedding.
        reference_narr = data["narr_rgb"][0:1]   # [1, 3, H, W]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._output_path,
            series      = series.astype(np.float32),
            mean        = mean.astype(np.float32),
            cov         = cov.astype(np.float32),
            class_ids   = np.array(self.atoms.class_ids,   dtype=np.int32),
            class_names = np.array(self.atoms.class_names, dtype=object),
            cmd_filter  = np.array([cmd_filter if cmd_filter is not None else -1]),
            n_frames    = np.array([n_frames]),
            reference_narr = reference_narr.astype(np.uint8),
        )

        elapsed = time.time() - t0
        print(f"[BaselineComputer] Done. {n_frames} frames in {elapsed:.1f}s.")
        print(f"  Baseline saved → {self._output_path}")
        return series


    @torch.no_grad()
    def diagnose_lrp_policy_match(self, lrp, wide_rgb, narr_rgb, cmd, spd):
        cm = lrp._model_eval
        lrp.update_context(wide_rgb, narr_rgb, spd)

        x = torch.from_numpy(wide_rgb).float().to(lrp.device)
        if x.dim() == 3: x = x.unsqueeze(0)
        if x.shape[-1] == 3: x = x.permute(0, 3, 1, 2).contiguous()

        # --- POLICY path ---
        p_wide_post  = cm.backbone_wide(cm.normalize(x / 255.))
        p_wide_pool  = p_wide_post.mean(dim=[2, 3])                          # [1, 512]
        p_narr_post  = cm.backbone_narr(cm.normalize(lrp._current_narr / 255.))
        p_narr_ctx   = cm.bottleneck_narr(p_narr_post.mean(dim=[2, 3]))      # [1, 64]
        p_concat     = torch.cat([p_wide_pool, p_narr_ctx], dim=1)           # [1, 576]
        p_logits     = cm.act_head(p_concat)                                 # [1, 312]

        # --- MODEL_LRP path ---
        m = lrp.model_lrp
        m_wide_post  = m.backbone(m.normalize(x / 255.))
        m_wide_pool  = m.flatten(m.pool(m_wide_post))                        # [1, 512]
        m_narr_ctx   = m.head.fixed_context                                  # [1, 64], frozen
        m_concat     = torch.cat([m_wide_pool, m_narr_ctx], dim=1)           # [1, 576]
        m_logits     = m(x)                                                  # [1, 312]

        def diff(a, b, name):
            d = (a - b).abs()
            print(f"{name:20s} max={d.max().item():.3e}  mean={d.mean().item():.3e}  "
                  f"shape={tuple(a.shape)}")

        diff(p_wide_pool, m_wide_pool, "wide GAP")
        diff(p_narr_ctx,  m_narr_ctx,  "narr ctx (64-d)")
        diff(p_concat,    m_concat,    "concat (576-d)")
        diff(p_logits,    m_logits,    "act_head out (312)")

        # And the actual policy-return tensors at (cmd, spd):
        s, t, b = cm.policy(x, lrp._current_narr, cmd)
        raw = m_logits.view(1, lrp.num_cmds, lrp.num_speeds,
                            lrp.num_steers + lrp.num_throts + 1)
        diff(s, raw[0, cmd, :, :lrp.num_steers],                              "steer slice")
        diff(t, raw[0, cmd, :, lrp.num_steers:lrp.num_steers+lrp.num_throts], "throt slice")
        diff(b, raw[0, cmd, :, -1],                                           "brake slice")
