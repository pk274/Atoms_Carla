"""
atoms_carla.py
--------------
ATOMs (Attention-Oriented Metrics) for the World on Rails CameraModel
operating in the CARLA driving simulator.

Adapted from the reference ATOMs / ATOMs_uitb implementation (Beylier et al., 2024).
The structure and computation stay as close as possible to the reference;
only the following aspects differ by necessity:

  Online processing
    Frames are fed one at a time via process_frame(). No pre-collected
    dataset, no object-presence filtering. Absent objects receive zero
    attention naturally (mask product = 0).

  Segmentation source
    CARLA's semantic segmentation camera encodes class IDs in the red
    channel (value x → tag x, 0–22 for CARLA 0.9.x). These are converted
    to binary masks [num_classes, H, W] per frame.

  FC layer definition
    The FC layer is the second 256-dim hidden layer of act_head — output of
    act_head[3] (second ReLU), just before the final linear projection.
    This is the closest equivalent to the Fc layer in the paper and is the
    layer that represents "the final world model on which the agent chooses
    its action." Requires the _attribute_to_fc / _WideCameraToFC additions
    to lrp_camera_model.py.

  Command conditioning
    LRP1 (output→FC) is initialized at the logits for the currently active
    navigation command. LRP2 (FC node→input) is command-independent.

  Data storage
    Per-frame series stored as a list of numpy arrays; accessible as a
    pandas DataFrame via get_series_df(). More convenient for analysis
    than raw tensors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ATOMs_Analysis.utils.visualization_carla import visualize_relevance, visualize_segmentation
from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel


# ---------------------------------------------------------------------------
# Semantic class registries
# ---------------------------------------------------------------------------

# Raw CARLA class IDs (CARLA 0.9.x, tags 0-28). Used for WoR data.
CARLA_CLASSES: Dict[int, str] = {
    0:  "Unlabeled",
    1:  "Roads",
    2:  "SideWalks",
    3:  "Building",
    4:  "Wall",
    5:  "Fence",
    6:  "Pole",
    7:  "TrafficLight",
    8:  "TrafficSign",
    9:  "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bycicle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail"
}

# TransFuser/LEAD grouped class IDs (save_grouped_semantic=True). Used for TFV6 data.
# The LEAD dataset maps CARLA's 32 raw tags to these 10 classes via
# SEMANTIC_SEGMENTATION_CONVERTER before saving segmentation PNGs.
TFV6_CLASSES: Dict[int, str] = {
    0: "Unlabeled",       # sky, buildings, vegetation, sidewalks, terrain, …
    1: "Vehicle",         # Car, Truck, Bus, Motorcycle
    2: "Road",
    3: "TrafficLight",
    4: "Pedestrian",
    5: "RoadLine",        # lane markings
    6: "Obstacle",        # cones / traffic warnings
    7: "SpecialVehicle",
    8: "StopSign",
    9: "Biker",           # Rider, Bicycle
}

# Driving-relevant subset for WoR reduced-class mode.
REDUCED_CLASS_IDS: List[int] = [12, 24, 1, 2, 14, 7, 9, 11]
# Pedestrian, RoadLine, Road, SideWalk, Car, TrafficLight, Vegetation, Sky

NUM_CARLA_CLASSES: int = len(CARLA_CLASSES)   # 29


# ---------------------------------------------------------------------------
# Helper: cumulative-relevance filter
# Mirrors relevance_filter() from the reference implementation.
# ---------------------------------------------------------------------------

def _relevance_filter(r: torch.Tensor, p: float) -> List[int]:
    """
    Return indices of the top neurons that together account for at least
    fraction p of total absolute relevance, sorted by descending relevance.

    This mirrors the reference filter_relevance utility and implements the
    90%-mass selection described in the paper (Section 2, step 1).
    """
    r = r.abs()
    total = r.sum().item()
    if total == 0.0:
        return []
    sorted_idx = torch.argsort(r, descending=True)
    cumsum = torch.cumsum(r[sorted_idx], dim=0) / total
    # number of neurons whose cumsum is strictly below p, plus one more to
    # cross the threshold — matches the reference behaviour
    n_keep = int((cumsum < p).sum().item()) + 1
    n_keep = min(n_keep, len(r))
    print(f"[relevance_filter] keeping {n_keep}/{len(r)} neurons ({100*n_keep/len(r):.1f}%)")
    kept_mass   = r[sorted_idx[:n_keep]].sum().item()
    active      = int((r > 0).sum().item())
    print(
        f"[relevance_filter] total_mass={total:.4f}  "
        f"kept={n_keep}/{len(r)} neurons ({100*n_keep/len(r):.1f}%)  "
        f"kept_mass={kept_mass:.4f}  "
        f"top_neuron={r[sorted_idx[0]].item():.4f}  "
        f"active={active}/{len(r)}"
    )
    return sorted_idx[:n_keep].tolist()


# ---------------------------------------------------------------------------
# Helper: segmentation image → binary masks
# ---------------------------------------------------------------------------

def seg_to_masks(seg_red: np.ndarray, class_ids: List[int]) -> torch.Tensor:
    """
    Convert a CARLA semantic segmentation red-channel image to binary masks.

    Parameters
    ----------
    seg_red   : np.ndarray [H, W], dtype uint8
                Red channel of the semantic segmentation image.
                Pixel value x → semantic tag x (CARLA convention).
    class_ids : List[int]
                Ordered list of class IDs to generate masks for.

    Returns
    -------
    torch.Tensor [num_classes, H, W], float32, values in {0.0, 1.0}
    """
    if isinstance(seg_red, torch.Tensor):
        seg_red_np = seg_red.detach().numpy()
    else:
        seg_red_np = seg_red
    masks = np.stack(
        [(seg_red_np == cid).astype(np.float32) for cid in class_ids],
        axis=0,
    )
    return torch.from_numpy(masks)


# ---------------------------------------------------------------------------
# ATOMsCarla
# ---------------------------------------------------------------------------

class ATOMsCarla:
    """
    Online ATOMs for the World on Rails CameraModel in CARLA.

    Stays structurally close to the reference ATOMs class. The four analysis
    modes, the two-pass LRP procedure, and the node-filtering step are
    unchanged. What changes is the data interface (online, per-frame) and
    the segmentation source (CARLA semantic camera).

    Parameters
    ----------
    lrp_model     : LRPCameraModel
                    Already initialized; fix_context() must have been called.
    p_relevance   : float
                    Fraction of FC relevance mass used to select nodes.
                    0.9 (90%) matches the paper default.
    default_cmd   : int
                    Navigation command used when cmd is not passed to
                    process_frame(). 3 = FOLLOW_LANE (World on Rails).
    mode_analysis : int
                    Analysis mode (mirrors reference):
                      1  node-level     LRP1→filter→LRP2 per node  [paper default]
                      2  layer-level    FC→input, single map
                      3  node-output    output→input directly
                      4  layer-output   output→input, single map
    use_reduced   : bool
                    If True, track only the 7 driving-relevant classes
                    (REDUCED_CLASS_IDS) instead of all 29.

    Usage
    -----
    atoms = ATOMsCarla(lrp_model, p_relevance=0.9, default_cmd=3)

    # Inside image_agent.run_step, once fix_context has been called:
    atoms.process_frame(
        wide_rgb = wide_rgbs_,          # [1, 3, H, W] uint8
        seg_red  = seg_array[:, :, 2],  # [H, W] uint8, red channel of CARLA seg
        cmd      = cmd,                 # int from waypointer
    )

    # After episode:
    df   = atoms.get_series_df()     # per-frame DataFrame
    mean = atoms.get_hierarchical()  # normalized mean over episode [num_classes]
    atoms.reset()

    Notes on computational cost
    ---------------------------
    Mode 1 (node-level) runs one LRP1 pass plus one LRP2 pass per selected
    node (typically 10–40 nodes for p=0.9). This is the most expensive mode.
    For live deployment in CARLA, consider running atoms.process_frame() only
    every N steps, or using mode 4 (single pass) for real-time monitoring and
    mode 1 for post-hoc analysis on recorded frames.
    """

    def __init__(
        self,
        lrp_model:     LRPCameraModel,
        p_relevance:   float = 0.9,
        default_cmd:   int   = 3,
        mode_analysis: int   = 2,
        use_reduced:   bool  = False,
        class_map:     Optional[Dict[int, str]] = None,
    ):
        self.lrp           = lrp_model
        self.p_relevance   = p_relevance
        self.default_cmd   = default_cmd
        self.mode_analysis = mode_analysis

        # Class configuration — use the supplied class_map if provided, else CARLA defaults.
        # For TFV6 data (save_grouped_semantic=True) pass class_map=TFV6_CLASSES so that
        # the 10 TransFuser class IDs are interpreted correctly.
        _class_map = class_map if class_map is not None else CARLA_CLASSES
        self.class_map = _class_map          # stored for callers (e.g. visualize_segmentation)
        if use_reduced and class_map is None:
            self.class_ids = REDUCED_CLASS_IDS
        else:
            self.class_ids = list(_class_map.keys())
        self.class_names = [_class_map.get(c, f"Class_{c}") for c in self.class_ids]
        self.num_classes = len(self.class_ids)

        # Dispatch table — mirrors reference ATOMs
        self._compute_sub = {
            1: self._compute_node_level,
            2: self._compute_layer_level,
            3: self._compute_node_output_level,
        }.get(mode_analysis, self._compute_node_level)

        # Runtime state (reset between episodes)
        self._hierarchical: np.ndarray   = np.zeros(self.num_classes, dtype=np.float64)
        self._frame_series: List         = []   # list of np.ndarray [num_classes]
        self._frame_cmds:   List[int]    = []
        self._n_frames:     int          = 0

        self.saliency_data_wide_default = None   # default (softmax) seed — feeds _hierarchical
        self.saliency_data_narr_default = None
        self.saliency_data_wide_brake = None     # forced brake seed (PLOT_COMPARATIVE_REL only)
        self.saliency_data_narr_brake = None
        self.saliency_data_wide_drive = None     # forced drive seed (PLOT_COMPARATIVE_REL only)
        self.saliency_data_narr_drive = None

        self._frame_brake: List[bool] = []
        self._last_is_brake: bool = False

        self._frame_wide_frac = []
        self._last_wide_frac = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        wide_rgb:  torch.Tensor,
        narr_rgb:  torch.Tensor,
        seg_wide:  np.ndarray,
        seg_narr:  np.ndarray,
        cmd:       Optional[int]   = None,
        spd:       Optional[float] = None,
        data:      Optional[dict]  = None,
    ) -> np.ndarray:
        if cmd is None:
            cmd = self.default_cmd
        if spd is None:
            spd = 0.0

        # Pass data dict if the LRP model supports it (TFV6 Option B)
        if data is not None and hasattr(self.lrp, '_data_cache'):
            self.lrp.update_context(wide_rgb, narr_rgb, spd, data=data)
        elif hasattr(self.lrp, '_data_cache'):
            # TFV6 without full data dict — pass cmd so the command token is
            # a valid one-hot (not all-zeros), reducing conditioning distortion.
            self.lrp.update_context(wide_rgb, narr_rgb, spd, cmd=cmd)
        else:
            self.lrp.update_context(wide_rgb, narr_rgb, spd)

        self._current_masks_wide = seg_to_masks(seg_wide, self.class_ids)
        self._current_masks_narr = (
            seg_to_masks(seg_narr, self.class_ids) if seg_narr is not None else None
        )
        self._current_spd        = spd
        prev = self._hierarchical.copy()

        self._compute_sub(wide_rgb, narr_rgb, cmd)

        contribution = self._hierarchical - prev
        self._frame_series.append(contribution.copy())
        self._frame_cmds.append(cmd)
        self._frame_brake.append(self._last_is_brake)
        self._frame_wide_frac.append(self._last_wide_frac)
        self._n_frames += 1

        total = contribution.sum()
        return contribution / (total + 1e-12)

    def get_hierarchical(self, normalize: bool = True) -> np.ndarray:
        """
        Cumulative hierarchical attention over all processed frames.

        Parameters
        ----------
        normalize : If True (default), returns attention normalized to sum 1.

        Returns
        -------
        np.ndarray [num_classes]
        """
        h = self._hierarchical.copy()
        if normalize:
            h = h / (h.sum() + 1e-12)
        return h

    def get_series_df(self, normalize_rows: bool = True) -> pd.DataFrame:
        """
        Per-frame hierarchical attention as a pandas DataFrame.

        Columns : class names + 'cmd'
        Rows    : one per processed frame (in order)

        Parameters
        ----------
        normalize_rows : If True (default), each row sums to 1
                         (relative attention within each frame).
        """
        if not self._frame_series:
            return pd.DataFrame(columns=self.class_names + ["cmd"])

        arr = np.stack(self._frame_series, axis=0)   # [T, C]

        if normalize_rows:
            row_sums = arr.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-12, row_sums)
            arr = arr / row_sums

        df = pd.DataFrame(arr, columns=self.class_names)
        df["cmd"] = self._frame_cmds
        df["wide_frac"] = self._frame_wide_frac
        return df

    def get_mean_df(self) -> pd.DataFrame:
        """
        Mean normalized attention grouped by navigation command.

        Rows    : one per unique command seen
        Columns : class names

        Useful for comparing attention patterns across driving situations
        (e.g., following lane vs. turning).
        """
        df = self.get_series_df(normalize_rows=True)
        if df.empty:
            return df
        return df.groupby("cmd")[self.class_names].mean()

    def reset(self):
        self._hierarchical       = np.zeros(self.num_classes, dtype=np.float64)
        self._frame_series       = []
        self._frame_cmds         = []
        self._frame_brake:  List[bool]  = []
        self._frame_wide_frac: List[float] = []
        self._last_is_brake: bool  = False
        self._last_wide_frac: float = 1.0
        self._n_frames           = 0
        self._current_masks_wide = None
        self._current_masks_narr = None

    # ------------------------------------------------------------------
    # Analysis mode implementations
    # Mirrors _compute_node_level etc. from reference ATOMs / ATOMs_uitb
    # ------------------------------------------------------------------

    def _compute_node_level(
        self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor, cmd: int
    ) -> None:
        r_nodes = self._lrp1_nodes(wide_rgb, narr_rgb, cmd)
        # LRP1 (output→fc) correctly sets _last_is_brake; LRP2 (fc→input) always
        # returns is_brake=False and would overwrite it — save and restore here.
        is_brake_lrp1 = self._last_is_brake
        node_ids = _relevance_filter(r_nodes, self.p_relevance)
        if not node_ids:
            return

        lrp2_cache: dict = {} if conf.PLOT_COMPARATIVE_REL else None

        for node_id in node_ids:
            wide_r, narr_r = self._lrp2_pixels(wide_rgb, narr_rgb, node_id=node_id, cmd=cmd)
            if lrp2_cache is not None:
                lrp2_cache[node_id] = (wide_r, narr_r)
            R_sum  = self._give_element_selectivity(wide_r, narr_r)
            # abs(): AttnLRP (softmax/matmul rules) can produce negative F_c
            # relevances.  The paper formula assumes z+-only LRP (R_k ≥ 0).
            # Taking abs preserves the non-negativity of ATOMs attention
            # profiles, consistent with how _relevance_filter selects nodes.
            node_w = abs(r_nodes[node_id].item())
            self._hierarchical += np.asarray(R_sum, dtype=np.float64) * node_w
        self._last_is_brake = is_brake_lrp1

        if lrp2_cache:
            self._set_comparative_maps_node_level(
                wide_rgb, narr_rgb, cmd, node_ids, lrp2_cache
            )

    def _compute_layer_level(
        self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor, cmd: int
    ) -> None:
        wide_r, narr_r = self._saliency_map(
            wide_rgb, narr_rgb, beg="fc", end="input", cmd=cmd
        )
        self._hierarchical += np.asarray(
            self._give_element_selectivity(wide_r, narr_r), dtype=np.float64
        )

    def _compute_node_output_level(
        self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor, cmd: int
    ) -> None:
        wide_r, narr_r = self._saliency_map(
            wide_rgb, narr_rgb, beg="output", end="input", cmd=cmd
        )
        self._hierarchical += np.asarray(
            self._give_element_selectivity(wide_r, narr_r), dtype=np.float64
        )


    # ------------------------------------------------------------------
    # Saliency generation
    # Mirrors _generate_saliency_nodes / _generate_saliency_maps in reference
    # ------------------------------------------------------------------

    def _lrp1_nodes(
        self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor, cmd: int
    ) -> torch.Tensor:
        # LRP1 is output->fc: no narrow pixel map needed, wide only
        wide_r, _, _, is_brake = self.lrp.forward_relevance(
            wide_rgb, narr_rgb, beg="output", end="fc", cmd=cmd, spd=self._current_spd
        )
        self._last_is_brake  = is_brake
        self._last_wide_frac = 1.0   # fc mode has no narrow relevance
        # Return raw LRP1 relevances (paper formula uses R_k(x), not normalized)
        return wide_r

    def _lrp2_pixels(
        self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor, node_id: int, cmd: int
    ):
        # Always compute the default map — this is the one used for _hierarchical.
        wide_r, narr_r, wide_frac, is_brake = self.lrp.forward_relevance(
            wide_rgb, narr_rgb=narr_rgb,
            beg="fc", end="input", cmd=cmd, spd=self._current_spd,
            node_id=node_id
        )
        self._last_is_brake  = is_brake
        self._last_wide_frac = wide_frac if wide_frac is not None else 1.0
        norm_w = self._last_wide_frac
        self.saliency_data_wide_default = wide_r / norm_w
        self.saliency_data_narr_default = narr_r / (1 - norm_w) if narr_r is not None else None
        # Mirror into brake/drive for backwards-compatible access
        if is_brake:
            self.saliency_data_wide_brake = self.saliency_data_wide_default
            self.saliency_data_narr_brake = self.saliency_data_narr_default
        else:
            self.saliency_data_wide_drive = self.saliency_data_wide_default
            self.saliency_data_narr_drive = self.saliency_data_narr_default

        # Comparative maps for mode 1 are handled by _set_comparative_maps_node_level
        # in _compute_node_level after all nodes are processed — not here.

        return wide_r, narr_r

    def _set_comparative_maps_node_level(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: torch.Tensor,
        cmd: int,
        node_ids: list,
        lrp2_cache: dict,
    ) -> None:
        """
        Set saliency_data_wide_brake/drive for mode 1 visualization.

        Runs two cheap LRP1 (output→fc) passes with forced_brake/forced_drive
        seeds to get per-node weight vectors r_brake and r_drive. The
        already-computed LRP2 pixel maps are then re-weighted:

            saliency_wide_brake = Σ_k |r_brake[k]| * lrp2_map[k]
            saliency_wide_drive = Σ_k |r_drive[k]| * lrp2_map[k]

        Nodes brake-dominant under their respective LRP1 seed contribute more
        to the brake map; drive-dominant nodes contribute more to the drive map.
        The comparative map (drive - brake) is non-trivial when the two LRP1
        weight distributions differ — which they do for TFV6 where the seed
        controls which speed-bin output is emphasized.
        """
        r_brake, _, _, _ = self.lrp.forward_relevance(
            wide_rgb, narr_rgb=narr_rgb,
            beg="output", end="fc", cmd=cmd, spd=self._current_spd,
            forced_brake=True,
        )
        r_drive, _, _, _ = self.lrp.forward_relevance(
            wide_rgb, narr_rgb=narr_rgb,
            beg="output", end="fc", cmd=cmd, spd=self._current_spd,
            forced_drive=True,
        )

        brake_wide = None
        drive_wide = None
        brake_narr = None
        drive_narr = None

        for node_id in node_ids:
            wide_r, narr_r = lrp2_cache[node_id]
            wb = abs(float(r_brake[node_id].item())) if node_id < len(r_brake) else 0.0
            wd = abs(float(r_drive[node_id].item())) if node_id < len(r_drive) else 0.0
            if brake_wide is None:
                brake_wide = wide_r * wb
                drive_wide = wide_r * wd
                if narr_r is not None:
                    brake_narr = narr_r * wb
                    drive_narr = narr_r * wd
            else:
                brake_wide = brake_wide + wide_r * wb
                drive_wide = drive_wide + wide_r * wd
                if narr_r is not None:
                    brake_narr = brake_narr + narr_r * wb
                    drive_narr = drive_narr + narr_r * wd

        if brake_wide is None:
            return

        norm_b = brake_wide.abs().sum().item() + 1e-12
        norm_d = drive_wide.abs().sum().item() + 1e-12
        self.saliency_data_wide_brake = brake_wide / norm_b
        self.saliency_data_wide_drive = drive_wide / norm_d
        if brake_narr is not None:
            self.saliency_data_narr_brake = brake_narr / (brake_narr.abs().sum().item() + 1e-12)
            self.saliency_data_narr_drive = drive_narr / (drive_narr.abs().sum().item() + 1e-12)

    def _saliency_map(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: torch.Tensor,
        beg: str,
        end: str,
        cmd: int,
    ):
        # Always compute the default map — this is the one used for _hierarchical.
        wide_r, narr_r, wide_frac, is_brake = self.lrp.forward_relevance(
            wide_rgb, narr_rgb=narr_rgb,
            beg=beg, end=end, cmd=cmd, spd=self._current_spd
        )
        self._last_is_brake  = is_brake
        self._last_wide_frac = wide_frac if wide_frac is not None else 1.0
        norm_w = self._last_wide_frac
        self.saliency_data_wide_default = wide_r / norm_w
        self.saliency_data_narr_default = narr_r / (1 - norm_w) if narr_r is not None else None
        # Mirror into brake/drive for backwards-compatible access
        if is_brake:
            self.saliency_data_wide_brake = self.saliency_data_wide_default
            self.saliency_data_narr_brake = self.saliency_data_narr_default
        else:
            self.saliency_data_wide_drive = self.saliency_data_wide_default
            self.saliency_data_narr_drive = self.saliency_data_narr_default

        if conf.PLOT_COMPARATIVE_REL:
            # Compute forced maps for visualization only — does NOT feed _hierarchical.
            wide_r_b, narr_r_b, wide_frac_b, _ = self.lrp.forward_relevance(
                wide_rgb, narr_rgb=narr_rgb,
                beg=beg, end=end, cmd=cmd, spd=self._current_spd,
                forced_brake=True
            )
            wide_r_d, narr_r_d, wide_frac_d, _ = self.lrp.forward_relevance(
                wide_rgb, narr_rgb=narr_rgb,
                beg=beg, end=end, cmd=cmd, spd=self._current_spd,
                forced_drive=True
            )
            norm_b = wide_frac_b if wide_frac_b is not None else 1.0
            norm_d = wide_frac_d if wide_frac_d is not None else 1.0
            self.saliency_data_wide_brake = wide_r_b / norm_b
            self.saliency_data_narr_brake = narr_r_b / (1 - norm_b) if narr_r_b is not None else None
            self.saliency_data_wide_drive = wide_r_d / norm_d
            self.saliency_data_narr_drive = narr_r_d / (1 - norm_d) if narr_r_d is not None else None

        return wide_r, narr_r

    # ------------------------------------------------------------------
    # Selectivity: relevance map → per-class sums
    # Mirrors _give_element_selectivity_node from reference ATOMs
    # ------------------------------------------------------------------

    def _give_element_selectivity(
        self,
        wide_r: torch.Tensor,
        narr_r: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Project cross-normalized relevance onto semantic classes.
        Both maps already sum to (wide_fraction) and (narr_fraction)
        respectively, so we can accumulate them directly.
        """
        def _class_sums(r: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
            if r.dim() == 4:
                r = r.squeeze(0)
            r_hw = r.sum(dim=0)
            if masks.shape[-2:] != r_hw.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0).float(),
                    size=tuple(r_hw.shape),
                    mode="nearest",
                ).squeeze(0)
            raw  = (masks * r_hw.unsqueeze(0)).flatten(1).sum(dim=1)
            nz   = ((masks > 0) & (r_hw.unsqueeze(0) != 0)).float() \
                       .flatten(1).sum(dim=1).clamp(min=1.0)
            return raw / nz

        result = _class_sums(wide_r, self._current_masks_wide)

        if narr_r is not None and not conf.WIDE_ONLY_PROFILE:
            result = result + _class_sums(narr_r, self._current_masks_narr)

        return result