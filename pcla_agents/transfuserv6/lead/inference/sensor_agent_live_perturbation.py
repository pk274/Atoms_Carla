"""
sensor_agent_live_perturbation.py
----------------------------------
SensorAgent subclass that injects a perturbation at conf.INJECTION_TIME seconds
and records all frames (clean + perturbed) for offline ATOMs analysis via
run_online_analysis.py.

Activation
----------
Set  conf.LIVE_PERTURBATION_RECORDING_MODE = True  in atoms_config.py.
The perturbation type and intensity are read from conf.PERTURBATION,
conf.INTENSITY, and conf.CAM_INDEX.

The agent inherits semantic cameras from DataCollectionSensorAgent so ATOMs
can use segmentation masks.  Baseline/test collection (BASELINE_RECORDING_MODE,
TESTSET_RECORDING_MODE) is left to DataCollectionSensorAgent; both modes can
coexist independently.

Data layout written to disk:
    conf.TEST_DATA_DIR / "live_pert_frames" /
        run_<perturbation>_live_pert_<timestamp>_<n>.npz

Each npz contains:
    wide_rgb      : [N, 3, H, W]  uint8
    seg_red_wide  : [N, H, W]     uint8
    cmd           : [N]            int32
    speed         : [N]            float32
    is_brake      : [N]            int8
    frame_idx     : [N]            int32  (raw CARLA tick index, sampled every TEST_SAMPLE_INTERVAL)
    is_perturbed  : [N]            int8   (1 after INJECTION_TIME, 0 before)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

_pcla_root = Path(__file__).resolve().parents[5]
if str(_pcla_root) not in sys.path:
    sys.path.insert(0, str(_pcla_root))

# Add transfuserv6 dir so that `lead.*` imports resolve
_transfuserv6_dir = Path(__file__).resolve().parents[2]
if str(_transfuserv6_dir) not in sys.path:
    sys.path.insert(0, str(_transfuserv6_dir))

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.detection.dataset import TestDataCollector
from ATOMs_Analysis.perturbation_manager import PerturbationManager

from lead.common.constants import SEMANTIC_SEGMENTATION_CONVERTER
from lead.inference.sensor_agent_data_collection import (
    DataCollectionSensorAgent,
)

# Lookup table: raw CARLA class ID (0-28) → grouped TFV6 class ID (0-9).
_SEG_CONVERTER = np.uint8(list(SEMANTIC_SEGMENTATION_CONVERTER.values()))

# Only the 3 forward-facing cameras (indices 1-3) are saved.
# The LEAD baseline dataset was collected with 3 cameras (1152 px wide);
# using 6 cameras here would produce 2304 px images incompatible with it.
_N_FORWARD_CAMS = 3
_CAM_PX         = 384   # pixels per camera

LOG = logging.getLogger(__name__)


def get_entry_point():  # dead: disable
    return "LivePerturbationSensorAgent"


class LivePerturbationSensorAgent(DataCollectionSensorAgent):
    """
    DataCollectionSensorAgent + perturbation injection + live-pert recording.

    Perturbation is applied to input_data["rgb"] ([3,H,W] uint8) inside tick()
    once self._injection_active is True.  The flag is set in run_step() when
    timestamp >= conf.INJECTION_TIME.

    The perturbed (or clean, before injection) frame is fed to the model as
    normal — the parent run_step() uses whatever tick() returns.

    Both the clean frames (pre-injection) and perturbed frames (post-injection)
    are recorded to the live-pert collector so that run_online_analysis.py has
    a continuous time-series.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _finish_setup(self):
        super()._finish_setup()

        self._pm               = PerturbationManager(verbose=False)
        self._injection_active = False
        # Pending data for PGD frames: collected in tick(), recorded in _perturb_tensor_hook
        self._pending_seg_wide = None
        self._pending_cmd      = None
        self._pending_speed    = None

        if conf.LIVE_PERTURBATION_RECORDING_MODE:
            self._live_pert_collector = TestDataCollector(
                sample_every=conf.TEST_SAMPLE_INTERVAL,
                perturbation_name=conf.PERTURBATION,
            )
            LOG.info(
                "[LivePerturbation] Recording enabled — "
                f"perturbation='{conf.PERTURBATION}', "
                f"intensity={conf.INTENSITY}, "
                f"injection_time={conf.INJECTION_TIME}s"
            )
        else:
            self._live_pert_collector = None
            LOG.info("[LivePerturbation] LIVE_PERTURBATION_RECORDING_MODE is False — "
                     "no live-pert frames will be saved.")

    # ------------------------------------------------------------------
    # Adversarial perturbation hook — called from SensorAgent.run_step
    # after tensor prep, before the forward pass
    # ------------------------------------------------------------------

    def _perturb_tensor_hook(self, tensors: dict) -> dict:
        if not self._injection_active or conf.PERTURBATION != "pgd":
            return tensors

        with torch.enable_grad():
            tensors["rgb"] = self._pm.pgd_attack_tfv6(
                nets=self.closed_loop_inference.nets,
                data=tensors,
                target=conf.PGD_TARGET,
                epsilon=conf.EPSILON,
                n_steps=conf.PGD_N_STEPS,
            )

        # Record the PGD-perturbed frame now that we have the adversarial image.
        # tick() stored the seg/cmd/speed as pending; recording was deferred to here
        # so that the SAVED image contains the actual perturbation the model sees.
        if self._live_pert_collector is not None and self._pending_seg_wide is not None:
            perturbed_uint8 = (
                tensors["rgb"].squeeze(0).clamp(0, 255).byte().cpu().numpy()
            )  # [3, H, W] uint8
            self._live_pert_collector.add_frame(
                wide_rgb         = perturbed_uint8,
                narr_rgb         = None,
                seg_red_wide     = self._pending_seg_wide,
                seg_red_narr     = None,
                cmd              = self._pending_cmd,
                speed            = self._pending_speed,
                live_perturbation= True,
                is_perturbed     = True,
            )
            self._pending_seg_wide = None
            self._pending_cmd      = None
            self._pending_speed    = None

        return tensors

    # ------------------------------------------------------------------
    # run_step override — sets injection flag from timestamp
    # ------------------------------------------------------------------

    def run_step(self, input_data: dict, timestamp=None, vehicle=None):
        if timestamp is not None and timestamp >= conf.INJECTION_TIME:
            if not self._injection_active:
                msg = (
                    f"!!! PERTURBATION '{conf.PERTURBATION}' ACTIVATED "
                    f"at t={timestamp:.1f}s (intensity={conf.INTENSITY}) !!!"
                )
                LOG.warning(msg)
                print(msg, flush=True)
            self._injection_active = True
        return super().run_step(input_data, timestamp, vehicle)

    # ------------------------------------------------------------------
    # tick override — apply perturbation then record
    # ------------------------------------------------------------------

    def tick(self, input_data: dict, vehicle) -> dict:
        # Parent chain: DataCollectionSensorAgent.tick → SensorAgent.tick
        # This preprocesses input_data["rgb"] → [3, H, W] uint8 and optionally
        # records clean baseline/test frames if those modes are enabled.
        input_data = super().tick(input_data, vehicle)

        if "rgb" not in input_data:
            return input_data

        # ── Crop to forward cameras only ──────────────────────────────────
        # The LEAD baseline uses only 3 forward cameras (1152 px wide).
        # Rear cameras (indices 4-6) are discarded so live frames stay compatible.
        fwd_width = _N_FORWARD_CAMS * _CAM_PX  # 1152
        if input_data["rgb"].shape[-1] > fwd_width:
            input_data["rgb"] = input_data["rgb"][..., :fwd_width]

        # ── Inject non-PGD perturbation ───────────────────────────────────
        # PGD is applied to the float tensor later in _perturb_tensor_hook;
        # all other perturbations are applied here to the uint8 image.
        if self._injection_active and conf.PERTURBATION != "pgd":
            input_data["rgb"] = self._pm.perturb_tfv6_image(
                input_data["rgb"],
                perturbation=conf.PERTURBATION,
                intensity=conf.INTENSITY,
                camera_index=conf.CAM_INDEX,
                n_cameras=_N_FORWARD_CAMS,
            )

        # ── Build segmentation map (forward cameras only) ─────────────────
        if self._live_pert_collector is None:
            return input_data

        if not hasattr(self, "training_config"):
            return input_data

        seg_slices: List[np.ndarray] = []
        for idx in range(1, _N_FORWARD_CAMS + 1):   # cameras 1-3 only
            key = f"semantics_{idx}"
            if key not in input_data:
                LOG.warning(f"[LivePerturbation] '{key}' missing — frame skipped")
                return input_data
            _, sem_bgra = input_data[key]            # (ts, [H, W, 4] BGRA)
            raw_ids = sem_bgra[:, :, 2].astype(np.uint8)
            seg_slices.append(_SEG_CONVERTER[raw_ids])  # raw CARLA ID → grouped TFV6 ID

        seg_wide = np.concatenate(seg_slices, axis=1)   # [H, 3*W]
        cmd      = int(np.argmax(input_data["command"]))
        speed    = float(input_data.get("speed", 0.0))

        if self._injection_active and conf.PERTURBATION == "pgd":
            # PGD image is not ready yet — _perturb_tensor_hook will record it
            # after applying the attack.  Store pending data for that call.
            self._pending_seg_wide = seg_wide
            self._pending_cmd      = cmd
            self._pending_speed    = speed
        else:
            self._live_pert_collector.add_frame(
                wide_rgb         = input_data["rgb"],   # [3, H, W] uint8
                narr_rgb         = None,
                seg_red_wide     = seg_wide,
                seg_red_narr     = None,
                cmd              = cmd,
                speed            = speed,
                live_perturbation= True,
                is_perturbed     = self._injection_active,
            )
            self._pending_seg_wide = None
            self._pending_cmd      = None
            self._pending_speed    = None

        return input_data

    # ------------------------------------------------------------------
    # Cleanup — flush remaining buffer
    # ------------------------------------------------------------------

    def destroy(self, _=None):
        if self._live_pert_collector is not None:
            saved = self._live_pert_collector.save_run(live_perturbation=True)
            if saved:
                LOG.info(f"[LivePerturbation] Run saved → {saved}")
        super().destroy(_)
