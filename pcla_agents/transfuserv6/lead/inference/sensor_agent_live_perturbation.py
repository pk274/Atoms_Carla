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

from lead.inference.sensor_agent_data_collection import (
    DataCollectionSensorAgent,
)

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

        # ── Inject perturbation ────────────────────────────────────────
        # "pgd" is handled as a tensor-level attack in _perturb_tensor_hook;
        # all other names go through the numpy registry here.
        if self._injection_active and conf.PERTURBATION != "pgd":
            n_cams = (
                self.training_config.num_cameras
                if hasattr(self, "training_config")
                else 6
            )
            input_data["rgb"] = self._pm.perturb_tfv6_image(
                input_data["rgb"],
                perturbation=conf.PERTURBATION,
                intensity=conf.INTENSITY,
                camera_index=conf.CAM_INDEX,
                n_cameras=n_cams,
            )

        # ── Record for offline analysis ────────────────────────────────
        if self._live_pert_collector is None:
            return input_data

        if not hasattr(self, "training_config"):
            return input_data

        config = self.training_config
        seg_slices: List[np.ndarray] = []
        for idx in range(1, config.num_cameras + 1):
            key = f"semantics_{idx}"
            if key not in input_data:
                LOG.warning(f"[LivePerturbation] '{key}' missing — frame skipped")
                return input_data
            _, sem_bgra = input_data[key]            # (ts, [H, W, 4] BGRA)
            seg_slices.append(sem_bgra[:, :, 2].astype(np.uint8))

        seg_wide = np.concatenate(seg_slices, axis=1)            # [H, num_cams*W]
        cmd      = int(np.argmax(input_data["command"]))
        speed    = float(input_data.get("speed", 0.0))

        self._live_pert_collector.add_frame(
            wide_rgb         = input_data["rgb"],   # [3, H, W] — perturbed or clean
            narr_rgb         = None,                # TFV6 wide-only
            seg_red_wide     = seg_wide,
            seg_red_narr     = None,
            cmd              = cmd,
            speed            = speed,
            live_perturbation= True,
            is_perturbed     = self._injection_active,
        )

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
