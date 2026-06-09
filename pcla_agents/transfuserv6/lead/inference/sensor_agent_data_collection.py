"""
sensor_agent_data_collection.py
--------------------------------
Subclass of SensorAgent that adds semantic segmentation cameras and records
driving frames using BaselineDataCollector (same on-disk format as WoR).

Activated by adding the "datacollect" variant to agents.json:
    "datacollect": {
        "agent": ".../sensor_agent_data_collection.py",
        "config": ".../visiononly_resnet34"
    }

Recording is enabled when atoms_config.BASELINE_RECORDING_MODE or
TESTSET_RECORDING_MODE is True.  Frames are saved at the end of each episode
(on destroy()) to:
    conf.BASELINE_DATA_DIR / "frames" / run_<timestamp>_<n>.npz   (baseline)
    conf.TEST_DATA_DIR     / "frames" / run_<timestamp>_<n>.npz   (test)

Each .npz contains stacked arrays [N, ...] compatible with BaselineDataLoader:
    wide_rgb      : [N, 3, H, W]  uint8
    seg_red_wide  : [N, H, W]     uint8  (red channel = CARLA semantic class ID)
    cmd           : [N]            int32
    speed         : [N]            float32
    is_brake      : [N]            int8
    frame_idx     : [N]            int32
(narr_rgb and seg_red_narr are omitted — TFV6 uses WIDE_ONLY_PROFILE)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

# Resolve ATOMs_Analysis imports regardless of where the script is run from
_pcla_root = Path(__file__).resolve().parents[5]
if str(_pcla_root) not in sys.path:
    sys.path.insert(0, str(_pcla_root))

# Add transfuserv6 dir so that `lead.*` imports resolve
_transfuserv6_dir = Path(__file__).resolve().parents[2]
if str(_transfuserv6_dir) not in sys.path:
    sys.path.insert(0, str(_transfuserv6_dir))

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.detection.baseline_dataset import BaselineDataCollector

from lead.common.constants import SEMANTIC_SEGMENTATION_CONVERTER
from lead.inference.sensor_agent import SensorAgent

# Lookup table: raw CARLA class ID (0-28) → grouped TFV6 class ID (0-9).
# Mirrors the transform applied by the LEAD expert pipeline (save_grouped_semantic=True)
# so that live-collected segmentation maps are compatible with LEAD-dataset baselines.
_SEG_CONVERTER = np.uint8(list(SEMANTIC_SEGMENTATION_CONVERTER.values()))

LOG = logging.getLogger(__name__)


def get_entry_point():  # dead: disable
    return "DataCollectionSensorAgent"


class DataCollectionSensorAgent(SensorAgent):
    """
    SensorAgent + semantic cameras + BaselineDataCollector.

    Override chain (Python MRO):
        DataCollectionSensorAgent.tick()     → SensorAgent.tick()
        DataCollectionSensorAgent.sensors()  → SensorAgent.sensors()
        DataCollectionSensorAgent.destroy()  → SensorAgent.destroy()
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _finish_setup(self):
        super()._finish_setup()

        collecting = conf.BASELINE_RECORDING_MODE or conf.TESTSET_RECORDING_MODE

        if collecting:
            self._data_collector = BaselineDataCollector()
            self._collecting     = True
            LOG.info(
                "[DataCollection] Recording enabled "
                f"(baseline={conf.BASELINE_RECORDING_MODE}, "
                f"test={conf.TESTSET_RECORDING_MODE})"
            )
        else:
            self._data_collector = None
            self._collecting     = False
            LOG.info("[DataCollection] Recording disabled (no recording mode set)")

    # ------------------------------------------------------------------
    # Sensor list — semantic cameras matching each RGB camera
    # ------------------------------------------------------------------

    def sensors(self) -> List[dict]:
        base = super().sensors()

        if not hasattr(self, "training_config"):
            return base

        config = self.training_config
        semantic_sensors = []
        for idx in range(1, config.num_cameras + 1):
            cam = config.camera_calibration[idx]
            semantic_sensors.append(
                {
                    "type":   "sensor.camera.semantic_segmentation",
                    "x":      cam["pos"][0],
                    "y":      cam["pos"][1],
                    "z":      cam["pos"][2],
                    "roll":   cam["rot"][0],
                    "pitch":  cam["rot"][1],
                    "yaw":    cam["rot"][2],
                    "width":  cam["width"],
                    "height": cam["height"],
                    "fov":    cam["fov"],
                    "id":     f"semantics_{idx}",
                }
            )

        return base + semantic_sensors

    # ------------------------------------------------------------------
    # Tick — extract semantic data and feed collector
    # ------------------------------------------------------------------

    def tick(self, input_data: dict, vehicle) -> dict:
        input_data = super().tick(input_data, vehicle)

        if not self._collecting or self._data_collector is None:
            return input_data

        if not hasattr(self, "training_config"):
            return input_data

        config = self.training_config
        seg_slices = []
        for idx in range(1, config.num_cameras + 1):
            key = f"semantics_{idx}"
            if key not in input_data:
                LOG.warning(f"[DataCollection] '{key}' missing — skipping frame")
                return input_data
            _, sem_bgra = input_data[key]          # (timestamp, [H, W, 4] BGRA)
            raw_ids = sem_bgra[:, :, 2].astype(np.uint8)  # red channel = raw CARLA class ID
            seg_slices.append(_SEG_CONVERTER[raw_ids])    # convert to grouped TFV6 class IDs

        seg_wide = np.concatenate(seg_slices, axis=1)  # [H, num_cameras*W]

        rgb   = input_data["rgb"]                      # [3, H, W] after tick preprocessing
        cmd   = int(np.argmax(input_data["command"]))  # one-hot → index
        speed = float(input_data.get("speed", 0.0))

        self._data_collector.add_frame(
            wide_rgb      = rgb,
            narr_rgb      = None,   # TFV6 is wide-only
            seg_red_wide  = seg_wide,
            seg_red_narrow= None,
            cmd           = cmd,
            speed         = speed,
        )

        return input_data

    # ------------------------------------------------------------------
    # Cleanup — flush buffer to disk
    # ------------------------------------------------------------------

    #def destroy(self, _=None):
    #    if self._collecting and self._data_collector is not None:
    #        saved = self._data_collector.save_run()
    #        if saved:
    #            LOG.info(f"[DataCollection] Run saved → {saved}")
#
    #    super().destroy(_)
