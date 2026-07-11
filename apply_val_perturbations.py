"""Apply perturbations to the local validation set.

Reads clean frames from conf.VAL_DATA_DIR/frames/ and writes
val_data/val_labeled.npz with the same 5-way 20% spec used for the test set.
PGD frames are labelled perturbed but contain clean pixels — adversarial
profiles come from the HPC job.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ATOMs_Analysis.atoms_config import ExperimentConfig
conf = ExperimentConfig()

from ATOMs_Analysis.perturbation_manager import PerturbationManager
from ATOMs_Analysis.detection.dataset import (
    PerturbationApplier, PerturbationSpec, PerturbationEntry,
)

pm = PerturbationManager()

spec = PerturbationSpec([
    PerturbationEntry(fraction=0.20, perturbation=None),
    PerturbationEntry(fraction=0.20, perturbation="gaussian_noise",   intensity=conf.NOISE_INTENSITY),
    PerturbationEntry(fraction=0.20, perturbation="brightness_scale", intensity=conf.BRIGHTNESS_INTENSITY),
    PerturbationEntry(fraction=0.20, perturbation="camera_loss",      intensity=1),
    PerturbationEntry(fraction=0.20, perturbation="pgd",              intensity=conf.PGD_EPSILON, fgsm_target=conf.PGD_TARGET),
])

applier = PerturbationApplier(pm, model=None, data_dir=conf.VAL_DATA_DIR)
out = applier.apply(spec, seed=42, output_name="val_labeled")
print(f"Done → {out}")
