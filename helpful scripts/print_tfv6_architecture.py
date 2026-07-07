"""
Prints the TFV6 model architecture layer by layer and saves it to a text file.

Usage:
    python print_tfv6_architecture.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pcla_agents" / "transfuserv6"))

import torch

from lead.training.config_training import TrainingConfig
from lead.tfv6.tfv6 import TFv6

TFV6_MODEL_DIR = Path("pcla_agents/transfuserv6_pretrained/visiononly_resnet34")
OUT_FILE = Path("tfv6_architecture.txt")


def main():
    with open(TFV6_MODEL_DIR / "config.json") as f:
        training_config = TrainingConfig(json.load(f))

    device = torch.device("cpu")
    model = TFv6(device, training_config)

    ckpt_files = sorted(TFV6_MODEL_DIR.glob("model*.pth"))
    if ckpt_files:
        print(f"Loading checkpoint: {ckpt_files[0]}")
        state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
        current_state = model.state_dict()
        drop_keys = [k for k, v in state_dict.items()
                     if k in current_state and current_state[k].shape != v.shape]
        for k in drop_keys:
            state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = []
    lines.append(repr(model))
    lines.append("")
    lines.append(f"Total parameters:     {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("Named modules (module_path : class_name)")
    lines.append("=" * 80)
    for name, module in model.named_modules():
        if name == "":
            continue
        lines.append(f"{name} : {module.__class__.__name__}")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved architecture dump to {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
