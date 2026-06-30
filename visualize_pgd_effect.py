"""
Visualize the effect of PGD adversarial perturbation on a single TFV6 frame.

Uses HPC settings: epsilon=14.0, n_steps=8, target="brake".
Picks one frame from test_data_alt/frames/, runs the attack, and saves
a side-by-side comparison (clean vs perturbed) plus a difference heatmap.
"""

import sys
import json
import numpy as np
from pathlib import Path as _Path

# The transfuserv6 `lead` package uses bare `from lead.xxx` imports, so its
# parent directory must be on sys.path before any pcla_agents.transfuserv6
# import is attempted.
_lead_parent = _Path(__file__).parent / "pcla_agents/transfuserv6"
if str(_lead_parent) not in sys.path:
    sys.path.insert(0, str(_lead_parent))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
MODEL_DIR  = ROOT / "pcla_agents/transfuserv6_pretrained/visiononly_resnet34"
FRAMES_DIR = ROOT / "data/TFV6/test_data_alt/frames"
OUT_PATH   = ROOT / "data/TFV6/results_alt/atoms_analysis/pgd_visual_example.png"

# HPC / atoms_config.py settings
PGD_EPSILON = 4.0
PGD_N_STEPS = 5
PGD_TARGET  = "brake"

# Speed bins matching TFV6 two-hot training target
SPEED_BINS = [0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0]

# Index of the forward-centre camera (0-based) in the 6-camera strip
FRONT_CAM_IDX = 1   # front-centre in the 6-cam 384×2304 layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(device):
    # Use bare `lead.*` imports so beartype sees a single module identity.
    # (pcla_agents.transfuserv6 is on sys.path, so `lead` resolves to its
    # lead/ subdirectory; using pcla_agents.transfuserv6.lead.* in parallel
    # creates a second identity and breaks beartype's isinstance check.)
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.tfv6 import TFv6

    with open(MODEL_DIR / "config.json") as f:
        training_config = TrainingConfig(json.load(f))

    model = TFv6(device, training_config)
    ckpt_files = sorted(MODEL_DIR.glob("model*.pth"))
    print(f"  Loading checkpoint: {ckpt_files[0].name}")
    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
    current_state = model.state_dict()
    drop_keys = [k for k, v in state_dict.items()
                 if k in current_state and current_state[k].shape != v.shape]
    for k in drop_keys:
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def make_minimal_data(spd: float, cmd: int, device: torch.device) -> dict:
    cmd_vec = torch.zeros(1, 6, dtype=torch.float32, device=device)
    cmd_vec[0, max(0, min(cmd, 5))] = 1.0
    return {
        "speed":                 torch.tensor([[spd]], dtype=torch.float32, device=device),
        "command":               cmd_vec,
        "target_point":          torch.zeros(1, 2, dtype=torch.float32, device=device),
        "target_point_previous": torch.zeros(1, 2, dtype=torch.float32, device=device),
        "target_point_next":     torch.zeros(1, 2, dtype=torch.float32, device=device),
        "acceleration":          torch.zeros(1, 1, dtype=torch.float32, device=device),
    }


def get_predictions(model, data: dict, device: torch.device):
    """Return (waypoints [N_wp, 2], speed_prob [8]) for the given data dict."""
    with torch.no_grad():
        result = model.forward(data)   # Prediction namedtuple
        logits = result.pred_target_speed_distribution   # [1, 8]
        wps    = result.pred_future_waypoints            # [1, N_wp, 2]
    prob = torch.softmax(logits[0], dim=0).cpu().numpy()
    wps  = wps[0].cpu().numpy()   # [N_wp, 2]  (x=lateral, y=longitudinal in agent frame)
    return wps, prob


def decode_speed(prob: np.ndarray) -> float:
    """Expected speed in m/s from a softmax distribution over 8 bins."""
    return float(np.dot(prob, SPEED_BINS))


def front_cam(rgb_hwc: np.ndarray) -> np.ndarray:
    """Crop the front-centre camera from the 6-cam concatenated strip."""
    H, W = rgb_hwc.shape[:2]
    cam_w = W // 6
    lo = FRONT_CAM_IDX * cam_w
    return rgb_hwc[:, lo:lo + cam_w]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load a frame
    # ------------------------------------------------------------------
    frame_files = sorted(FRAMES_DIR.glob("*.npz"))
    if not frame_files:
        sys.exit(f"No .npz files found in {FRAMES_DIR}")

    # Pick a frame that has reasonable speed (not stopped) for a more
    # interesting visual — scan up to 50 files.
    chosen_npz = None
    chosen_idx = 0
    for npz_path in frame_files[:50]:
        d = np.load(npz_path)
        speeds = d["speed"]
        # Prefer frames where the agent is moving (spd > 3 m/s)
        candidates = np.where(speeds > 3.0)[0]
        if len(candidates) >= 5:
            chosen_npz  = npz_path
            chosen_idx  = int(candidates[len(candidates) // 2])  # middle of the run
            break
    if chosen_npz is None:
        chosen_npz = frame_files[0]
        chosen_idx = 0

    d = np.load(chosen_npz)
    wide_np = d["wide_rgb"][chosen_idx]   # [3, H, W] uint8
    cmd     = int(d["cmd"][chosen_idx])
    spd     = float(d["speed"][chosen_idx])
    print(f"Frame: {chosen_npz.name}  idx={chosen_idx}  cmd={cmd}  spd={spd:.1f} m/s")

    # ------------------------------------------------------------------
    # 2. Build model and data dict
    # ------------------------------------------------------------------
    print("Loading TFV6 model...")
    model = load_model(device)

    wide_t = torch.from_numpy(wide_np).float().unsqueeze(0).to(device)  # [1, 3, H, W]
    data   = {**make_minimal_data(spd, cmd, device), "rgb": wide_t}

    # visiononly_resnet34 has radar_detection=True — supply zeros so forward()
    # doesn't crash; the random-weight radar branch has no effect on gradients
    # since the model was trained vision-only.
    n_pts = model.config.num_radar_sensors * model.config.num_radar_points_per_sensor
    data["radar"] = torch.zeros(1, n_pts, 5, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 3. Clean predictions
    # ------------------------------------------------------------------
    wps_clean, prob_clean = get_predictions(model, data, device)
    spd_clean  = decode_speed(prob_clean)
    print(f"Clean speed prediction: {spd_clean:.2f} m/s")
    print(f"Clean waypoints (x=lateral, y=forward): {wps_clean.tolist()}")

    # ------------------------------------------------------------------
    # 4. Run PGD attack
    # ------------------------------------------------------------------
    from ATOMs_Analysis.perturbation_manager import PerturbationManager
    pm = PerturbationManager(verbose=True)
    pm.attack_interval = 1   # always craft a fresh δ

    print(f"Running PGD: epsilon={PGD_EPSILON}, n_steps={PGD_N_STEPS}, target={PGD_TARGET!r}")
    adv_t = pm.pgd_attack_tfv6(
        nets    = [model],
        data    = data,
        target  = PGD_TARGET,
        epsilon = PGD_EPSILON,
        n_steps = PGD_N_STEPS,
    )  # [1, 3, H, W] float32

    # ------------------------------------------------------------------
    # 5. Adversarial predictions
    # ------------------------------------------------------------------
    data_adv = {**data, "rgb": adv_t.to(device)}
    wps_adv, prob_adv = get_predictions(model, data_adv, device)
    spd_adv  = decode_speed(prob_adv)
    print(f"Adversarial speed prediction: {spd_adv:.2f} m/s")
    print(f"Adversarial waypoints: {wps_adv.tolist()}")
    lateral_shift = wps_adv[:, 0].mean() - wps_clean[:, 0].mean()
    print(f"Mean lateral shift (x): {lateral_shift:+.3f} m  (positive = right)")

    # ------------------------------------------------------------------
    # 6. Prepare image patches (front-centre camera, HWC uint8)
    # ------------------------------------------------------------------
    wide_clean_hwc = wide_np.transpose(1, 2, 0)    # [H, W, 3]
    wide_adv_hwc   = adv_t[0].cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

    patch_clean = front_cam(wide_clean_hwc)
    patch_adv   = front_cam(wide_adv_hwc)

    diff = (patch_adv.astype(np.float32) - patch_clean.astype(np.float32))
    diff_amp = np.abs(diff).sum(axis=2)   # L1 over colour channels

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.04, right=0.97,
        top=0.88, bottom=0.12,
        wspace=0.06, hspace=0.30,
    )

    ax_clean = fig.add_subplot(gs[:, 0])
    ax_adv   = fig.add_subplot(gs[:, 1])
    ax_diff  = fig.add_subplot(gs[:, 2])
    ax_bot   = fig.add_axes([0.06, 0.03, 0.88, 0.10])   # bottom panel

    # — images —
    ax_clean.imshow(patch_clean)
    ax_clean.set_title("Clean input", fontsize=13, fontweight="bold")
    ax_clean.axis("off")

    ax_adv.imshow(patch_adv)
    ax_adv.set_title(f"After PGD  (ε={PGD_EPSILON:.0f}/255, {PGD_N_STEPS} steps)",
                     fontsize=13, fontweight="bold")
    ax_adv.axis("off")

    # — perturbation heatmap —
    im = ax_diff.imshow(diff_amp, cmap="hot", vmin=0)
    ax_diff.set_title("Perturbation magnitude  |Δ|₁", fontsize=13, fontweight="bold")
    ax_diff.axis("off")
    cb = fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
    cb.set_label("pixel units", fontsize=10)

    # — bottom panel: speed distribution for brake/max_speed, waypoints for steer targets —
    if PGD_TARGET in ("brake", "max_speed"):
        x     = np.arange(len(SPEED_BINS))
        width = 0.35
        ax_bot.bar(x - width/2, prob_clean, width, color="#2196F3", alpha=0.85,
                   label=f"Clean  ({spd_clean:.1f} m/s)")
        ax_bot.bar(x + width/2, prob_adv,   width, color="#F44336", alpha=0.85,
                   label=f"Adversarial  ({spd_adv:.1f} m/s)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f"{s:.1f}" for s in SPEED_BINS], fontsize=9)
        ax_bot.set_xlabel("Target-speed bin  [m/s]", fontsize=10)
        ax_bot.set_ylabel("Probability", fontsize=10)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.set_title(
            f"Speed distribution — clean vs adversarial  "
            f"(PGD target: {PGD_TARGET}, ε={PGD_EPSILON:.0f}, {PGD_N_STEPS} steps)",
            fontsize=11,
        )
    else:
        # steer_left / steer_right — show lateral waypoint shift
        n_wp = wps_clean.shape[0]
        wp_x = np.arange(n_wp)
        ax_bot.plot(wp_x, wps_clean[:, 0], "o-", color="#2196F3", linewidth=2, markersize=5,
                    label=f"Clean  (mean x = {wps_clean[:,0].mean():+.2f} m)")
        ax_bot.plot(wp_x, wps_adv[:, 0],   "o-", color="#F44336", linewidth=2, markersize=5,
                    label=f"Adversarial  (mean x = {wps_adv[:,0].mean():+.2f} m, "
                          f"Δ = {lateral_shift:+.2f} m)")
        ax_bot.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.4)
        ax_bot.set_xticks(wp_x)
        ax_bot.set_xticklabels([f"wp{i+1}" for i in range(n_wp)], fontsize=9)
        ax_bot.set_xlabel("Predicted waypoint", fontsize=10)
        ax_bot.set_ylabel("Lateral offset x  [m]\n(right = +)", fontsize=10)
        ax_bot.set_title(
            f"Predicted lateral trajectory — clean vs adversarial  "
            f"(PGD target: {PGD_TARGET}, ε={PGD_EPSILON:.0f}, {PGD_N_STEPS} steps)",
            fontsize=11,
        )
    ax_bot.legend(fontsize=10, loc="upper right")
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.suptitle(
        f"PGD adversarial attack on TFV6  |  "
        f"route {chosen_npz.stem}, frame {chosen_idx}  |  "
        f"ground-truth speed {spd:.1f} m/s",
        fontsize=14, fontweight="bold", y=0.97,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
