"""
lrp_camera_model.py
-------------------
Layer-wise Relevance Propagation for CameraModel (World on Rails agent).

Mirrors the LRPModel API from the reference Stable-Baselines3/Atari implementation.
Uses the zennit library for attribution.

Key design decisions:
  - ResNetCanonizer (from zennit.torchvision) handles BasicBlock skip connections
    and merges BatchNorm into the preceding Conv layers.
  - The inline `.mean(dim=[2,3])` from CameraModel.forward() is replaced by explicit
    AdaptiveAvgPool2d + Flatten so that zennit's hooks cover every operation.
  - Both cameras are attributed jointly in a single backward pass; relevance flows
    simultaneously through both the wide and narrow camera pathways.
  - The input tensor is explicitly given requires_grad=True in _prepare_input so
    that torch.autograd.grad (called internally by zennit) can compute gradients.
"""

import torch
import torch.nn as nn
from zennit.rules import Pass, WSquare, AlphaBeta
from zennit.types import Convolution, Activation
from zennit.types import Linear as AnyLinear
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.attribution import Gradient
from zennit.torchvision import ResNetCanonizer
import math
from typing import Tuple


class JointCameraForLRP(nn.Module):
    """Single-pass LRP wrapper: both cameras in one backward. Full output→input path."""

    def __init__(self, camera_model):
        super().__init__()
        self.normalize       = camera_model.normalize
        self.backbone_wide   = camera_model.backbone_wide
        self.pool_wide       = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone_narr   = camera_model.backbone_narr
        self.pool_narr       = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck_narr = camera_model.bottleneck_narr
        self.flatten         = nn.Flatten(1)
        self.act_head        = camera_model.act_head

    def forward(self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor) -> torch.Tensor:
        w = self.flatten(self.pool_wide(
            self.backbone_wide(self.normalize(wide_rgb / 255.0))))   # [B, 512]
        n = self.bottleneck_narr(self.flatten(self.pool_narr(
            self.backbone_narr(self.normalize(narr_rgb / 255.0)))))  # [B, 64]
        return self.act_head(torch.cat([w, n], dim=1))               # [B, 312]


class JointCameraToFC(nn.Module):
    """
    Same pipeline as JointCameraForLRP but stops at act_head[0:4],
    outputting the 256-dim FC activation instead of the final logits.
    Used for fc→input mode: relevance flows from a FC node back to both input images.
    """

    def __init__(self, camera_model):
        super().__init__()
        self.normalize        = camera_model.normalize
        self.backbone_wide    = camera_model.backbone_wide
        self.pool_wide        = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone_narr    = camera_model.backbone_narr
        self.pool_narr        = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck_narr  = camera_model.bottleneck_narr
        self.flatten          = nn.Flatten(1)
        self.act_head_partial = nn.Sequential(
            *list(camera_model.act_head.children())[:4]
        )

    def forward(self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor) -> torch.Tensor:
        w = self.flatten(self.pool_wide(
            self.backbone_wide(self.normalize(wide_rgb / 255.0))))   # [B, 512]
        n = self.bottleneck_narr(self.flatten(self.pool_narr(
            self.backbone_narr(self.normalize(narr_rgb / 255.0)))))  # [B, 64]
        return self.act_head_partial(torch.cat([w, n], dim=1))       # [B, 256]


# ---------------------------------------------------------------------------
# Public LRP class
# ---------------------------------------------------------------------------

class LRPCameraModel:
    """
    LRP attribution for the World on Rails CameraModel.

    Args:
        model_eval: Pretrained CameraModel already in .eval() mode.
        uitb:       Use AlphaBeta(2,1) instead of AlphaBeta(1,0). Default False.
        device:     Torch device. Defaults to CUDA if available.

    Attribution modes (kwargs to forward_relevance_joint / forward_relevance):
        beg='output', end='input'  ->  action logits → pixels       (full path)
        beg='output', end='fc'     ->  action logits → 256-dim FC activation
        beg='fc',     end='input'  ->  FC node → pixels             (backbone only)

    Example::

        lrp = LRPCameraModel(camera_model)
        lrp.update_context(wide_rgb, narr_rgb)
        wide_rel, narr_rel, wide_frac, is_brake = lrp.forward_relevance_joint(
            wide_rgb, narr_rgb, cmd=3, spd=4.0
        )
    """

    def __init__(self,
        model_eval,
        min_speeds: float = 0.0,
        max_speeds: float = 8.0,
        include_throttle: bool = False,
        uitb: bool = False,
        device: torch.device = None,
        undo_resnet_amplification: bool = True):

        self.uitb             = uitb
        self.alpha, self.beta = (2, 1) if uitb else (1, 0)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_cmds         = model_eval.num_cmds
        self.num_steers       = model_eval.num_steers
        self.num_throts       = model_eval.num_throts
        self.num_speeds       = 4
        self.all_speeds       = model_eval.all_speeds
        self.min_speeds       = min_speeds
        self.max_speeds       = max_speeds
        self.include_throttle = include_throttle

        self.undo_resnet_amplification = undo_resnet_amplification

        self._model_eval  = model_eval
        self._context_set = False

        self._act_head_ref = model_eval.act_head
        self.composite     = self._create_composite()

        # Build joint models once — they hold no per-frame state
        self.model_lrp    = JointCameraForLRP(model_eval).to(self.device).eval()
        self.fc_model_lrp = JointCameraToFC(model_eval).to(self.device).eval()
        self._act_head_partial_ref = nn.Sequential(
            *list(model_eval.act_head.children())[:4]
        ).to(self.device).eval()
        self._context_set = True   # always ready

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def update_context(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: torch.Tensor,
        spd: float = None,
    ):
        """
        Kept for API compatibility (atoms_carla.py calls this once per frame).
        Models are built at __init__ time and need no per-frame rebuild.
        Only asserts that the underlying CameraModel is still in eval mode.
        """
        assert not self._model_eval.training,              "camera_model must be in eval() mode"
        assert not self._model_eval.backbone_wide.training, "backbone_wide must be in eval() mode"
        assert not self._model_eval.backbone_narr.training, "backbone_narr must be in eval() mode"

    @staticmethod
    def _lerp_bins(spd: float, min_speeds: float, max_speeds: float, num_speeds: int):
        spd = max(min_speeds, min(max_speeds, spd))
        x   = (spd - min_speeds) / (max_speeds - min_speeds) * (num_speeds - 1)
        x0  = max(min(math.floor(x), num_speeds - 1), 0)
        x1  = max(min(math.ceil(x),  num_speeds - 1), 0)
        return x0, x1, float(x - x0)

    def _create_composite(self) -> SpecialFirstLayerMapComposite:
        layer_map = [
            (Activation,  Pass()),
            # AvgPool intentionally excluded: Pass() returns the pooled gradient
            # [B,C,1,1] unchanged, but the pre-pool tensor is [B,C,H,W] — size
            # mismatch. Standard autograd backward (no rule) correctly expands the
            # gradient to [B,C,H,W] with uniform spatial distribution.
            (Convolution, AlphaBeta(alpha=self.alpha, beta=self.beta)),
            (AnyLinear,   AlphaBeta(alpha=self.alpha, beta=self.beta, zero_params='bias')),
        ]
        first_map = [(Convolution, WSquare())]
        return SpecialFirstLayerMapComposite(
            layer_map=layer_map,
            first_map=first_map,
            canonizers=[ResNetCanonizer()],
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Cast to float, move to device, and enable gradients.
        requires_grad=True is mandatory: torch.autograd.grad requires the
        input to be part of the computation graph.
        """
        return rgb.float().to(self.device).requires_grad_(True)

    # ------------------------------------------------------------------
    # Public attribution entry-point
    # ------------------------------------------------------------------

    def forward_relevance(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: torch.Tensor,
        cmd: int = None,
        spd: float = None,
        node_id: int = None,
        raw: bool = False,
        beg: str = 'output',
        end: str = 'input',
        forced_brake: bool = False,
        forced_drive: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        """
        Joint single-pass LRP: both cameras attributed in one backward pass.

        Parameters
        ----------
        wide_rgb : [1, 3, H, W] in [0, 255]
        narr_rgb : [1, 3, H_n, W_n] in [0, 255]
        beg      : 'output' or 'fc'
        end      : 'input' or 'fc'
        cmd      : required for output→* modes
        spd      : required for output→* modes
        node_id  : required for fc→input mode
        raw      : skip cross-normalization if True

        Returns
        -------
        output→input : (wide_rel [1,3,H,W], narr_rel [1,3,H_n,W_n], wide_fraction, is_brake)
        output→fc    : (fc_rel [256], fc_rel [256], 1.0, is_brake)
        fc→input     : (wide_rel [1,3,H,W], narr_rel [1,3,H_n,W_n], wide_fraction, is_brake)
        """
        if not self._context_set:
            raise RuntimeError("Call update_context() before forward_relevance_joint().")

        wide_x   = self._prepare_input(wide_rgb)
        narr_x   = self._prepare_input(narr_rgb)
        is_brake = None

        # ----------------------------------------------------------------
        # output → input  (full path, joint backward through both ResNets)
        # ----------------------------------------------------------------
        if beg == 'output' and end == 'input':
            if cmd is None or spd is None:
                raise ValueError("output→input mode requires cmd and spd.")

            selector, is_brake = self._build_drive_brake_selector(
                wide_x, narr_x, cmd, spd, forced_brake=forced_brake, forced_drive=forced_drive
            )
            wide_rel, narr_rel = self._attribute_joint(
                self.model_lrp, wide_x, narr_x, selector
            )

            if self.undo_resnet_amplification:
                _, _, narr_frac_concat = self._attribute_to_concat(
                    wide_x, narr_x, selector
                )
                wide_frac_concat = 1.0 - narr_frac_concat
                wide_abs = float(wide_rel.abs().sum())
                narr_abs = float(narr_rel.abs().sum())
                total    = wide_abs + narr_abs + 1e-12
                wide_frac_pixel = wide_abs / total
                narr_frac_pixel = 1.0 - wide_frac_pixel
                if wide_frac_pixel > 1e-9:
                    wide_rel = wide_rel * (wide_frac_concat / wide_frac_pixel)
                if narr_frac_pixel > 1e-9:
                    narr_rel = narr_rel * (narr_frac_concat / narr_frac_pixel)

            wide_rel, narr_rel, wide_fraction = self._cross_normalize(
                wide_rel, narr_rel, raw=raw
            )
            return wide_rel, narr_rel, float(wide_fraction), is_brake

        # ----------------------------------------------------------------
        # output → fc  (action logits → 256-dim FC activation)
        # ----------------------------------------------------------------
        elif beg == 'output' and end == 'fc':
            if cmd is None or spd is None:
                raise ValueError("output→fc mode requires cmd and spd.")

            selector, is_brake = self._build_drive_brake_selector(
                wide_x, narr_x, cmd, spd, forced_brake=forced_brake, forced_drive=forced_drive
            )
            fc_rel = self._attribute_to_fc(wide_x, narr_x, selector)
            return fc_rel, fc_rel, 1.0, is_brake

        # ----------------------------------------------------------------
        # fc → input  (FC node → both input images, backbone only)
        # ----------------------------------------------------------------
        elif beg == 'fc' and end == 'input':
            selector = self._one_hot_node(node_id) if node_id is not None else None
            if cmd is not None and spd is not None:
                _, is_brake = self._build_drive_brake_selector(
                    wide_x, narr_x, cmd, spd, forced_brake=forced_brake, forced_drive=forced_drive
                )

            wide_rel, narr_rel = self._attribute_joint(
                self.fc_model_lrp, wide_x, narr_x, selector
            )

            if self.undo_resnet_amplification:
                _, _, narr_frac_concat = self._attribute_to_concat(
                    wide_x, narr_x, selector, head=self._act_head_partial_ref
                )
                wide_frac_concat = 1.0 - narr_frac_concat
                wide_abs = float(wide_rel.abs().sum())
                narr_abs = float(narr_rel.abs().sum())
                total    = wide_abs + narr_abs + 1e-12
                wide_frac_pixel = wide_abs / total
                narr_frac_pixel = 1.0 - wide_frac_pixel
                if wide_frac_pixel > 1e-9:
                    wide_rel = wide_rel * (wide_frac_concat / wide_frac_pixel)
                if narr_frac_pixel > 1e-9:
                    narr_rel = narr_rel * (narr_frac_concat / narr_frac_pixel)

            wide_rel, narr_rel, wide_fraction = self._cross_normalize(
                wide_rel, narr_rel, raw=raw
            )
            return wide_rel, narr_rel, float(wide_fraction), is_brake

        else:
            raise ValueError(f"Unsupported mode: beg='{beg}', end='{end}'")


    def _cross_normalize(self, wide_r, narr_r, raw=False):
        W_raw = wide_r.abs().sum().item()
        N_raw = narr_r.abs().sum().item()
        total = W_raw + N_raw + 1e-12
        wide_fraction = W_raw / total
        if raw:
            return wide_r, narr_r, wide_fraction
        wide_unit = wide_r.abs() / (W_raw + 1e-12)
        narr_unit = narr_r.abs() / (N_raw + 1e-12)
        wide_rel  = wide_unit * wide_fraction
        narr_rel  = narr_unit * (1.0 - wide_fraction)
        return wide_rel, narr_rel, wide_fraction

    # ------------------------------------------------------------------
    # Internal attribution helpers
    # ------------------------------------------------------------------

    def _attribute(self, model, x, selector=None):
        """LRP through a single-input model. Returns raw relevance tensor (not detached)."""
        with torch.enable_grad():
            with Gradient(model, self.composite) as attributor:
                if selector is not None:
                    _, relevance = attributor(x, selector)
                else:
                    _, relevance = attributor(x)
        return relevance

    def _attribute_joint(self, model, wide_x, narr_x, selector=None):
        """Joint backward through a two-input model via composite.context.
        Returns (wide_rel, narr_rel) detached on CPU."""
        with torch.enable_grad():
            with self.composite.context(model):
                output   = model(wide_x, narr_x)
                grad_out = (
                    selector(output) if selector is not None
                    else torch.ones_like(output)
                )
                wide_rel, narr_rel = torch.autograd.grad(
                    outputs      = output,
                    inputs       = [wide_x, narr_x],
                    grad_outputs = grad_out,
                )
        return wide_rel.detach().cpu(), narr_rel.detach().cpu()

    def _attribute_to_fc(self, wide_x, narr_x, selector=None):
        """LRP from action output → 256-dim FC activation (bypasses both ResNets).
        Runs a joint forward through joint_fc_model_lrp to get the correct FC
        activation, then propagates through the final linear only.
        Returns fc_rel [256] detached on CPU."""
        with torch.no_grad():
            fc_act = self.fc_model_lrp(wide_x, narr_x)   # [B, 256]

        final_linear = self._act_head_ref[4]
        fc_input     = fc_act.detach().requires_grad_(True)

        relevance = self._attribute(final_linear, fc_input, selector)
        return relevance.squeeze(0).detach().cpu()               # [256]

    

    def _attribute_to_concat(
        self,
        wide_x: torch.Tensor,
        narr_x: torch.Tensor,
        selector,
        head=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run LRP from the action output back to the 576-dim concatenation point.
        Returns (wide_rel_512, narr_rel_64, narr_frac_at_concat).

        Relevance is measured right at the junction where the 512-dim wide
        embedding and 64-dim narrow bottleneck are concatenated, bypassing
        both ResNets entirely.

        wide_x, narr_x: float tensors in [0, 255] on device (prepared inputs).
        """
        if head is None:
            head = self._act_head_ref
        with torch.no_grad():
            w = self.model_lrp.flatten(self.model_lrp.pool_wide(
                self.model_lrp.backbone_wide(
                    self.model_lrp.normalize(wide_x / 255.0))))      # [1, 512]
            n = self.model_lrp.bottleneck_narr(
                self.model_lrp.flatten(self.model_lrp.pool_narr(
                    self.model_lrp.backbone_narr(
                        self.model_lrp.normalize(narr_x / 255.0))))) # [1, 64]
            concat_in = torch.cat([w, n], dim=1)                     # [1, 576]

        concat_leaf = concat_in.detach().requires_grad_(True)
        rel = self._attribute(head, concat_leaf, selector).detach().cpu()
        wide_rel = rel[0, :512]
        narr_rel = rel[0, 512:]

        wide_abs  = float(wide_rel.abs().sum())
        narr_abs  = float(narr_rel.abs().sum())
        total     = wide_abs + narr_abs + 1e-12
        narr_frac = narr_abs / total

        return wide_rel, narr_rel, narr_frac

    # ------------------------------------------------------------------
    # Selectors
    # ------------------------------------------------------------------

    def _build_drive_brake_selector(
        self, wide_rgb: torch.Tensor, narr_rgb: torch.Tensor, cmd: int, spd: float, forced_brake: bool = False,
        forced_drive: bool = False
    ):
        """
        Decide brake vs drive from the agent's own output, then build a
        relevance initialization mask in the flat act_head output space.

        Brake mode  (brake_prob > 0.5):
            R = lerp-interpolated weight at the brake logit for (cmd, spd).
        Drive mode  (brake_prob <= 0.5):
            R = lerp-interpolated softmax(steer_logit) weights at the 9 steer
            positions for (cmd, spd). If include_throttle=True, relevance is
            split 0.5/0.5 between steer and throttle softmax weights.

        Returns
        -------
        selector  : callable  compatible with zennit's Gradient attributor
        is_brake  : bool      True if brake mode was chosen
        """
        base   = self.num_steers + self.num_throts + 1   # 13
        stride = self.num_speeds * base                   # 52

        x0, x1, w = self._lerp_bins(
            spd, self.min_speeds, self.max_speeds, self.num_speeds
        )

        with torch.no_grad():
            steer_logits, throt_logits, brake_logits = self._model_eval.policy(
                wide_rgb, narr_rgb, cmd
            )

            sl_x0 = steer_logits[x0]; sl_x1 = steer_logits[x1]
            tl_x0 = throt_logits[x0]; tl_x1 = throt_logits[x1]
            bl_x0 = brake_logits[x0]; bl_x1 = brake_logits[x1]

            steer_lerped = (1 - w) * sl_x0 + w * sl_x1
            throt_lerped = (1 - w) * tl_x0 + w * tl_x1
            brake_lerped = (1 - w) * bl_x0 + w * bl_x1

            steer_rep = steer_lerped.repeat(self.num_throts)
            throt_rep = throt_lerped.repeat_interleave(self.num_steers)
            combined  = torch.cat([steer_rep, throt_rep, brake_lerped.unsqueeze(0)])

            brake_prob = torch.softmax(combined, dim=0)[-1].item()
            is_brake   = brake_prob > 0.5

            mask = torch.zeros(1, self.num_cmds * self.num_speeds * base)

            if (is_brake or forced_brake) and not forced_drive:
                bi = self.num_steers + self.num_throts
                mask[0, cmd * stride + x0 * base + bi] += (1 - w)
                mask[0, cmd * stride + x1 * base + bi] += w
            else:
                sl_x0_soft = torch.softmax(sl_x0, dim=0)
                sl_x1_soft = torch.softmax(sl_x1, dim=0)
                steer_w    = 0.5 if self.include_throttle else 1.0

                for i in range(self.num_steers):
                    mask[0, cmd * stride + x0 * base + i] += (
                        steer_w * (1 - w) * sl_x0_soft[i].item()
                    )
                    mask[0, cmd * stride + x1 * base + i] += (
                        steer_w * w * sl_x1_soft[i].item()
                    )

                if self.include_throttle:
                    tl_x0_soft = torch.softmax(tl_x0, dim=0)
                    tl_x1_soft = torch.softmax(tl_x1, dim=0)
                    for j in range(self.num_throts):
                        mask[0, cmd * stride + x0 * base + self.num_steers + j] += (
                            0.5 * (1 - w) * tl_x0_soft[j].item()
                        )
                        mask[0, cmd * stride + x1 * base + self.num_steers + j] += (
                            0.5 * w * tl_x1_soft[j].item()
                        )

        mask = mask.to(self.device)

        def selector(output: torch.Tensor) -> torch.Tensor:
            return mask.expand_as(output)

        return selector, is_brake

    def _one_hot_node(self, node: int):
        def selector(output: torch.Tensor) -> torch.Tensor:
            vec = torch.zeros_like(output)
            vec[:, node] = 1.0
            return vec
        return selector

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def attribute_action(
        self,
        wide_rgb: torch.Tensor,
        cmd: int,
        action_type: str = "steer",
        bin_idx: int = 0,
    ) -> torch.Tensor:
        """
        High-level wrapper: attribute a specific command and action type.

        Args:
            wide_rgb:    Wide-camera image [B, 3, H, W] in [0, 255].
            cmd:         Command index (0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=FOLLOW, ...).
            action_type: One of 'steer', 'throttle', 'brake'.
            bin_idx:     Bin index within steer or throttle (ignored for 'brake').

        Returns:
            Relevance map [B, 3, H, W] on CPU.
        """
        if action_type == "steer":
            if not (0 <= bin_idx < self.num_steers):
                raise ValueError(f"bin_idx {bin_idx} out of range [0, {self.num_steers})")
            action_idx = bin_idx
        elif action_type == "throttle":
            if not (0 <= bin_idx < self.num_throts):
                raise ValueError(f"bin_idx {bin_idx} out of range [0, {self.num_throts})")
            action_idx = self.num_steers + bin_idx
        elif action_type == "brake":
            action_idx = self.num_steers + self.num_throts
        else:
            raise ValueError(
                f"Unknown action_type '{action_type}'. Use 'steer', 'throttle', or 'brake'."
            )

        return self.forward_relevance(
            wide_rgb,
            beg="output",
            end="input",
            cmd=cmd,
        )
