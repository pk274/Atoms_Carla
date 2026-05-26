"""
lrp_lbc.py
----------
Layer-wise Relevance Propagation for the LBC RGBPointModel.

Architecture overview:
  RGB [B,3,H,W]
    → normalize (ImageNet) → backbone (ResNet34) → [B,512,kh,kw]
    → concat with speed embd [B,640,kh,kw]
    → upconv decoder (3x ConvTranspose + Conv2d) → [B,C,H/4,W/4]
    → SpatialSoftmax → [B,C,2]  (waypoints)

Relevance is seeded at the pre-SpatialSoftmax feature map for the current
command and propagated back through the decoder and backbone to input pixels.
Speed embeddings are always detached so all relevance flows through the
visual backbone path.

Attribution modes (same interface as LRPCameraModel):
    beg='output', end='fc'    → backbone channel activations [512] as node weights
    beg='fc',     end='input' → LRP from a single backbone channel → pixels
    beg='output', end='input' → LRP from pre-softmax heatmap → pixels
"""

import torch
import torch.nn as nn
from zennit.rules import Pass, WSquare, AlphaBeta
from zennit.types import Convolution, Activation
from zennit.types import Linear as AnyLinear
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.torchvision import ResNetCanonizer
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Internal wrapper modules
# ---------------------------------------------------------------------------

class LBCToPreSoftmax(nn.Module):
    """
    LBC model up to (but not including) the final SpatialSoftmax.

    Inputs
    ------
    rgb         : [B, 3, H, W] float, requires_grad=True
    spd_spatial : [B, 128, kh, kw] detached speed embedding (spatially broadcast)

    Output
    ------
    [B, output_channel, H/4, W/4]  pre-softmax waypoint heatmap
    """

    def __init__(self, rgb_model: nn.Module):
        super().__init__()
        self.normalize    = rgb_model.normalize
        self.backbone     = rgb_model.backbone
        upconv_children   = list(rgb_model.upconv.children())
        self.upconv_no_ss = nn.Sequential(*upconv_children[:-1])  # drop SpatialSoftmax

    def forward(self, rgb: torch.Tensor, spd_spatial: torch.Tensor) -> torch.Tensor:
        feat     = self.backbone(self.normalize(rgb / 255.))      # [B, 512, kh, kw]
        combined = torch.cat([feat, spd_spatial], dim=1)          # [B, 640, kh, kw]
        return self.upconv_no_ss(combined)                        # [B, C, H/4, W/4]


class LBCBackbonePooled(nn.Module):
    """
    Backbone + global average pool → [B, 512].
    Used for fc→input LRP: each of the 512 channels is an attributable node.
    """

    def __init__(self, rgb_model: nn.Module):
        super().__init__()
        self.normalize = rgb_model.normalize
        self.backbone  = rgb_model.backbone
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten   = nn.Flatten(1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.flatten(
            self.pool(self.backbone(self.normalize(rgb / 255.)))
        )                                                          # [B, 512]


# ---------------------------------------------------------------------------
# Public LRP class
# ---------------------------------------------------------------------------

class LRPLBCModel:
    """
    LRP attribution for the LBC RGBPointModel.

    Provides the same public interface as LRPCameraModel so that ATOMsCarla
    works with either agent's LRP wrapper without modification.

    Parameters
    ----------
    model_eval : RGBPointModel already in .eval() mode
    num_cmds   : number of navigation commands (default 6)
    uitb       : use AlphaBeta(2,1) instead of (1,0)
    device     : torch device (defaults to CUDA if available)

    Node space
    ----------
    The LBC equivalent of WOR's 256-dim FC layer is the 512-dim globally-pooled
    backbone output.  Mode 'output→fc' returns relu(backbone_activations) as
    proxy node weights; mode 'fc→input' runs LRP from a single backbone channel
    back to pixels.

    Notes
    -----
    - narr_rgb is accepted by forward_relevance for API compatibility but ignored.
    - is_brake is always False (no brake concept in waypoint regression).
    - wide_frac is always 1.0 (single-camera agent).
    - forced_brake / forced_drive are accepted but ignored.
    - Speed embeddings are detached: all relevance flows through the backbone.
    """

    def __init__(
        self,
        model_eval,
        num_cmds:  int           = 6,
        uitb:      bool          = False,
        device:    torch.device  = None,
    ):
        self.uitb             = uitb
        self.alpha, self.beta = (2, 1) if uitb else (1, 0)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_cmds = num_cmds

        # Infer num_plan from output_channel of the Conv2d just before SpatialSoftmax
        upconv_children = list(model_eval.upconv.children())
        output_channel  = upconv_children[-2].out_channels   # Conv2d(64 → output_channel)
        self.num_plan   = output_channel // num_cmds

        self.kh = model_eval.kh
        self.kw = model_eval.kw

        self._model_eval = model_eval

        self.composite        = self._create_composite()
        self.presoftmax_model = LBCToPreSoftmax(model_eval).to(self.device).eval()
        self.backbone_model   = LBCBackbonePooled(model_eval).to(self.device).eval()

    # ------------------------------------------------------------------
    # Setup / API compat
    # ------------------------------------------------------------------

    def update_context(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: Optional[torch.Tensor] = None,
        spd:      float                  = None,
    ) -> None:
        """API compat with LRPCameraModel. Only checks that the model is in eval mode."""
        assert not self._model_eval.training, "rgb_model must be in eval() mode"

    def _create_composite(self) -> SpecialFirstLayerMapComposite:
        layer_map = [
            (Activation,  Pass()),
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
        return rgb.float().to(self.device).requires_grad_(True)

    def _compute_spd_spatial(self, spd: float) -> torch.Tensor:
        """Detached speed embedding broadcast to [1, 128, kh, kw]."""
        spd_t = torch.tensor([[spd]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embd = self._model_eval.spd_encoder(spd_t)              # [1, 128]
        return embd[..., None, None].expand(1, 128, self.kh, self.kw).detach()

    # ------------------------------------------------------------------
    # Public attribution entry-point
    # ------------------------------------------------------------------

    def forward_relevance(
        self,
        wide_rgb:     torch.Tensor,
        narr_rgb:     Optional[torch.Tensor] = None,
        cmd:          Optional[int]           = None,
        spd:          Optional[float]         = None,
        node_id:      Optional[int]           = None,
        raw:          bool                    = False,
        beg:          str                     = 'output',
        end:          str                     = 'input',
        forced_brake: bool                    = False,
        forced_drive: bool                    = False,
    ) -> Tuple[torch.Tensor, None, float, bool]:
        """
        LRP attribution for LBC.  narr_rgb, forced_brake, and forced_drive are
        accepted for interface compatibility with LRPCameraModel but are unused.

        Returns
        -------
        (wide_rel, None, 1.0, False)
            output→fc    : wide_rel is [512]  backbone node activations
            fc→input     : wide_rel is [1, 3, H, W]
            output→input : wide_rel is [1, 3, H, W]
        """
        if spd is None:
            spd = 0.0
        if cmd is None:
            cmd = 0

        rgb_x   = self._prepare_input(wide_rgb)
        spd_emb = self._compute_spd_spatial(spd)

        # ----------------------------------------------------------------
        # output → fc  (backbone activations as proxy node relevances)
        # ----------------------------------------------------------------
        if beg == 'output' and end == 'fc':
            fc_rel = self._attribute_to_backbone(rgb_x)
            return fc_rel, fc_rel, 1.0, False

        # ----------------------------------------------------------------
        # fc → input  (single backbone channel → pixels via LRP)
        # ----------------------------------------------------------------
        elif beg == 'fc' and end == 'input':
            if node_id is None:
                raise ValueError("fc→input mode requires node_id.")
            wide_rel = self._attribute_backbone(rgb_x, self._one_hot_node(node_id))
            return wide_rel, None, 1.0, False

        # ----------------------------------------------------------------
        # output → input  (pre-softmax seeded → pixels via LRP)
        # ----------------------------------------------------------------
        elif beg == 'output' and end == 'input':
            wide_rel = self._attribute_presoftmax_to_input(rgb_x, spd_emb, cmd)
            return wide_rel, None, 1.0, False

        else:
            raise ValueError(f"Unsupported mode: beg='{beg}', end='{end}'")

    # ------------------------------------------------------------------
    # Internal attribution helpers
    # ------------------------------------------------------------------

    def _attribute_to_backbone(self, rgb_x: torch.Tensor) -> torch.Tensor:
        """
        Return relu(globally-pooled backbone activations) as proxy node relevances.

        No backward pass is performed here — activations serve directly as weights,
        analogous to LRP1 in WOR but without propagating through the upconv decoder.
        The result is command-agnostic because the backbone is shared across commands.

        Returns a normalized [512] CPU tensor.
        """
        with torch.no_grad():
            feat   = self._model_eval.backbone(
                self._model_eval.normalize(rgb_x / 255.)
            )                                                    # [1, 512, kh, kw]
            pooled = feat.mean(dim=[2, 3]).squeeze(0)            # [512]
            act    = pooled.clamp(min=0)                         # keep positive activations only
        total = act.sum() + 1e-12
        return (act / total).cpu()                               # [512], sums to 1

    def _attribute_backbone(
        self,
        rgb_x:    torch.Tensor,
        selector,
    ) -> torch.Tensor:
        """LRP from a single backbone channel (globally pooled) → input pixels."""
        with torch.enable_grad():
            with self.composite.context(self.backbone_model):
                output   = self.backbone_model(rgb_x)            # [1, 512]
                grad_out = selector(output)
                (rgb_rel,) = torch.autograd.grad(
                    outputs      = output,
                    inputs       = [rgb_x],
                    grad_outputs = grad_out,
                )
        return rgb_rel.detach().cpu()                            # [1, 3, H, W]

    def _attribute_presoftmax_to_input(
        self,
        rgb_x:   torch.Tensor,
        spd_emb: torch.Tensor,
        cmd:     int,
    ) -> torch.Tensor:
        """
        LRP seeded at the pre-SpatialSoftmax heatmap (positive activations for the
        current command's waypoint channels), propagated through the decoder and
        backbone to input pixels.

        spd_emb is detached, so the gradient is computed only w.r.t. rgb_x.
        """
        with torch.enable_grad():
            with self.composite.context(self.presoftmax_model):
                output   = self.presoftmax_model(rgb_x, spd_emb)  # [1, C, H/4, W/4]
                grad_out = self._build_presoftmax_selector(output, cmd)
                (rgb_rel,) = torch.autograd.grad(
                    outputs      = output,
                    inputs       = [rgb_x],
                    grad_outputs = grad_out,
                )
        return rgb_rel.detach().cpu()                              # [1, 3, H, W]

    # ------------------------------------------------------------------
    # Selectors
    # ------------------------------------------------------------------

    def _build_presoftmax_selector(
        self,
        output: torch.Tensor,
        cmd:    int,
    ) -> torch.Tensor:
        """
        Relevance seed for output→input: positive activations in the current
        command's waypoint channels, zero for all other commands.
        Shape matches output [1, C, H/4, W/4].
        """
        seed = torch.zeros_like(output)
        lo   = cmd * self.num_plan
        hi   = (cmd + 1) * self.num_plan
        seed[:, lo:hi, :, :] = output[:, lo:hi, :, :].clamp(min=0)
        return seed

    def _one_hot_node(self, node: int):
        def selector(output: torch.Tensor) -> torch.Tensor:
            vec = torch.zeros_like(output)
            vec[:, node] = 1.0
            return vec
        return selector
