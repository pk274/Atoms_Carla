"""
lrp_transfuser.py
-----------------
Layer-wise Relevance Propagation for TransFuser v6 (visiononly_resnet34, LTF mode).

Architecture (relevant path):
  RGB [B, 3, 384, 2304]   (6 cameras concatenated)
    → NormalizeImageNet
    → timm ResNet34 (4 stages: 64 / 128 / 256 / 512 channels)
       at each stage: GPT cross-modal fusion block (n_layer=2, n_head=4)
       LiDAR = deterministic x/y grid (no gradient in LTF mode)
    → final image features  [B, 512, 12, 72]
    → AdaptiveAvgPool2d(1,1) → Flatten → [B, 512]   ← FC node space

Attribution modes (same interface as LRPCameraModel / LRPLBCModel):
    beg='output', end='fc'    → [512] backbone node activations
    beg='fc',     end='input' → [1, 3, H, W] LRP from a single backbone node
    beg='output', end='input' → [1, 3, H, W] LRP seeded at positive backbone features

LRP rules (zennit):
    Convolution (first): WSquare
    Convolution (rest):  AlphaBeta(α=1, β=0)
    Linear:              AlphaBeta(α=1, β=0)
    BatchNorm / LayerNorm / activations: Pass

SelfAttention replacement:
    GPT blocks use F.scaled_dot_product_attention (opaque fused kernel).
    Replaced with SelfAttentionExplicit which uses explicit matmul+softmax so
    that zennit can hook the constituent Linear layers.  The softmax and matmul
    themselves use standard autograd (conservative approximation of AttnLRP).

See design_decisions.md for rationale.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zennit.rules import Pass, WSquare, AlphaBeta
from zennit.types import Convolution, Activation
from zennit.types import Linear as AnyLinear
from zennit.composites import SpecialFirstLayerMapComposite

from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class NormalizeImageNet(nn.Module):
    """
    ImageNet normalisation as an nn.Module so zennit can intercept it.
    Applies Pass rule (the normalisation is a known linear transform).
    """
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.MEAN.to(x.device, dtype=x.dtype)
        std  = self.STD.to(x.device,  dtype=x.dtype)
        return (x / 255.0 - mean) / std


class SelfAttentionExplicit(nn.Module):
    """
    Reimplements SelfAttention without F.scaled_dot_product_attention.

    Uses explicit Q·K^T/sqrt(d) → softmax → A·V so that:
    - zennit can register AlphaBeta hooks on the four Linear layers
    - autograd can compute gradients through the softmax and matmul steps

    Constructed via from_module(), which shares the Linear layer objects
    from a deep-copied SelfAttention so weights are identical.
    """

    def __init__(
        self,
        key:       nn.Linear,
        query:     nn.Linear,
        value:     nn.Linear,
        proj:      nn.Linear,
        resid_drop: nn.Dropout,
        n_head:    int,
    ):
        super().__init__()
        self.key       = key
        self.query     = query
        self.value     = value
        self.proj      = proj
        self.resid_drop = resid_drop
        self.n_head    = n_head

    @classmethod
    def from_module(cls, attn) -> "SelfAttentionExplicit":
        """Build from an existing SelfAttention (deep-copied), sharing its Linears."""
        return cls(
            key=attn.key,
            query=attn.query,
            value=attn.value,
            proj=attn.proj,
            resid_drop=attn.resid_drop,
            n_head=attn.n_head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        hs = c // self.n_head

        k = self.key(x).view(b, t, self.n_head, hs).transpose(1, 2)    # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, hs).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, hs).transpose(1, 2)  # (b, nh, t, hs)

        scores  = torch.matmul(q, k.transpose(-2, -1)) * (hs ** -0.5)  # (b, nh, t, t)
        weights = torch.softmax(scores, dim=-1)
        y = torch.matmul(weights, v)                                    # (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------------
# Backbone wrapper for LRP
# ---------------------------------------------------------------------------

class TFv6ImageBackboneForLRP(nn.Module):
    """
    Full image backbone (normalize → ResNet34 + 4 GPT fusion blocks → avgpool)
    prepared for zennit LRP.

    - Deep-copies the GPT transformers so the inference model is unaffected.
    - Replaces SelfAttention with SelfAttentionExplicit in the copy.
    - References the image/lidar encoders and channel-projection convs from
      the original backbone (shared, read-only).
    - Generates the LTF LiDAR grid internally (no requires_grad).
    - Does NOT use channels_last memory format (would interfere with zennit).

    Output: [B, 512] globally averaged image features.
    """

    def __init__(self, backbone):
        super().__init__()

        self.config           = backbone.config
        self.image_encoder    = backbone.image_encoder
        self.lidar_encoder    = backbone.lidar_encoder
        self.lidar_channel_to_img = backbone.lidar_channel_to_img
        self.img_channel_to_lidar = backbone.img_channel_to_lidar
        self.avgpool_img      = backbone.avgpool_img
        self.avgpool_lidar    = backbone.avgpool_lidar
        self.avgpool_final    = nn.AdaptiveAvgPool2d((1, 1))
        self.normalize        = NormalizeImageNet()

        # Deep-copy transformers so in-place SelfAttention replacement is safe
        self.transformers = copy.deepcopy(backbone.transformers)
        for gpt in self.transformers:
            for block in gpt.blocks:
                block.attn = SelfAttentionExplicit.from_module(block.attn)

    def _make_lidar(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Deterministic LTF LiDAR grid [1, 2, H_lidar, W_lidar], no gradient."""
        h = self.config.lidar_height_pixel
        w = self.config.lidar_width_pixel
        xs = torch.linspace(0, 1, w)
        ys = torch.linspace(0, 1, h)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        lidar = torch.zeros(1, 2, h, w, device=device, dtype=dtype)
        lidar[0, 0] = y_grid
        lidar[0, 1] = x_grid
        return lidar  # no requires_grad → no gradient flows through lidar branch

    def _forward_block(self, layers, return_layers, features: torch.Tensor) -> torch.Tensor:
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _fuse(
        self,
        image_features: torch.Tensor,
        lidar_features: torch.Tensor,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_pool = self.avgpool_img(image_features)
        lid_pool = self.avgpool_lidar(lidar_features)
        lid_pool = self.lidar_channel_to_img[idx](lid_pool)

        img_out, lid_out = self.transformers[idx](img_pool, lid_pool)

        lid_out = self.img_channel_to_lidar[idx](lid_out)
        img_out = F.interpolate(img_out, size=image_features.shape[2:], mode="bilinear", align_corners=False)
        lid_out = F.interpolate(lid_out, size=lidar_features.shape[2:], mode="bilinear", align_corners=False)

        return image_features + img_out, lidar_features + lid_out

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W] float, requires_grad=True
        Returns:
            [B, 512] globally pooled image features
        """
        image_features = self.normalize(rgb)
        lidar_features = self._make_lidar(rgb.device, rgb.dtype).expand(rgb.shape[0], -1, -1, -1)

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # Skip encoder stem if it is a separate return layer
        if len(self.image_encoder.return_layers) > 4:
            image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        for i in range(4):
            image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self._fuse(image_features, lidar_features, i)

        return self.avgpool_final(image_features).flatten(1)  # [B, 512]


# ---------------------------------------------------------------------------
# Public LRP class
# ---------------------------------------------------------------------------

class LRPTFv6Model:
    """
    LRP attribution for TransFuser v6 (visiononly_resnet34, LTF mode).

    Provides the same public interface as LRPCameraModel and LRPLBCModel so
    that ATOMsCarla works unchanged with any agent.

    Node space
    ----------
    512-dim globally averaged image backbone output (Option A).
    See design_decisions.md for rationale.

    Parameters
    ----------
    backbone_eval : TransfuserBackbone already in .eval() mode
    uitb          : use AlphaBeta(2,1) instead of (1,0)
    device        : torch device (defaults to CUDA if available)

    API compatibility notes
    -----------------------
    - narr_rgb is accepted but treated as part of wide_rgb (TFV6 concatenates
      all cameras into one tensor; pass the full concatenated RGB as wide_rgb).
    - is_brake is always False.
    - wide_frac is always 1.0 (single concatenated camera).
    - forced_brake / forced_drive are accepted but ignored.
    """

    def __init__(
        self,
        backbone_eval,
        uitb:   bool         = False,
        device: torch.device = None,
    ):
        self.uitb             = uitb
        self.alpha, self.beta = (2, 1) if uitb else (1, 0)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        assert not backbone_eval.training, "backbone must be in eval() mode"

        self.backbone_model = (
            TFv6ImageBackboneForLRP(backbone_eval).to(self.device).eval()
        )
        self.composite = self._create_composite()

    # ------------------------------------------------------------------
    # Setup / API compatibility
    # ------------------------------------------------------------------

    def update_context(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: Optional[torch.Tensor] = None,
        spd:      float                  = None,
    ) -> None:
        """API compatibility — just verifies eval mode."""
        assert not self.backbone_model.training, "model must be in eval() mode"

    def _create_composite(self) -> SpecialFirstLayerMapComposite:
        layer_map = [
            (Activation,       Pass()),
            (nn.BatchNorm2d,   Pass()),
            (nn.LayerNorm,     Pass()),
            (Convolution,      AlphaBeta(alpha=self.alpha, beta=self.beta)),
            (AnyLinear,        AlphaBeta(alpha=self.alpha, beta=self.beta, zero_params="bias")),
        ]
        first_map = [(Convolution, WSquare())]
        return SpecialFirstLayerMapComposite(
            layer_map=layer_map,
            first_map=first_map,
            canonizers=[],
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb.float().to(self.device).requires_grad_(True)

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
        beg:          str                     = "output",
        end:          str                     = "input",
        forced_brake: bool                    = False,
        forced_drive: bool                    = False,
    ) -> Tuple[torch.Tensor, None, float, bool]:
        """
        LRP attribution for TFV6.  narr_rgb, cmd, spd, forced_brake, and
        forced_drive are accepted for interface compatibility but are unused
        (TFV6's backbone is command-agnostic; speed is not part of the backbone).

        Returns
        -------
        (wide_rel, None, 1.0, False)
            output→fc    : wide_rel is [512] backbone node activations
            fc→input     : wide_rel is [1, 3, H, W]
            output→input : wide_rel is [1, 3, H, W]
        """
        rgb_x = self._prepare_input(wide_rgb)

        # ----------------------------------------------------------------
        # output → fc  (backbone activations as proxy node relevances)
        # ----------------------------------------------------------------
        if beg == "output" and end == "fc":
            fc_rel = self._attribute_to_backbone(rgb_x)
            return fc_rel, fc_rel, 1.0, False

        # ----------------------------------------------------------------
        # fc → input  (single backbone channel → pixels via LRP)
        # ----------------------------------------------------------------
        elif beg == "fc" and end == "input":
            if node_id is None:
                raise ValueError("fc→input mode requires node_id.")
            wide_rel = self._attribute_backbone(rgb_x, self._one_hot_node(node_id))
            return wide_rel, None, 1.0, False

        # ----------------------------------------------------------------
        # output → input  (positive backbone features → pixels via LRP)
        # ----------------------------------------------------------------
        elif beg == "output" and end == "input":
            wide_rel = self._attribute_output_to_input(rgb_x)
            return wide_rel, None, 1.0, False

        else:
            raise ValueError(f"Unsupported mode: beg='{beg}', end='{end}'")

    # ------------------------------------------------------------------
    # Internal attribution helpers
    # ------------------------------------------------------------------

    def _attribute_to_backbone(self, rgb_x: torch.Tensor) -> torch.Tensor:
        """
        Return relu(globally-pooled backbone activations) normalised to sum 1.
        No backward pass — activations serve directly as proxy node weights.
        Returns a [512] CPU tensor.
        """
        with torch.no_grad():
            pooled = self.backbone_model(rgb_x).squeeze(0)  # [512]
            act    = pooled.clamp(min=0)
        total = act.sum() + 1e-12
        return (act / total).cpu()

    def _attribute_backbone(
        self,
        rgb_x:    torch.Tensor,
        selector,
    ) -> torch.Tensor:
        """LRP from a single backbone node (one-hot selector) → input pixels."""
        with torch.enable_grad():
            with self.composite.context(self.backbone_model):
                output      = self.backbone_model(rgb_x)   # [1, 512]
                grad_out    = selector(output)
                (rgb_rel,)  = torch.autograd.grad(
                    outputs      = output,
                    inputs       = [rgb_x],
                    grad_outputs = grad_out,
                )
        return rgb_rel.detach().cpu()  # [1, 3, H, W]

    def _attribute_output_to_input(self, rgb_x: torch.Tensor) -> torch.Tensor:
        """
        LRP seeded at positive backbone activations (uniform seed over active
        nodes), propagated to input pixels.
        """
        with torch.enable_grad():
            with self.composite.context(self.backbone_model):
                output = self.backbone_model(rgb_x)          # [1, 512]
                seed   = output.clamp(min=0)                 # positive activations only
                (rgb_rel,) = torch.autograd.grad(
                    outputs      = output,
                    inputs       = [rgb_x],
                    grad_outputs = seed,
                )
        return rgb_rel.detach().cpu()  # [1, 3, H, W]

    # ------------------------------------------------------------------
    # Selectors
    # ------------------------------------------------------------------

    def _one_hot_node(self, node: int):
        def selector(output: torch.Tensor) -> torch.Tensor:
            vec = torch.zeros_like(output)
            vec[:, node] = 1.0
            return vec
        return selector
