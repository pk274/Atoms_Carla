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
    → final lidar features  [B, num_lidar_ch, H_bev, W_bev]
    → PlanningContextEncoder  (BEV tokens + speed/cmd/tp status tokens)
    → TransformerDecoder (6 layers, 256-dim, 8 heads)
    → speed_query token [B, 256]   ← F_c node space (Option B)
    → target_speed_decoder (Linear 256→256 → ReLU → Linear 256→8)
    → speed logits [B, 8]          ← LRP1 seed

Attribution modes (same interface as LRPCameraModel / LRPLBCModel):
    beg='output', end='fc'    → [256] speed-query relevances (LRP1)
    beg='fc',     end='input' → [1, 3, H, W] pixel map from a single node (LRP2)
    beg='output', end='input' → [1, 3, H, W] pixel map seeded at positive speed-query

LRP rules (zennit + custom AttnLRP):
    Convolution (first): WSquare
    Convolution (rest):  AlphaBeta(α=1, β=0)
    AttentionLinear:     Epsilon(ε=1e-2)    ← K/Q/V/proj in all attention blocks
    Linear (FFN):        AlphaBeta(α=1, β=0)
    BatchNorm / LayerNorm / activations: Pass

GPT blocks (backbone cross-modal fusion):
    SelfAttentionExplicit replaces the original SelfAttention that uses
    F.scaled_dot_product_attention (opaque fused kernel). Softmax and
    matmul use AttnLRP custom autograd (LRPSoftmax, LRPMatMul).

Planning decoder:
    TransformerDecoderLayerExplicit replaces nn.TransformerDecoderLayer.
    MultiheadAttentionExplicit replaces nn.MultiheadAttention.
    Both use LRPSoftmax and LRPMatMul for AttnLRP-compliant backward.

Residual/skip connections (ResNet34 backbone, GPT fusion, PlanningDecoder):
    Every `+` in this model (BasicBlock skip connections, GPT block residual
    streams, backbone cross-modal fusion, decoder self/cross-attn + FFN
    residuals) is replaced with LRPResidualAdd, which implements Ratio-Based
    Relevance Splitting (Otsuki et al. 2024, arXiv:2407.09115) instead of
    raw autograd. Without this, `d(a+b)/da = d(a+b)/db = 1` sends the FULL
    upstream relevance to BOTH branches at every residual junction — with
    ~50+ such junctions in this model, relevance is duplicated rather than
    conserved throughout. See design_decisions.md for the analysis and why
    an absolute-value ratio (not zennit's signed `Norm` rule) is used.

See design_decisions.md for rationale.
"""

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.resnet import BasicBlock as TimmBasicBlock, Bottleneck as TimmBottleneck

from zennit.rules import Pass, WSquare, AlphaBeta, Epsilon
from zennit.types import Convolution, Activation
from zennit.types import Linear as AnyLinear
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.canonizers import SequentialMergeBatchNorm

from typing import Dict, Optional, Tuple

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf


# ---------------------------------------------------------------------------
# AttnLRP custom autograd functions
# ---------------------------------------------------------------------------

class LRPSoftmax(torch.autograd.Function):
    """
    AttnLRP softmax rule (Proposition 3.1, Eq. 13).
    R^{l-1}_i = x_i * (R^l_i - s_i * sum_j R^l_j)
    where s = softmax(x).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        s = torch.softmax(x, dim=-1)
        ctx.save_for_backward(x, s)
        return s

    @staticmethod
    def backward(ctx, R: torch.Tensor) -> torch.Tensor:
        x, s = ctx.saved_tensors
        R_sum = R.sum(dim=-1, keepdim=True)
        return x * (R - s * R_sum)


class LRPMatMul(torch.autograd.Function):
    """
    AttnLRP bi-linear matmul rule (Proposition 3.3, Eq. 15).
    For O = A @ B:
        R_A = (R / denom) @ B^T * A
        R_B = A^T @ (R / denom) * B
    where denom = 2*O + eps*sign(O).
    Applied to both the Q·K^T and A·V products inside attention.
    """
    EPS = 1e-6

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        O = torch.matmul(A, B)
        ctx.save_for_backward(A, B, O)
        return O

    @staticmethod
    def backward(ctx, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A, B, O = ctx.saved_tensors
        sign = O.sign()
        sign[sign == 0] = 1.0
        denom = 2.0 * O + LRPMatMul.EPS * sign
        scaled_R = R / denom
        R_A = torch.matmul(scaled_R, B.transpose(-2, -1)) * A
        R_B = torch.matmul(A.transpose(-2, -1), scaled_R) * B
        return R_A, R_B


# ---------------------------------------------------------------------------
# Residual/skip-connection relevance splitting (Otsuki et al. 2024)
# ---------------------------------------------------------------------------

class LRPResidualAdd(torch.autograd.Function):
    """
    Ratio-Based Relevance Splitting for residual/skip connections.

    Otsuki et al. 2024, "Layer-Wise Relevance Propagation with Conservation
    Property for ResNet" (arXiv:2407.09115), Eq. 5.  Plain autograd through
    z = a + b sends the FULL upstream relevance to BOTH a and b (since
    dz/da = dz/db = 1), silently duplicating relevance at every residual
    junction instead of splitting it.  This model has ~50+ such junctions
    (ResNet34 image/lidar encoders, GPT cross-modal fusion, backbone fusion,
    PlanningDecoder self/cross-attn + FFN residuals) — left unguarded, none
    of them conserve relevance, and the duplication compounds across every
    one of them.

    This Function splits relevance in proportion to each branch's absolute
    forward contribution:
        R_a = R_z * |a| / (|a| + |b| + eps)
        R_b = R_z * |b| / (|a| + |b| + eps)
    so that R_a + R_b == R_z (exact conservation, up to the eps stabilizer).

    Absolute value (not signed, unlike zennit's built-in `Sum` + `Norm` rule
    pairing, which zennit's own ResNetCanonizer uses for torchvision ResNets)
    avoids denominator collapse when a and b are similar magnitude but
    opposite sign (a+b ~ 0 while |a|+|b| >> 0) — the same failure mode
    LRPMatMul.EPS guards against above, and the same class of instability
    documented in docs/lrp_todo.md "Bug C" (near-zero ε-rule denominators
    causing ~1e14x amplification with sign oscillation).
    """
    EPS = 1e-6

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        a_abs, b_abs = a.abs(), b.abs()
        denom = a_abs + b_abs + LRPResidualAdd.EPS
        R_a = R * a_abs / denom
        R_b = R * b_abs / denom
        return R_a, R_b


# ---------------------------------------------------------------------------
# AttentionLinear marker — receives ε-rule instead of AlphaBeta
# ---------------------------------------------------------------------------

class AttentionLinear(nn.Linear):
    """
    nn.Linear subclass used for K/Q/V/proj projections inside all attention
    blocks. Registered separately in the composite layer_map to receive the
    ε-rule (AttnLRP recommendation) instead of AlphaBeta used for Conv/FFN.
    """
    pass


def _make_attn_linear(src: nn.Linear) -> AttentionLinear:
    """Share weights from an existing nn.Linear into a new AttentionLinear."""
    a = AttentionLinear(src.in_features, src.out_features, bias=src.bias is not None,
                        device=src.weight.device, dtype=src.weight.dtype)
    a.weight = src.weight
    a.bias   = src.bias
    return a


# ---------------------------------------------------------------------------
# SelfAttentionExplicit (for GPT backbone cross-modal fusion blocks)
# ---------------------------------------------------------------------------

class SelfAttentionExplicit(nn.Module):
    """
    Reimplements SelfAttention without F.scaled_dot_product_attention.

    - K/Q/V/proj are AttentionLinear so they receive the ε-rule.
    - Q·K^T and A·V use LRPMatMul (AttnLRP Prop 3.3).
    - Softmax uses LRPSoftmax (AttnLRP Prop 3.1).
    """

    def __init__(
        self,
        key:        AttentionLinear,
        query:      AttentionLinear,
        value:      AttentionLinear,
        proj:       AttentionLinear,
        resid_drop: nn.Dropout,
        n_head:     int,
    ):
        super().__init__()
        self.key        = key
        self.query      = query
        self.value      = value
        self.proj       = proj
        self.resid_drop = resid_drop
        self.n_head     = n_head

    @classmethod
    def from_module(cls, attn) -> "SelfAttentionExplicit":
        """Build from a deep-copied SelfAttention, wrapping its Linears."""
        return cls(
            key=_make_attn_linear(attn.key),
            query=_make_attn_linear(attn.query),
            value=_make_attn_linear(attn.value),
            proj=_make_attn_linear(attn.proj),
            resid_drop=attn.resid_drop,
            n_head=attn.n_head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        hs = c // self.n_head

        k = self.key(x).view(b, t, self.n_head, hs).transpose(1, 2)    # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, hs).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, hs).transpose(1, 2)  # (b, nh, t, hs)

        scores  = LRPMatMul.apply(q, k.transpose(-2, -1)) * (hs ** -0.5)
        weights = LRPSoftmax.apply(scores)
        y       = LRPMatMul.apply(weights, v)                           # (b, nh, t, hs)
        y       = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------------
# MultiheadAttentionExplicit (for PlanningDecoder TransformerDecoderLayers)
# ---------------------------------------------------------------------------

class MultiheadAttentionExplicit(nn.Module):
    """
    Explicit MHA replacing nn.MultiheadAttention.

    - Extracts Q/K/V/out as separate AttentionLinear layers.
    - Uses LRPSoftmax and LRPMatMul for AttnLRP-compliant backward.
    - Returns (output, None) to match nn.MultiheadAttention signature.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.batch_first = True   # nn.TransformerDecoder reads this attribute
        assert self.head_dim * num_heads == embed_dim

        self.q_proj   = AttentionLinear(embed_dim, embed_dim)
        self.k_proj   = AttentionLinear(embed_dim, embed_dim)
        self.v_proj   = AttentionLinear(embed_dim, embed_dim)
        self.out_proj = AttentionLinear(embed_dim, embed_dim)

    @classmethod
    def from_module(cls, mha: nn.MultiheadAttention) -> "MultiheadAttentionExplicit":
        E = mha.embed_dim
        H = mha.num_heads
        m = cls(E, H)

        w = mha.in_proj_weight   # [3E, E]
        b = mha.in_proj_bias     # [3E] or None

        m.q_proj.weight = nn.Parameter(w[:E].clone())
        m.k_proj.weight = nn.Parameter(w[E:2*E].clone())
        m.v_proj.weight = nn.Parameter(w[2*E:].clone())

        if b is not None:
            m.q_proj.bias = nn.Parameter(b[:E].clone())
            m.k_proj.bias = nn.Parameter(b[E:2*E].clone())
            m.v_proj.bias = nn.Parameter(b[2*E:].clone())

        m.out_proj.weight = nn.Parameter(mha.out_proj.weight.clone())
        m.out_proj.bias   = nn.Parameter(mha.out_proj.bias.clone())

        return m

    def forward(
        self,
        query:              torch.Tensor,
        key:                torch.Tensor,
        value:              torch.Tensor,
        key_padding_mask:   Optional[torch.Tensor] = None,
        attn_mask:          Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        B, T_q, _ = query.shape
        T_kv       = key.shape[1]
        H, D       = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, T_q, H, D).transpose(1, 2)    # (B, H, T_q, D)
        k = self.k_proj(key).view(B, T_kv, H, D).transpose(1, 2)     # (B, H, T_kv, D)
        v = self.v_proj(value).view(B, T_kv, H, D).transpose(1, 2)   # (B, H, T_kv, D)

        scores = LRPMatMul.apply(q, k.transpose(-2, -1)) * (D ** -0.5)

        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights = LRPSoftmax.apply(scores)                             # (B, H, T_q, T_kv)
        y       = LRPMatMul.apply(weights, v)                          # (B, H, T_q, D)
        y       = y.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(y), None


# ---------------------------------------------------------------------------
# TransformerDecoderLayerExplicit
# ---------------------------------------------------------------------------

class TransformerDecoderLayerExplicit(nn.Module):
    """
    Replaces nn.TransformerDecoderLayer.

    Uses MultiheadAttentionExplicit for self-attn and cross-attn so that
    AttnLRP rules are active throughout the planning decoder.

    Post-norm layout matches PyTorch's default TransformerDecoderLayer.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.self_attn   = MultiheadAttentionExplicit(d_model, nhead)
        self.cross_attn  = MultiheadAttentionExplicit(d_model, nhead)
        self.linear1     = nn.Linear(d_model, dim_feedforward)
        self.linear2     = nn.Linear(dim_feedforward, d_model)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        # Hardcoded GELU matches PlanningDecoder's TransformerDecoderLayer
        # (activation=nn.GELU() in training config).  from_module does not
        # copy the activation; if the config changes, update this too.
        self.activation  = nn.GELU()

    @classmethod
    def from_module(cls, layer: nn.TransformerDecoderLayer) -> "TransformerDecoderLayerExplicit":
        d  = layer.self_attn.embed_dim
        h  = layer.self_attn.num_heads
        ff = layer.linear1.out_features
        m  = cls(d, h, ff)

        m.self_attn  = MultiheadAttentionExplicit.from_module(layer.self_attn)
        m.cross_attn = MultiheadAttentionExplicit.from_module(layer.multihead_attn)

        for attr in ("linear1", "linear2", "norm1", "norm2", "norm3"):
            src = getattr(layer, attr)
            dst = getattr(m, attr)
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        return m

    def forward(
        self,
        tgt:                    torch.Tensor,
        memory:                 torch.Tensor,
        tgt_mask:               Optional[torch.Tensor] = None,
        memory_mask:            Optional[torch.Tensor] = None,
        tgt_key_padding_mask:   Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal:          bool = False,
        memory_is_causal:       bool = False,
    ) -> torch.Tensor:
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(LRPResidualAdd.apply(tgt, tgt2))

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(LRPResidualAdd.apply(tgt, tgt2))

        # FFN
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt  = self.norm3(LRPResidualAdd.apply(tgt, tgt2))
        return tgt


# ---------------------------------------------------------------------------
# NormalizeImageNet
# ---------------------------------------------------------------------------

class NormalizeImageNet(nn.Module):
    """ImageNet normalisation as an nn.Module so zennit can intercept it."""
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.MEAN.to(x.device, dtype=x.dtype)
        std  = self.STD.to(x.device,  dtype=x.dtype)
        return (x / 255.0 - mean) / std


# ---------------------------------------------------------------------------
# Residual-block patching helpers (ResNet34 encoders + GPT fusion blocks)
# ---------------------------------------------------------------------------

def _basic_block_forward_explicit(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for timm.models.resnet.BasicBlock.forward with the
    residual add routed through LRPResidualAdd instead of the raw
    `x += shortcut`. Mirrors timm's own forward (timm/models/resnet.py)
    exactly except for that one line — if a future timm version changes the
    BasicBlock forward structure, update this to match.
    """
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.drop_block(x)
    x = self.act1(x)
    x = self.aa(x)

    x = self.conv2(x)
    x = self.bn2(x)

    if self.se is not None:
        x = self.se(x)
    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)

    x = LRPResidualAdd.apply(x, shortcut)
    x = self.act2(x)
    return x


def _patch_resnet_basic_blocks(module: nn.Module) -> int:
    """
    Monkey-patch every timm ResNet BasicBlock inside `module` in place so its
    residual add uses LRPResidualAdd. `module` must be a private deep copy
    (never the live driving-agent backbone), since this mutates instance
    `.forward` bindings and would otherwise change inference outside of LRP.

    Raises if any Bottleneck block is found (resnet50+): those have a
    different forward structure and are not covered by this function.

    Returns the number of BasicBlocks patched, so callers can assert the
    patch actually found something — silently matching zero blocks (e.g.
    from a class/version mismatch) would reproduce exactly the WoR
    ResNetCanonizer bug this fix is meant to avoid (see design_decisions.md).
    """
    n_patched = 0
    n_bottleneck = 0
    for m in module.modules():
        if isinstance(m, TimmBasicBlock):
            m.forward = _basic_block_forward_explicit.__get__(m, type(m))
            n_patched += 1
        elif isinstance(m, TimmBottleneck):
            n_bottleneck += 1
    if n_bottleneck > 0:
        raise NotImplementedError(
            f"{n_bottleneck} timm Bottleneck block(s) found (e.g. resnet50+); "
            "_patch_resnet_basic_blocks only handles BasicBlock (resnet18/34). "
            "Extend it to cover Bottleneck before switching to a deeper backbone."
        )
    return n_patched


def _gpt_block_forward_explicit(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for transfuser_backbone.Block.forward with both
    residual adds routed through LRPResidualAdd instead of raw `+`.
    self.attn must already be a SelfAttentionExplicit (AttnLRP-aware) —
    this is assigned by the caller before binding this forward.
    """
    x = LRPResidualAdd.apply(x, self.attn(self.ln1(x)))
    x = LRPResidualAdd.apply(x, self.mlp(self.ln2(x)))
    return x


# ---------------------------------------------------------------------------
# TFv6FullModelForLRP — backbone + planning decoder in a single LRP-ready module
# ---------------------------------------------------------------------------

class TFv6FullModelForLRP(nn.Module):
    """
    Wraps the TransFuser backbone and PlanningDecoder into a single nn.Module
    so a single zennit composite context covers the full attribution graph.

    Forward:
        rgb [B, 3, H, W] → speed_query token [B, 256]

    The speed_query is the F_c node space (Option B, 256-dim).
    LRP1: seed from target_speed_decoder(speed_query).max(-1) → backprop to speed_query.
    LRP2: one-hot at speed_query[:, k] → backprop to rgb.
    """

    def __init__(self, backbone, planning_decoder):
        """
        Parameters
        ----------
        backbone         : TransfuserBackbone (eval mode)
        planning_decoder : PlanningDecoder (eval mode)
        """
        super().__init__()

        cfg = backbone.config

        # --- Backbone components ---
        self.config               = cfg
        self.lidar_channel_to_img = backbone.lidar_channel_to_img
        self.img_channel_to_lidar = backbone.img_channel_to_lidar
        self.avgpool_img          = backbone.avgpool_img
        self.avgpool_lidar        = backbone.avgpool_lidar
        # NormalizeImageNet instead of fn.normalize_imagenet: the functional
        # version uses x.clone() + in-place channel writes, which create
        # CopySlices autograd nodes that break zennit hook pairing.
        self.normalize            = NormalizeImageNet()

        # image_encoder / lidar_encoder are deep-copied — NOT shared with the
        # live driving-agent backbone — because their BasicBlock residual
        # connections are monkey-patched below to route through
        # LRPResidualAdd instead of raw `+=`; mutating the live model's
        # forward would change inference outside of LRP. The patch only
        # changes backward/relevance, not forward values (LRPResidualAdd.
        # forward returns a+b unchanged), so get_backbone_features() and other
        # no-grad forward-only paths through these copies stay numerically
        # identical to going through the original backbone.
        self.image_encoder = copy.deepcopy(backbone.image_encoder)
        self.lidar_encoder = copy.deepcopy(backbone.lidar_encoder)
        n_img_blocks = _patch_resnet_basic_blocks(self.image_encoder)
        n_lid_blocks = _patch_resnet_basic_blocks(self.lidar_encoder)
        assert n_img_blocks > 0, (
            "image_encoder: no timm BasicBlock found to patch for "
            "LRPResidualAdd — did the backbone architecture change from resnet34?"
        )
        assert n_lid_blocks > 0, (
            "lidar_encoder: no timm BasicBlock found to patch for "
            "LRPResidualAdd — did the backbone architecture change from resnet34?"
        )

        # Deep-copy transformers → replace SelfAttention with explicit
        # attention, and patch the block-level residual adds.
        self.transformers = copy.deepcopy(backbone.transformers)
        for gpt in self.transformers:
            for block in gpt.blocks:
                block.attn = SelfAttentionExplicit.from_module(block.attn)
                block.forward = _gpt_block_forward_explicit.__get__(block, type(block))

        # --- Planning decoder components (deep-copied) ---
        pd = copy.deepcopy(planning_decoder)

        # Replace nn.TransformerDecoderLayer with explicit versions
        new_layers = nn.ModuleList([
            TransformerDecoderLayerExplicit.from_module(layer)
            for layer in pd.transformer_decoder.layers
        ])
        pd.transformer_decoder.layers = new_layers

        self.planning_context_encoder = pd.planning_context_encoder
        self.transformer_decoder      = pd.transformer_decoder
        self.query                    = pd.query             # nn.Parameter [1, Q, 256]
        self.target_speed_decoder     = pd.target_speed_decoder
        # Waypoint decoder — present only when predict_temporal_spatial_waypoints is True
        self.wp_decoder = pd.wp_decoder if hasattr(pd, 'wp_decoder') else None

        # Compute speed query index (mirrors PlanningDecoder.forward)
        idx = 0
        if cfg.predict_spatial_path:
            idx += cfg.num_route_points_prediction
        if cfg.predict_temporal_spatial_waypoints:
            idx += cfg.num_way_points_prediction
        self._speed_query_idx = idx
        # First waypoint query index: queries[:, _wp_start : _speed_query_idx]
        _route_offset  = cfg.num_route_points_prediction if cfg.predict_spatial_path else 0
        self._wp_start = _route_offset

    # ------------------------------------------------------------------
    # Backbone helper (mirrors TFv6ImageBackboneForLRP)
    # ------------------------------------------------------------------

    def _make_lidar(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h = self.config.lidar_height_pixel
        w = self.config.lidar_width_pixel
        xs = torch.linspace(0, 1, w)
        ys = torch.linspace(0, 1, h)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        lidar = torch.zeros(1, 2, h, w, device=device, dtype=dtype)
        lidar[0, 0] = y_grid
        lidar[0, 1] = x_grid
        return lidar  # no requires_grad

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
        idx:            int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_pool = self.avgpool_img(image_features)
        lid_pool = self.avgpool_lidar(lidar_features)
        lid_pool = self.lidar_channel_to_img[idx](lid_pool)

        img_out, lid_out = self.transformers[idx](img_pool, lid_pool)

        lid_out = self.img_channel_to_lidar[idx](lid_out)
        img_out = F.interpolate(img_out, size=image_features.shape[2:],
                                mode="bilinear", align_corners=False)
        lid_out = F.interpolate(lid_out, size=lidar_features.shape[2:],
                                mode="bilinear", align_corners=False)

        return (
            LRPResidualAdd.apply(image_features, img_out),
            LRPResidualAdd.apply(lidar_features, lid_out),
        )

    def _run_backbone(
        self, rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (lidar_features, image_features) — both after 4 GPT fusions."""
        image_features = self.normalize(rgb)
        lidar_features = self._make_lidar(rgb.device, rgb.dtype).expand(rgb.shape[0], -1, -1, -1)

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        if len(self.image_encoder.return_layers) > 4:
            image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        for i in range(4):
            image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self._fuse(image_features, lidar_features, i)

        return lidar_features, image_features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rgb:         torch.Tensor,
        data:        dict,
        _return_wps: bool = False,
        _return_wp_queries: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            rgb         : [B, 3, H, W] float, requires_grad=True
            data        : status dict with at least the keys the config expects
                          (speed, command, target_point, …)
            _return_wps : if True and wp_decoder is available, also return the
                          predicted future waypoints [B, N_wp, 2].  Used only by
                          get_planning_action_and_features for MDX-v2; all other
                          callers leave this at False.
            _return_wp_queries : if True, also return the raw waypoint query
                          tokens [B, N_wp, 256] — the waypoint head's F_c
                          analogue (penultimate layer before the per-token
                          wp_decoder Linear).  Used by _attribute_wp_to_input
                          for the ADD_WAYPOINT_SEEDS profile block.
        Returns:
            speed_query : [B, 256] — the F_c node space
            (optionally, when _return_wps=True and wp_decoder is present)
            (speed_query, waypoints) where waypoints is [B, N_wp, 2]
            (optionally, when _return_wp_queries=True)
            (speed_query, wp_queries) where wp_queries is [B, N_wp, 256]
        """
        bev_features, _ = self._run_backbone(rgb)

        context_tokens = self.planning_context_encoder(
            bev_features=bev_features,
            radar_logits=None,
            radar_predictions=None,
            data=data,
            log={},
        )

        bs = context_tokens.shape[0]
        queries = self.transformer_decoder(
            self.query.repeat(bs, 1, 1),
            context_tokens,
        )

        speed_query = queries[:, self._speed_query_idx]   # [B, 256]

        if _return_wp_queries:
            wp_queries = queries[:, self._wp_start : self._speed_query_idx]
            return speed_query, wp_queries                # [B, N_wp, 256]

        if _return_wps and self.wp_decoder is not None:
            wp_queries = queries[:, self._wp_start : self._speed_query_idx]
            waypoints  = torch.cumsum(self.wp_decoder(wp_queries), dim=1)  # [B, N_wp, 2]
            return speed_query, waypoints

        return speed_query


# ---------------------------------------------------------------------------
# Public LRP class
# ---------------------------------------------------------------------------

class LRPTFv6Model:
    """
    LRP attribution for TransFuser v6 (visiononly_resnet34, LTF mode).

    Provides the same public interface as LRPCameraModel and LRPLBCModel so
    that ATOMsCarla works unchanged with any agent.

    Node space (Option B)
    ---------------------
    256-dim speed-query token inside PlanningDecoder TransformerDecoder.
    This is the layer closest to the agent's speed/action decision and
    corresponds to F_c in the ATOMs paper (the "final world model" layer).

    LRP1 (output → fc):
        Seeds from max of target_speed_decoder logits.
        Backpropagates through target_speed_decoder + full TransformerDecoder
        + backbone to speed_query level.  Returns [256] relevances.

    LRP2 (fc → input):
        One-hot seed at speed_query node k.
        Backpropagates through full model to input pixels.
        Returns [1, 3, H, W] pixel relevance map.

    Parameters
    ----------
    backbone_eval      : TransfuserBackbone in .eval() mode
    planning_decoder   : PlanningDecoder in .eval() mode
    uitb               : use AlphaBeta(2,1) instead of (1,0) for Conv/FFN.
                         (2,1) retains some negative-activation relevance
                         instead of discarding it entirely (pure z+ = (1,0)
                         clips all negative contributions) -- a candidate
                         lever against the severe relevance attenuation
                         documented in design_decisions.md ("TFV6 LRP:
                         residual/skip-connection conservation"), untested
                         as of 2026-07-02.
    zero_bias          : exclude bias terms from the AlphaBeta denominator
                         for AnyLinear layers (zennit's zero_params='bias'),
                         matching WoR/LBC's existing composite. TFV6 never
                         had this; D11 measured target_speed_decoder's bias
                         absorption at only 1-4%, so this is expected to be
                         a minor effect, not a fix for the 4-orders-of-
                         magnitude attenuation on its own. No effect on
                         Convolution layers (bias=False by ResNet convention,
                         nothing to zero).
    device             : torch device
    """

    def __init__(
        self,
        backbone_eval,
        planning_decoder   = None,
        uitb:   bool       = False,
        zero_bias: bool    = False,
        device: torch.device = None,
    ):
        self.uitb             = uitb
        self.zero_bias        = zero_bias
        self.alpha, self.beta = (2, 1) if uitb else (1, 0)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        assert not backbone_eval.training, "backbone must be in eval() mode"

        if planning_decoder is None:
            raise ValueError(
                "planning_decoder is required for Option B (LRP from PlanningDecoder speed query). "
                "Pass model.planning_decoder when constructing LRPTFv6Model."
            )
        assert not planning_decoder.training, "planning_decoder must be in eval() mode"
        self.full_model = (
            TFv6FullModelForLRP(backbone_eval, planning_decoder)
            .to(self.device).eval()
        )
        self.node_dim = 256

        self.composite   = self._create_composite()
        self._data_cache: Optional[dict] = None   # set by update_context

        # Vision-only guard: in non-LTF mode the backbone reads rasterized_lidar
        # from the data dict.  Since .npz frame files never store that key,
        # non-LTF would silently zero the LiDAR stream — worse, if someone adds
        # LiDAR saving later the model would suddenly use it.  Fail loudly now.
        assert backbone_eval.config.LTF, (
            "LRPTFv6Model requires LTF=True (synthetic positional-grid LiDAR). "
            "Non-LTF mode expects rasterized_lidar in every data dict, which "
            "frame .npz files do not provide."
        )

    # ------------------------------------------------------------------
    # Setup / API compatibility
    # ------------------------------------------------------------------

    def update_context(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: Optional[torch.Tensor] = None,
        spd:      float                  = None,
        cmd:      Optional[int]          = None,
        data:     Optional[dict]         = None,
        target_points: Optional[dict]    = None,
    ) -> None:
        """
        Store the per-frame status dict for use in forward_relevance.

        For TFV6 with Option B, data must contain the keys that
        PlanningContextEncoder expects (speed, command, target_point, …).
        If data is None, a minimal dict is built from spd, cmd and (when
        given) target_points.  Pass cmd (integer 0–5) so the command token
        is a valid one-hot rather than an all-zero vector, and target_points
        (dict with 'target_point'/'target_point_previous'/'target_point_next'
        in ego-frame meters) so route conditioning matches deployment.
        """
        assert not self.full_model.training, "model must be in eval() mode"
        self._data_cache = data if data is not None else _make_minimal_data(
            spd or 0.0, self.device,
            cmd=cmd if cmd is not None else 3,
            target_points=target_points,
        )

    def _create_composite(self) -> SpecialFirstLayerMapComposite:
        # zero_bias only applies to AnyLinear, matching WoR/LBC's composite
        # exactly (see docstring). Convolution layers use bias=False by
        # ResNet convention, so there is nothing to zero there regardless.
        linear_zero_params = 'bias' if self.zero_bias else None
        layer_map = [
            (Activation,      Pass()),
            # BatchNorm2d is folded into preceding Conv by SequentialMergeBatchNorm
            # before LRP runs; Pass here is a no-op fallback for any residual BN.
            (nn.BatchNorm2d,  Pass()),
            (nn.LayerNorm,    Pass()),
            # AttentionLinear before AnyLinear so it matches first
            (AttentionLinear, Epsilon(epsilon=1e-2)),
            (Convolution,     AlphaBeta(alpha=self.alpha, beta=self.beta)),
            (AnyLinear,       AlphaBeta(alpha=self.alpha, beta=self.beta,
                                        zero_params=linear_zero_params)),
        ]
        first_map = [(Convolution, WSquare())]
        return SpecialFirstLayerMapComposite(
            layer_map=layer_map,
            first_map=first_map,
            canonizers=[SequentialMergeBatchNorm()],
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input(self, rgb: torch.Tensor) -> torch.Tensor:
        expected_w = conf.N_CAMERAS * 384
        if rgb.shape[-1] != expected_w:
            warnings.warn(
                f"Input image width {rgb.shape[-1]} px does not match "
                f"conf.N_CAMERAS={conf.N_CAMERAS} × 384 = {expected_w} px. "
                "If using legacy 3-camera data set conf.N_CAMERAS = 3.",
                UserWarning, stacklevel=3,
            )
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
        LRP attribution for TFV6.

        forced_brake / forced_drive (used by PLOT_COMPARATIVE_REL):
          forced_brake=True  — seed at speed bin 0 (stop/brake), regardless
                               of the agent's actual prediction.
          forced_drive=True  — seed at the highest-probability non-brake bin
                               (argmax of bins 1–7), regardless of prediction.
          Neither flag        — default softmax-distribution seed.

        Returns
        -------
        (wide_rel, None, 1.0, is_brake)
            output→fc    : wide_rel is [256] node relevances (LRP1)
            fc→input     : wide_rel is [1, 3, H, W] pixel map (LRP2)
            output→input : wide_rel is [1, 3, H, W] pixel map
            is_brake      : True when the agent's top predicted bin is bin 0
        """
        rgb_x = self._prepare_input(wide_rgb)
        data  = self._get_data(spd)

        if beg == "output" and end == "fc":
            fc_rel, is_brake = self._attribute_to_fc(rgb_x, data, forced_brake, forced_drive)
            return fc_rel, None, 1.0, is_brake

        elif beg == "fc" and end == "input":
            if node_id is None:
                # Layer-level: seed from positive F_c activations (ATOMs mode 2).
                # forced_brake/drive not applicable here (seed is F_c activations,
                # not the output distribution).
                wide_rel = self._attribute_fc_to_input(rgb_x, data)
            else:
                wide_rel = self._attribute_backbone(rgb_x, data, self._one_hot_node(node_id))
            # is_brake requires a forward pass; skip it for fc→input to avoid cost.
            return wide_rel, None, 1.0, False

        elif beg == "wp_fc" and end == "input":
            # Waypoint-head mode-2 analogue: seed positive activations of the
            # waypoint query tokens (the wp head's F_c), backprop to pixels.
            wide_rel = self._attribute_wp_to_input(rgb_x, data)
            return wide_rel, None, 1.0, False

        elif beg == "output" and end == "input":
            # True output→input: backprop through target_speed_decoder + backbone.
            wide_rel, is_brake = self._attribute_true_output_to_input(
                rgb_x, data, forced_brake, forced_drive
            )
            return wide_rel, None, 1.0, is_brake

        else:
            raise ValueError(f"Unsupported mode: beg='{beg}', end='{end}'")

    # ------------------------------------------------------------------
    # Internal attribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_speed_seed(
        speed_logits: torch.Tensor,
        forced_brake: bool,
        forced_drive: bool,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Return (seed, is_brake) for LRP seeding at the speed-logit level.

        forced_brake  → one-hot at bin 0 (0 m/s / stop).
        forced_drive  → one-hot at the highest-probability non-brake bin
                        (argmax of logits[1:]).  If the model is already
                        driving this equals the normal argmax; if it is
                        braking this gives the best counterfactual drive bin.
        default       → softmax distribution over all bins (smooth, stable).

        is_brake reflects the model's ACTUAL prediction (argmax == bin 0),
        independent of which forced flag is set.
        """
        is_brake = bool(speed_logits.detach().argmax(dim=-1).item() == 0)
        if forced_brake:
            seed = torch.zeros_like(speed_logits)
            seed[0, 0] = 1.0
        elif forced_drive:
            drive_cls = int(speed_logits[0, 1:].detach().argmax().item()) + 1
            seed = torch.zeros_like(speed_logits)
            seed[0, drive_cls] = 1.0
        else:
            seed = torch.softmax(speed_logits.detach(), dim=-1)
        return seed, is_brake

    def _get_data(self, spd: Optional[float]) -> dict:
        """Return cached data dict, optionally updating speed."""
        if self._data_cache is not None:
            return self._data_cache
        return _make_minimal_data(spd or 0.0, self.device)

    def _attribute_to_fc(
        self,
        rgb_x:        torch.Tensor,
        data:         dict,
        forced_brake: bool = False,
        forced_drive: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        LRP1: backpropagate from speed logits to speed_query node space.
        Returns (node_rel [256], is_brake).
        """
        with torch.enable_grad():
            with self.composite.context(self.full_model):
                speed_query  = self.full_model(rgb_x, data)
                speed_logits = self.full_model.target_speed_decoder(speed_query)
                seed, is_brake = self._make_speed_seed(speed_logits, forced_brake, forced_drive)
                (node_rel,) = torch.autograd.grad(
                    outputs      = speed_logits,
                    inputs       = [speed_query],
                    grad_outputs = seed,
                )
        return node_rel.squeeze(0).detach().cpu(), is_brake

    def _attribute_backbone(
        self,
        rgb_x:    torch.Tensor,
        data:     dict,
        selector,
    ) -> torch.Tensor:
        """LRP2: single-node seed at speed_query → input pixels."""
        with torch.enable_grad():
            with self.composite.context(self.full_model):
                speed_query = self.full_model(rgb_x, data)    # [1, 256]
                grad_out    = selector(speed_query)
                (rgb_rel,)  = torch.autograd.grad(
                    outputs      = speed_query,
                    inputs       = [rgb_x],
                    grad_outputs = grad_out,
                )
        return rgb_rel.detach().cpu()   # [1, 3, H, W]

    def _attribute_fc_to_input(self, rgb_x: torch.Tensor, data: dict) -> torch.Tensor:
        """
        Layer-level LRP (ATOMs mode 2): seed from positive F_c activations,
        backprop to pixels.  Does NOT pass through target_speed_decoder -
        weights nodes by raw activation magnitude, not decision relevance.
        """
        with torch.enable_grad():
            with self.composite.context(self.full_model):
                speed_query = self.full_model(rgb_x, data)      # [1, 256]
                seed        = speed_query.clamp(min=0).detach()
                (rgb_rel,)  = torch.autograd.grad(
                    outputs      = speed_query,
                    inputs       = [rgb_x],
                    grad_outputs = seed,
                )
        return rgb_rel.detach().cpu()   # [1, 3, H, W]

    def _attribute_wp_to_input(self, rgb_x: torch.Tensor, data: dict) -> torch.Tensor:
        """
        Waypoint-head layer-level LRP (conf.ADD_WAYPOINT_SEEDS): the exact
        mode-2 analogue of _attribute_fc_to_input, applied to the waypoint
        head.  The waypoint query tokens [B, N_wp, 256] are the head's F_c
        equivalent — wp_decoder is a single per-token Linear(256→2), so these
        tokens are the penultimate representation of the lateral/spatial
        decision, exactly as speed_query is for the longitudinal one.

        Seed: positive activations of all N_wp tokens at once (clamp(min=0)),
        backpropagated to pixels in one pass — same seeding rule, same
        composite, same cost as the default mode-2 pass.
        """
        if self.full_model._wp_start >= self.full_model._speed_query_idx:
            raise RuntimeError(
                "ADD_WAYPOINT_SEEDS requires waypoint query tokens "
                "(predict_temporal_spatial_waypoints=True in the model config)."
            )
        with torch.enable_grad():
            with self.composite.context(self.full_model):
                _, wp_queries = self.full_model(rgb_x, data, _return_wp_queries=True)
                seed          = wp_queries.clamp(min=0).detach()
                (rgb_rel,)    = torch.autograd.grad(
                    outputs      = wp_queries,
                    inputs       = [rgb_x],
                    grad_outputs = seed,
                )
        return rgb_rel.detach().cpu()   # [1, 3, H, W]

    def _attribute_true_output_to_input(
        self,
        rgb_x:        torch.Tensor,
        data:         dict,
        forced_brake: bool = False,
        forced_drive: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Output-weighted pixel attribution (ATOMs mode 3).

        Two-step implementation to avoid numerical explosion:

        Step 1 — LRP1 (stable, short path):
            speed_logits → speed_query via target_speed_decoder with AlphaBeta.
            Yields 256-dim node relevances R_k ≥ 0.

        Step 2 — LRP2 (stable, same path as _attribute_backbone):
            speed_query → rgb_x using R_k as seed.

        Because autograd.grad is linear in grad_outputs, this gives exactly
        Σ_k R_k · pixel_map_k — the output-weighted sum of per-node pixel maps,
        equivalent to ATOMs mode 1 in a single backward pass.

        The single-pass alternative (outputs=speed_logits, inputs=[rgb_x]) is
        numerically unstable: backpropagating through both target_speed_decoder
        and transformer_decoder in one pass produces near-zero LRPMatMul
        denominators that cause relevance explosion (~10^15 scale).
        """
        with torch.enable_grad():
            with self.composite.context(self.full_model):
                speed_query  = self.full_model(rgb_x, data)
                speed_logits = self.full_model.target_speed_decoder(speed_query)
                seed, is_brake = self._make_speed_seed(speed_logits, forced_brake, forced_drive)

                # Step 1: LRP1 — from speed_logits to speed_query only
                (node_rel,) = torch.autograd.grad(
                    outputs      = speed_logits,
                    inputs       = [speed_query],
                    grad_outputs = seed,
                    retain_graph = True,   # keep graph for step 2
                )

                # Step 2: output-weighted LRP2 — from speed_query to pixels
                (rgb_rel,) = torch.autograd.grad(
                    outputs      = speed_query,
                    inputs       = [rgb_x],
                    grad_outputs = node_rel.detach(),
                )
        return rgb_rel.detach().cpu(), is_brake

    # ------------------------------------------------------------------
    # Inference helpers (no LRP, no grad)
    # ------------------------------------------------------------------

    def get_speed_logits(
        self,
        wide_rgb: torch.Tensor,
        cmd:      int   = 3,
        spd:      float = 0.0,
    ) -> "np.ndarray":
        """
        Forward pass → 8-bin speed logits from target_speed_decoder.

        Used by PEOCDetector (Sedlmeier et al., 2020) to compute policy entropy
        H(π) as the OOD anomaly score.

        Parameters
        ----------
        wide_rgb : [1, 3, H, W] float tensor (raw uint8 range [0, 255])
        cmd      : navigation command integer (0–5, leaderboard one-hot index)
        spd      : current speed in m/s

        Returns
        -------
        np.ndarray [8]  raw speed logits (before softmax)
        """
        import numpy as np
        wide_t = wide_rgb.float().to(self.device)
        data   = _make_minimal_data(float(spd), self.device, cmd=int(cmd))
        with torch.no_grad():
            speed_query  = self.full_model(wide_t, data)
            speed_logits = self.full_model.target_speed_decoder(speed_query)
        return speed_logits.squeeze(0).cpu().numpy()  # [8]

    def get_backbone_features(self, wide_rgb: torch.Tensor) -> "np.ndarray":
        """
        Extract 512-dim globally-averaged backbone features for MDX detection.

        Runs the image through the ResNet34 + GPT-fusion backbone and applies
        adaptive average pooling to produce a flat feature vector matching the
        penultimate-layer feature extraction used by MDXDetector.

        Parameters
        ----------
        wide_rgb : [1, 3, H, W] float tensor (raw uint8 range [0, 255])

        Returns
        -------
        np.ndarray [512]  ReLU-clamped, globally-pooled backbone features
        """
        import numpy as np
        wide_t = wide_rgb.float().to(self.device)
        with torch.no_grad():
            _, image_features = self.full_model._run_backbone(wide_t)
            # image_features: [1, 512, H', W']  (H'=12, W'=72 for 384×2304 input)
            pooled = F.adaptive_avg_pool2d(image_features, (1, 1))
            feat   = pooled.flatten(1).clamp(min=0).squeeze(0)  # [512]
        return feat.cpu().numpy()

    # MDX-v2 speed bins — used to decode expected target speed from distribution.
    _SPEED_BINS_PROXY = [0., 4., 8., 10., 13.89, 16., 17.78, 20.]  # m/s

    def get_planning_action_and_features(
        self,
        wide_rgb: torch.Tensor,
        cmd:      int   = 4,
        spd:      float = 0.0,
    ) -> "tuple":
        """
        Single forward pass → 256-d F_c feature + (steer_proxy, throttle_proxy, brake_proxy).

        Used in the MDX-v2 baseline fit loop. Combines feature extraction and
        action proxy derivation so each frame requires only one forward pass.

        steer_proxy  : mean lateral (x) offset of predicted future waypoints.
                       Non-degenerate even on straight roads.  Same signal as
                       the PGD steer target.
        throttle/brake: decoded expected target speed → min(v/20, 1) / v<0.5.
                        Reflects the policy's intent rather than ego speed.

        Parameters
        ----------
        wide_rgb : [1, 3, H, W] float tensor (raw uint8 range [0, 255])
        cmd      : navigation command integer (0–5)
        spd      : current speed in m/s

        Returns
        -------
        (feature: np.ndarray[256], steer: float, throttle: float, brake: float)
        """
        import numpy as np
        wide_t = wide_rgb.float().to(self.device)
        data   = _make_minimal_data(float(spd), self.device, cmd=int(cmd))
        with torch.no_grad():
            result = self.full_model(wide_t, data, _return_wps=True)
        if isinstance(result, tuple):
            speed_query, waypoints = result
            steer = float(waypoints[..., 0].mean())
        else:
            speed_query = result
            steer = 0.0   # wp_decoder unavailable — fall back to constant
        feature      = speed_query.squeeze(0).cpu().numpy()   # [256]
        speed_logits = self.full_model.target_speed_decoder(speed_query)
        bins  = torch.tensor(self._SPEED_BINS_PROXY, device=self.device, dtype=torch.float32)
        tgt_v = float((torch.softmax(speed_logits.squeeze(0), dim=-1) * bins).sum())
        return feature, steer, min(tgt_v / 20.0, 1.0), (1.0 if tgt_v < 0.5 else 0.0)

    def get_fc_features(
        self,
        wide_rgb: torch.Tensor,
        cmd:      int   = 4,
        spd:      float = 0.0,
    ) -> "np.ndarray":
        """
        Extract 256-d speed_query (F_c) features for MDX-v2 test scoring.

        Runs backbone + planning decoder; returns the F_c node vector just
        before target_speed_decoder — the true penultimate layer for TFV6.

        Parameters
        ----------
        wide_rgb : [1, 3, H, W] float tensor (raw uint8 range [0, 255])
        cmd      : navigation command integer (0–5)
        spd      : current speed in m/s

        Returns
        -------
        np.ndarray [256]  speed_query (F_c) feature vector
        """
        import numpy as np
        wide_t = wide_rgb.float().to(self.device)
        data   = _make_minimal_data(float(spd), self.device, cmd=int(cmd))
        with torch.no_grad():
            speed_query = self.full_model(wide_t, data)   # [B, 256]
        return speed_query.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Selectors
    # ------------------------------------------------------------------

    def _one_hot_node(self, node: int):
        def selector(output: torch.Tensor) -> torch.Tensor:
            vec = torch.zeros_like(output)
            vec[:, node] = 1.0
            return vec
        return selector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_minimal_data(
    spd: float,
    device: torch.device,
    cmd: int = 3,
    target_points: Optional[dict] = None,
) -> dict:
    """
    Build a minimal data dict for PlanningContextEncoder when real frame
    data is not available.

    cmd : navigation command integer (0–5, CARLA leaderboard one-hot index).
          Defaults to 3 (FOLLOW_LANE).  Converted to a one-hot vector of
          length 6 so the command token carries a valid directional signal.

    target_points : optional dict with keys 'target_point',
          'target_point_previous', 'target_point_next', each a length-2
          array-like in ego-frame meters (same convention as the training
          dataloader / sensor agent).  When None, TPs are zero — degenerate
          route conditioning; the deployed model uses three TP tokens, so
          pass real values whenever the npz provides them
          (conf.USE_REAL_TARGET_POINTS).
    """
    cmd_vec = torch.zeros(1, 6, dtype=torch.float32, device=device)
    cmd_vec[0, max(0, min(cmd, 5))] = 1.0

    def _tp(key: str) -> torch.Tensor:
        if target_points is not None and target_points.get(key) is not None:
            vals = [float(v) for v in target_points[key]][:2]
            return torch.tensor([vals], dtype=torch.float32, device=device)
        return torch.zeros(1, 2, dtype=torch.float32, device=device)

    return {
        "speed":              torch.tensor([[spd]], dtype=torch.float32, device=device),
        "command":            cmd_vec,
        "target_point":       _tp("target_point"),
        "target_point_previous": _tp("target_point_previous"),
        "target_point_next":  _tp("target_point_next"),
        "acceleration":       torch.zeros(1, 1, dtype=torch.float32, device=device),
    }
