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
    AttentionLinear:     Epsilon(ε=1e-6)    ← K/Q/V/proj in all attention blocks
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

See design_decisions.md for rationale.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zennit.rules import Pass, WSquare, AlphaBeta, Epsilon
from zennit.types import Convolution, Activation
from zennit.types import Linear as AnyLinear
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.canonizers import SequentialMergeBatchNorm

from typing import Dict, Optional, Tuple


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
        tgt = self.norm1(tgt + tgt2)

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + tgt2)

        # FFN
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt  = self.norm3(tgt + tgt2)
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

        # --- Backbone components (shared, read-only) ---
        self.config              = cfg
        self.image_encoder       = backbone.image_encoder
        self.lidar_encoder       = backbone.lidar_encoder
        self.lidar_channel_to_img= backbone.lidar_channel_to_img
        self.img_channel_to_lidar= backbone.img_channel_to_lidar
        self.avgpool_img         = backbone.avgpool_img
        self.avgpool_lidar       = backbone.avgpool_lidar
        # NormalizeImageNet instead of fn.normalize_imagenet: the functional
        # version uses x.clone() + in-place channel writes, which create
        # CopySlices autograd nodes that break zennit hook pairing.
        self.normalize           = NormalizeImageNet()

        # Deep-copy transformers → replace SelfAttention with explicit
        self.transformers = copy.deepcopy(backbone.transformers)
        for gpt in self.transformers:
            for block in gpt.blocks:
                block.attn = SelfAttentionExplicit.from_module(block.attn)

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

        # Compute speed query index (mirrors PlanningDecoder.forward)
        idx = 0
        if cfg.predict_spatial_path:
            idx += cfg.num_route_points_prediction
        if cfg.predict_temporal_spatial_waypoints:
            idx += cfg.num_way_points_prediction
        self._speed_query_idx = idx

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

        return image_features + img_out, lidar_features + lid_out

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

    def forward(self, rgb: torch.Tensor, data: dict) -> torch.Tensor:
        """
        Args:
            rgb  : [B, 3, H, W] float, requires_grad=True
            data : status dict with at least the keys the config expects
                   (speed, command, target_point, …)
        Returns:
            speed_query : [B, 256] — the F_c node space
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

        return queries[:, self._speed_query_idx]   # [B, 256]


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
    uitb               : use AlphaBeta(2,1) instead of (1,0) for Conv/FFN
    device             : torch device
    """

    def __init__(
        self,
        backbone_eval,
        planning_decoder   = None,
        uitb:   bool       = False,
        device: torch.device = None,
    ):
        self.uitb             = uitb
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
    ) -> None:
        """
        Store the per-frame status dict for use in forward_relevance.

        For TFV6 with Option B, data must contain the keys that
        PlanningContextEncoder expects (speed, command, target_point, …).
        If data is None, a minimal dict is built from spd and cmd.  Pass
        cmd (integer 0–5) so the command token is a valid one-hot rather
        than an all-zero vector, which would distort LRP attributions.
        """
        assert not self.full_model.training, "model must be in eval() mode"
        self._data_cache = data if data is not None else _make_minimal_data(
            spd or 0.0, self.device, cmd=cmd if cmd is not None else 3
        )

    def _create_composite(self) -> SpecialFirstLayerMapComposite:
        layer_map = [
            (Activation,      Pass()),
            # BatchNorm2d is folded into preceding Conv by SequentialMergeBatchNorm
            # before LRP runs; Pass here is a no-op fallback for any residual BN.
            (nn.BatchNorm2d,  Pass()),
            (nn.LayerNorm,    Pass()),
            # AttentionLinear before AnyLinear so it matches first
            (AttentionLinear, Epsilon(epsilon=1e-2)),
            (Convolution,     AlphaBeta(alpha=self.alpha, beta=self.beta)),
            (AnyLinear,       AlphaBeta(alpha=self.alpha, beta=self.beta)),
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

def _make_minimal_data(spd: float, device: torch.device, cmd: int = 3) -> dict:
    """
    Build a minimal data dict for PlanningContextEncoder when real frame
    data is not available.

    cmd : navigation command integer (0–5, CARLA leaderboard one-hot index).
          Defaults to 3 (FOLLOW_LANE).  Converted to a one-hot vector of
          length 6 so the command token carries a valid directional signal.
          target_point and acceleration remain zero — these are secondary
          conditioning inputs that require data not stored in the npz.
    """
    cmd_vec = torch.zeros(1, 6, dtype=torch.float32)
    cmd_vec[0, max(0, min(cmd, 5))] = 1.0
    return {
        "speed":              torch.tensor([[spd]], dtype=torch.float32),
        "command":            cmd_vec,
        "target_point":       torch.zeros(1, 2, dtype=torch.float32),
        "target_point_previous": torch.zeros(1, 2, dtype=torch.float32),
        "target_point_next":  torch.zeros(1, 2, dtype=torch.float32),
        "acceleration":       torch.zeros(1, 1, dtype=torch.float32),
    }
