"""
dlwm/model.py
=============
DLWM (arXiv:2604.00969) 一阶段复现 —— 模型与渲染逻辑

架构：
    多视角图像 [B, N, 3, H, W]
        → ResNet-101 + FPN（多尺度图像特征）
        → Gaussian Lifter（初始化 3D Gaussian 锚点）
        → GaussianEncoder（迭代精化：xyz, scale, rot, opacity, semantic_feat[C]）
        → GaussianRasterizer.render(gaussians, camera)
            ↓
        depth_map    [B, 1, H, W]  — Expected-Depth (Z-buffer 积分)
        semantic_map [B, C, H, W]  — alpha-blending（logits，无激活）

关键设计：
  - semantic_features 在 3D 空间保持 raw logits（无 softmax/sigmoid）
  - 渲染后的 semantic_map 接 softmax → CE Loss
  - 使用 gsplat.rasterization 实现高效 CUDA alpha-blending
  - 若 gsplat 未安装，自动退化到纯 PyTorch EWA 实现（仅调试用）

依赖：
    pip install gsplat torchvision
"""

from __future__ import annotations

import math
import os
import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import builder as mmseg_builder
from mmdet3d.registry import MODELS as MMDET3D_MODELS


# ===========================================================================
# 数据容器
# ===========================================================================

@dataclass
class Gaussians:
    """3D Gaussian 参数容器。

    所有张量形状均为 [B, N, *]，N = Gaussian 数量。

    Attributes:
        means      : [B, N, 3]  中心坐标（ego/world 坐标，米）
        scales     : [B, N, 3]  各轴尺度（正值，sigmoid 映射后乘以 scale_range）
        rotations  : [B, N, 4]  归一化四元数 (w, x, y, z)
        opacities  : [B, N, 1]  不透明度 ∈ [0, 1]
        semantics  : [B, N, C]  语义特征（raw logits，无激活）
    """
    means:     torch.Tensor   # [B, N, 3]
    scales:    torch.Tensor   # [B, N, 3]
    rotations: torch.Tensor   # [B, N, 4]
    opacities: torch.Tensor   # [B, N, 1]
    semantics: torch.Tensor   # [B, N, C]


@dataclass
class Camera:
    """单视角相机参数。

    Attributes:
        K       : [B, 4, 4]  齐次内参矩阵
        ego2cam : [B, 4, 4]  ego → camera 变换（extrinsic）
        width   : 图像宽度（像素）
        height  : 图像高度（像素）
        near    : 近平面（米）
        far     : 远平面（米）
    """
    K:       torch.Tensor  # [B, 4, 4]
    ego2cam: torch.Tensor  # [B, 4, 4]
    width:   int
    height:  int
    near:    float = 0.1
    far:     float = 100.0


# ===========================================================================
# 图像特征提取：ResNet-101 + FPN
# ===========================================================================

class ImageFeatureExtractor(nn.Module):
    """使用 GaussianFormer 的 backbone/neck 组件提取多尺度特征。

    输入 ：[B*N, 3, H, W]  — B*N 张图像（批次×相机数）
    输出 ：List[Tensor]    — 多尺度特征，形状均为 [B*N, C_out, H_i, W_i]
           默认按 (H/4, H/8, H/16, H/32) 或配置指定层级
    """

    def __init__(
        self,
        out_channels: int = 128,
        num_levels:   int = 4,
        pretrained:   bool = True,
        img_backbone_out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        img_backbone_config: Optional[Dict] = None,
        img_neck_config: Optional[Dict] = None,
        pretrained_path: str = "ckpts/r101_dcn_fcos3d_pretrain.pth",
    ) -> None:
        super().__init__()
        self.img_backbone_out_indices = img_backbone_out_indices

        if img_backbone_config is None:
            img_backbone_config = dict(
                _delete_=True,
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True),
            )
        if img_neck_config is None:
            img_neck_config = dict(
                type='FPN',
                num_outs=num_levels,
                start_level=1,
                out_channels=out_channels,
                add_extra_convs='on_output',
                relu_before_extra_convs=True,
                in_channels=[256, 512, 1024, 2048],
            )

        self.img_backbone = mmseg_builder.build_backbone(copy.deepcopy(img_backbone_config))
        try:
            self.img_neck = mmseg_builder.build_neck(copy.deepcopy(img_neck_config))
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            warnings.warn(
                f"mmseg neck build failed, falling back to mmdet3d MODELS.build: {exc}",
                RuntimeWarning,
            )
            self.img_neck = MMDET3D_MODELS.build(copy.deepcopy(img_neck_config))

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path: str) -> None:
        if not os.path.isfile(pretrained_path):
            warnings.warn(
                f"Pretrained weights not found at '{pretrained_path}', continue with random init.",
                RuntimeWarning,
            )
            return

        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        model_state = {}
        has_img_prefix = False
        for key in state_dict.keys():
            if key.startswith('img_backbone.') or key.startswith('img_neck.'):
                has_img_prefix = True
                break
        if has_img_prefix:
            for key, value in state_dict.items():
                if key.startswith('img_backbone.') or key.startswith('img_neck.'):
                    model_state[key] = value
        else:
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    model_state[f'img_backbone.{key[len("backbone."):]}'] = value
                elif key.startswith('neck.'):
                    model_state[f'img_neck.{key[len("neck."):]}'] = value
            if not model_state:
                warnings.warn(
                    "No backbone/neck-prefixed keys found in checkpoint; trying backbone-only load.",
                    RuntimeWarning,
                )
                self.img_backbone.load_state_dict(state_dict, strict=False)
                return

        self.load_state_dict(model_state, strict=False)

    def forward(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            imgs: [B*N, 3, H, W]

        Returns:
            feats: List[tensor], each [B*N, C, H_i, W_i]
        """
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        invalid_indices = [
            idx for idx in self.img_backbone_out_indices
            if idx < 0 or idx >= len(img_feats_backbone)
        ]
        if invalid_indices:
            raise IndexError(
                f"img_backbone_out_indices has invalid indices {invalid_indices} "
                f"for backbone outputs with length {len(img_feats_backbone)}"
            )

        img_feats = [img_feats_backbone[idx] for idx in self.img_backbone_out_indices]
        img_feats = self.img_neck(img_feats)

        if isinstance(img_feats, dict):
            if 'fpn_out' in img_feats:
                img_feats = img_feats['fpn_out']
            else:
                img_feats = list(img_feats.values())
        elif isinstance(img_feats, tuple):
            img_feats = list(img_feats)

        return img_feats


# ===========================================================================
# Gaussian Lifter：初始化 Gaussian 锚点
# ===========================================================================

class GaussianLifter(nn.Module):
    """从可学习参数初始化 Gaussian 锚点。

    输出的锚点张量 [B, num_anchor, D] 中：
        D = 3 (xyz) + 3 (scale) + 4 (quat) + 1 (opacity) + C (semantics)

    Args:
        num_anchor:    Gaussian 数量
        embed_dims:    实例特征维度
        num_classes:   语义类别数 C
        pc_range:      [x_min, y_min, z_min, x_max, y_max, z_max]
        scale_range:   [scale_min, scale_max]
    """

    def __init__(
        self,
        num_anchor:  int,
        embed_dims:  int,
        num_classes: int,
        pc_range:    List[float],
        scale_range: List[float],
    ) -> None:
        super().__init__()
        self.num_anchor  = num_anchor
        self.embed_dims  = embed_dims
        self.num_classes = num_classes
        self.pc_range    = pc_range
        self.scale_range = scale_range

        # 可学习锚点向量 [N, D]，D = 3+3+4+1+C
        D = 3 + 3 + 4 + 1 + num_classes
        self.anchor = nn.Parameter(torch.randn(num_anchor, D) * 0.01)

        # 实例特征 [N, embed_dims]
        self.instance_feature = nn.Parameter(
            torch.zeros(num_anchor, embed_dims)
        )

    def init_weights(self) -> None:
        nn.init.normal_(self.anchor, std=0.01)
        nn.init.xavier_uniform_(self.instance_feature)

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            anchor:           [B, N, D]
            instance_feature: [B, N, embed_dims]
        """
        anchor = self.anchor.unsqueeze(0).expand(batch_size, -1, -1)
        feat   = self.instance_feature.unsqueeze(0).expand(batch_size, -1, -1)
        return anchor, feat


# ===========================================================================
# 可变形注意力特征聚合（简化版，可替换为完整的 DeformableFeatureAggregation）
# ===========================================================================

class SimpleDeformableAggregation(nn.Module):
    """轻量级多视角特征聚合模块（可变形采样的简化实现）。

    将 N 个相机特征按投影坐标双线性采样后加权求和，更新实例特征。

    Args:
        embed_dims:  特征维度
        num_cams:    相机数量
        num_levels:  特征层数
    """

    def __init__(
        self,
        embed_dims: int,
        num_cams:   int,
        num_levels: int,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams   = num_cams
        self.num_levels = num_levels

        # 注意力权重（相机 × 层数）
        self.attention_weights = nn.Linear(embed_dims, num_cams * num_levels)

        # 各层通道到 embed_dims 的投影
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        instance_feature: torch.Tensor,   # [B, N, C]
        anchor:           torch.Tensor,   # [B, N, D]  D >= 3
        ms_img_feats:     List[torch.Tensor],  # each [B, num_cams, C, H_i, W_i]
        metas:            dict,
    ) -> torch.Tensor:                    # [B, N, C]
        """简化的特征聚合：将 Gaussian 中心投影到各相机，采样特征并加权。"""
        B, num_anchor, _ = instance_feature.shape
        device = instance_feature.device

        # 注意力权重 [B, N, num_cams * num_levels]
        attn_raw = self.attention_weights(instance_feature)
        attn_raw = attn_raw.view(B, num_anchor, self.num_cams, self.num_levels)
        attn = attn_raw.softmax(dim=-1)               # [B, N, num_cams, num_levels]

        # 提取 Gaussian 中心（前 3 维）— 注意：此时 anchor 的 xyz 尚未解码
        xyz_norm = anchor[..., :3].sigmoid()           # [B, N, 3]  归一化 xyz
        pc_range = metas.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        pr = anchor.new_tensor(pc_range)
        xyz_world = torch.stack([
            xyz_norm[..., 0] * (pr[3] - pr[0]) + pr[0],
            xyz_norm[..., 1] * (pr[4] - pr[1]) + pr[1],
            xyz_norm[..., 2] * (pr[5] - pr[2]) + pr[2],
        ], dim=-1)                                     # [B, N, 3]

        # 投影矩阵：projection [B, num_cams, 4, 4]
        projection = metas.get('projection')           # [B, num_cams, 4, 4]
        if projection is None:
            # 无相机投影矩阵时，直接跳过特征聚合（全零增量）
            return instance_feature

        if not isinstance(projection, torch.Tensor):
            projection = torch.stack(projection, dim=0)  # handle list-of-tensors

        # 齐次坐标 [B, N, 4]
        ones  = torch.ones(B, num_anchor, 1, device=device, dtype=xyz_world.dtype)
        xyz_h = torch.cat([xyz_world, ones], dim=-1)   # [B, N, 4]

        # 投影到各相机图像坐标 [B, num_cams, N, 4]
        # projection: [B, num_cams, 4, 4] @ xyz_h: [B, 1, N, 4, 1]
        proj = projection.to(device)                   # [B, num_cams, 4, 4]
        xyz_cam = torch.einsum(
            'bcij,bnj->bcni', proj, xyz_h
        )                                              # [B, num_cams, N, 4]

        depth = xyz_cam[..., 2:3].clamp(min=1e-4)
        uv    = xyz_cam[..., :2] / depth               # [B, num_cams, N, 2]

        # 归一化坐标 ∈ [-1, 1]（供 grid_sample 使用）
        H_list = [f.shape[-2] for f in ms_img_feats]
        W_list = [f.shape[-1] for f in ms_img_feats]

        sampled_feats = []
        for lvl, feat in enumerate(ms_img_feats):
            # feat: [B, num_cams, C, H_lvl, W_lvl]
            H_lvl, W_lvl = H_list[lvl], W_list[lvl]
            # 归一化 uv: [B, num_cams, N, 2]
            uv_norm = torch.stack([
                uv[..., 0] / (W_lvl - 1) * 2 - 1,
                uv[..., 1] / (H_lvl - 1) * 2 - 1,
            ], dim=-1)                                 # [B, num_cams, N, 2]

            # grid_sample 需要 [B*num_cams, C, H, W] 和 grid [B*num_cams, N, 1, 2]
            BN = B * self.num_cams
            feat_flat  = feat.view(BN, feat.shape[2], H_lvl, W_lvl)
            grid_flat  = uv_norm.view(BN, num_anchor, 1, 2)

            sampled = F.grid_sample(
                feat_flat, grid_flat,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )                                          # [B*num_cams, C, N, 1]
            sampled = sampled.squeeze(-1)              # [B*num_cams, C, N]
            sampled = sampled.view(B, self.num_cams, feat.shape[2], num_anchor)
            sampled = sampled.permute(0, 3, 1, 2)      # [B, N, num_cams, C]
            sampled_feats.append(sampled)

        # 按层堆叠 [B, N, num_cams, num_levels, C]
        sampled_stacked = torch.stack(sampled_feats, dim=3)  # [B, N, num_cams, L, C]

        # 用注意力权重聚合
        # attn: [B, N, num_cams, L] — 需插入最后一维与 C 对齐
        out = (sampled_stacked * attn.unsqueeze(-1)).sum(dim=(2, 3))  # [B, N, C]
        out = self.value_proj(out)
        out = self.output_proj(out + instance_feature)
        return out


# ===========================================================================
# Gaussian 回归头：MLP 将实例特征映射为 Gaussian 参数增量
# ===========================================================================

class GaussianRefinementHead(nn.Module):
    """将实例特征 + 锚点嵌入映射为 Gaussian 参数（绝对值或增量）。

    输出张量维度：
        0:3   → xyz  (sigmoid 激活后映射到 pc_range)
        3:6   → scale (sigmoid 激活后映射到 scale_range)
        6:10  → quaternion (L2 归一化)
        10:11 → opacity (sigmoid)
        11:   → semantic_features (无激活，raw logits)

    Args:
        embed_dims:  输入特征维度
        num_classes: 语义类别数 C
        pc_range:    [x_min, y_min, z_min, x_max, y_max, z_max]
        scale_range: [scale_min, scale_max]
    """

    def __init__(
        self,
        embed_dims:  int,
        num_classes: int,
        pc_range:    List[float],
        scale_range: List[float],
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pc_range    = pc_range
        self.scale_range = scale_range

        # 输出维度：xyz(3) + scale(3) + quat(4) + opacity(1) + semantics(C)
        out_dim = 3 + 3 + 4 + 1 + num_classes

        self.layers = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, out_dim),
        )

        # 锚点嵌入（将 D 维锚点投影到 embed_dims）
        D_anchor = 3 + 3 + 4 + 1 + num_classes
        self.anchor_embed = nn.Sequential(
            nn.Linear(D_anchor, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        instance_feature: torch.Tensor,   # [B, N, embed_dims]
        anchor:           torch.Tensor,   # [B, N, D]
    ) -> Tuple[torch.Tensor, Gaussians]:
        """
        Returns:
            anchor_updated: [B, N, D]    精化后的锚点向量（用于下一层）
            gaussians:      Gaussians    解码后的 Gaussian 参数
        """
        B, N, _ = instance_feature.shape
        pr = instance_feature.new_tensor(self.pc_range)   # [6]
        sr = instance_feature.new_tensor(self.scale_range)  # [2]

        # 锚点嵌入
        anc_emb = self.anchor_embed(anchor)                # [B, N, embed_dims]
        feat_in = torch.cat([instance_feature, anc_emb], dim=-1)  # [B, N, 2*embed_dims]
        delta   = self.layers(feat_in)                     # [B, N, out_dim]

        # ── 解码 xyz ────────────────────────────────────────────────
        # anchor 的前 3 维 + delta 前 3 维（残差），sigmoid → 归一化 → 解码
        xyz_raw = anchor[..., :3] + delta[..., :3]
        xyz_norm = xyz_raw.sigmoid()                       # [B, N, 3]  ∈ [0, 1]
        means = torch.stack([
            xyz_norm[..., 0] * (pr[3] - pr[0]) + pr[0],
            xyz_norm[..., 1] * (pr[4] - pr[1]) + pr[1],
            xyz_norm[..., 2] * (pr[5] - pr[2]) + pr[2],
        ], dim=-1)                                         # [B, N, 3]  ego 坐标

        # ── 解码 scale ───────────────────────────────────────────────
        scale_raw = anchor[..., 3:6] + delta[..., 3:6]
        scale_norm = scale_raw.sigmoid()                   # [B, N, 3]
        scales = sr[0] + (sr[1] - sr[0]) * scale_norm     # [B, N, 3]

        # ── 解码 rotation（四元数，L2 归一化） ────────────────────────
        rot_raw = anchor[..., 6:10] + delta[..., 6:10]
        rotations = F.normalize(rot_raw, dim=-1)           # [B, N, 4]

        # ── 解码 opacity ──────────────────────────────────────────────
        opa_raw   = anchor[..., 10:11] + delta[..., 10:11]
        opacities = opa_raw.sigmoid()                      # [B, N, 1]

        # ── 语义特征（无激活，raw logits） ─────────────────────────────
        sem_raw   = anchor[..., 11:] + delta[..., 11:]
        semantics = sem_raw                                # [B, N, C]  无任何激活

        # ── 更新锚点（以更新后的原始向量为下一层的 anchor） ────────────
        anchor_updated = torch.cat(
            [xyz_raw, scale_raw, rot_raw, opa_raw, sem_raw], dim=-1
        )  # [B, N, D]

        gaussians = Gaussians(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            semantics=semantics,
        )
        return anchor_updated, gaussians


# ===========================================================================
# Gaussian 渲染器：Expected Depth + Semantic Alpha-Blending
# ===========================================================================

def _try_import_gsplat():
    """尝试导入 gsplat，返回 rasterization 函数或 None。"""
    try:
        from gsplat import rasterization  # type: ignore[import]
        return rasterization
    except ImportError:
        return None


_GSPLAT_RASTERIZATION = _try_import_gsplat()


def render(
    gaussians: Gaussians,
    camera: Camera,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 3D Gaussians 渲染到目标相机视角。

    渲染方式（按优先级）：
      1. gsplat.rasterization —— 高效 CUDA 实现（推荐）
      2. PyTorch EWA 实现     —— 纯 Python，仅调试用

    Args:
        gaussians: Gaussians  3D Gaussian 参数 [B, N, *]
        camera:    Camera     目标相机参数

    Returns:
        depth_map:    [B, 1, H, W]  渲染深度图（Expected Depth，米）
        semantic_map: [B, C, H, W]  渲染语义特征图（raw logits，无激活）
        alpha_map:    [B, 1, H, W]  累积不透明度
    """
    if _GSPLAT_RASTERIZATION is not None:
        return _render_gsplat(gaussians, camera)
    else:
        return _render_pytorch_ewa(gaussians, camera)


# ---------------------------------------------------------------------------
# 渲染后端 1：gsplat
# ---------------------------------------------------------------------------

def _render_gsplat(
    gaussians: Gaussians,
    camera:    Camera,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用 gsplat.rasterization 渲染深度与语义特征。

    实现思路：
      - 将语义特征 [N, C] 与 z_c [N, 1] 拼接为 (C+1) 维"颜色"
      - gsplat 一次性完成 alpha-blending，最后一维即深度

    gsplat API 参考：
        https://docs.gsplat.studio/main/apis/rasterization.html
    """
    from gsplat import rasterization  # type: ignore[import]

    B = gaussians.means.shape[0]
    C = gaussians.semantics.shape[-1]
    H, W = camera.height, camera.width
    device = gaussians.means.device

    depth_list, sem_list, alpha_list = [], [], []

    for b in range(B):
        means_b  = gaussians.means[b]                            # [N, 3]
        quats_b  = F.normalize(gaussians.rotations[b], dim=-1)  # [N, 4]  gsplat 期望 (x,y,z,w)
        # gsplat 的四元数约定是 (x, y, z, w)，我们存储的是 (w, x, y, z)，需转换
        quats_b_gsplat = torch.cat([quats_b[:, 1:], quats_b[:, :1]], dim=-1)  # [N, 4] (x,y,z,w)
        scales_b = gaussians.scales[b]                           # [N, 3]
        opas_b   = gaussians.opacities[b, :, 0]                 # [N]

        # 计算相机坐标系下的 z_c（用于深度渲染）
        ego2cam = camera.ego2cam[b]                              # [4, 4]
        ones    = torch.ones(means_b.shape[0], 1, device=device)
        means_h = torch.cat([means_b, ones], dim=-1)            # [N, 4]
        means_cam = (ego2cam @ means_h.T).T[:, :3]              # [N, 3]
        z_c = means_cam[:, 2:3].clamp(camera.near, camera.far)  # [N, 1]

        # 拼接语义 + 深度 → "颜色" [N, C+1]
        colors = torch.cat([gaussians.semantics[b], z_c], dim=-1)  # [N, C+1]

        K3 = camera.K[b, :3, :3]                                 # [3, 3]

        renders, alpha, _ = rasterization(
            means=means_b,          # [N, 3]
            quats=quats_b_gsplat,   # [N, 4]  (x,y,z,w)
            scales=scales_b,        # [N, 3]
            opacities=opas_b,       # [N]
            colors=colors,          # [N, C+1]
            viewmats=ego2cam.unsqueeze(0),  # [1, 4, 4]  ego→cam
            Ks=K3.unsqueeze(0),             # [1, 3, 3]
            width=W,
            height=H,
            near_plane=camera.near,
            far_plane=camera.far,
            packed=False,
        )
        # renders: [1, H, W, C+1]  alpha: [1, H, W, 1]
        renders = renders[0].permute(2, 0, 1)   # [C+1, H, W]
        alpha_out = alpha[0].permute(2, 0, 1)   # [1, H, W]

        sem_list.append(renders[:C])             # [C, H, W]  语义 logits
        depth_list.append(renders[C:C+1])        # [1, H, W]  深度
        alpha_list.append(alpha_out)             # [1, H, W]

    # [B, C, H, W] 和 [B, 1, H, W]
    return (
        torch.stack(depth_list),    # [B, 1, H, W]
        torch.stack(sem_list),      # [B, C, H, W]
        torch.stack(alpha_list),    # [B, 1, H, W]
    )


# ---------------------------------------------------------------------------
# 渲染后端 2：PyTorch EWA（慢，仅调试）
# ---------------------------------------------------------------------------

def _render_pytorch_ewa(
    gaussians: Gaussians,
    camera:    Camera,
    max_radius_px: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """纯 PyTorch EWA Gaussian 光栅化（仅供调试，不推荐生产使用）。

    使用透视投影的 Jacobian 将 3D 协方差投影到 2D 屏幕空间，
    再对每像素执行 alpha-blending。

    性能警告：O(N × H × W) 的 Python 循环，仅适合 N < 1000、图像 < 128×128。
    """
    B, N, C = gaussians.semantics.shape
    H, W    = camera.height, camera.width
    device  = gaussians.means.device

    depth_list, sem_list, alpha_list = [], [], []
    for b in range(B):
        ego2cam = camera.ego2cam[b]         # [4, 4]
        K3      = camera.K[b, :3, :3]       # [3, 3]
        fx, fy  = K3[0, 0].item(), K3[1, 1].item()
        cx, cy  = K3[0, 2].item(), K3[1, 2].item()

        # ── 转换到相机坐标 ───────────────────────────────────────────
        ones      = torch.ones(N, 1, device=device)
        means_h   = torch.cat([gaussians.means[b], ones], dim=-1)  # [N, 4]
        means_cam = (ego2cam @ means_h.T).T[:, :3]                 # [N, 3]
        z         = means_cam[:, 2]                                # [N]

        # ── 2D 投影 ──────────────────────────────────────────────────
        z_safe = z.clamp(min=1e-4)
        u_proj = means_cam[:, 0] / z_safe * fx + cx   # [N]
        v_proj = means_cam[:, 1] / z_safe * fy + cy   # [N]

        visible = (
            (z > camera.near) & (z < camera.far)
            & (u_proj >= 0) & (u_proj < W)
            & (v_proj >= 0) & (v_proj < H)
        )

        depth_img = torch.zeros(1, H, W, device=device)
        sem_img   = torch.zeros(C, H, W, device=device)
        T_map     = torch.ones(H, W, device=device)
        alpha_acc = torch.zeros(1, H, W, device=device)

        if not visible.any():
            depth_list.append(depth_img)
            sem_list.append(sem_img)
            alpha_list.append(alpha_acc)
            continue

        idx   = visible.nonzero(as_tuple=True)[0]
        mc    = means_cam[idx]                         # [M, 3]
        sc    = gaussians.scales[b][idx]               # [M, 3]
        rot   = gaussians.rotations[b][idx]            # [M, 4]
        op    = gaussians.opacities[b][idx, 0]         # [M]
        fe    = gaussians.semantics[b][idx]            # [M, C]
        z_v   = z[idx]                                 # [M]
        u_v   = u_proj[idx]                            # [M]
        v_v   = v_proj[idx]                            # [M]

        # 按深度从近到远排序
        sort_idx = z_v.argsort()
        mc, sc, rot, op, fe, z_v, u_v, v_v = (
            mc[sort_idx], sc[sort_idx], rot[sort_idx], op[sort_idx],
            fe[sort_idx], z_v[sort_idx], u_v[sort_idx], v_v[sort_idx],
        )

        # EWA 2D 协方差（每次处理一个 Gaussian）
        W_cam = ego2cam[:3, :3]                        # [3, 3]
        M_count = mc.shape[0]

        for n in range(M_count):
            u0, v0 = u_v[n].item(), v_v[n].item()
            z_n = z_v[n].item()

            # 3D 协方差 → 2D（EWA 投影）
            R = _quat_to_rotmat_single(rot[n])         # [3, 3]
            S_diag = torch.diag(sc[n])                 # [3, 3]
            RS = R @ S_diag                            # [3, 3]
            Sigma3d = RS @ RS.T                        # [3, 3]

            # Jacobian（透视投影）
            J = torch.zeros(2, 3, device=device)
            J[0, 0] = fx / z_n
            J[0, 2] = -fx * mc[n, 0].item() / (z_n ** 2)
            J[1, 1] = fy / z_n
            J[1, 2] = -fy * mc[n, 1].item() / (z_n ** 2)

            Sigma_cam = W_cam @ Sigma3d @ W_cam.T
            cov2d = J @ Sigma_cam @ J.T                # [2, 2]
            cov2d[0, 0] = cov2d[0, 0].clamp(min=0.3)
            cov2d[1, 1] = cov2d[1, 1].clamp(min=0.3)

            det = (cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] ** 2).clamp(min=1e-6)
            cov_inv = torch.stack([
                cov2d[1, 1] / det, -cov2d[0, 1] / det,
                -cov2d[1, 0] / det, cov2d[0, 0] / det,
            ]).view(2, 2)

            # 影响半径
            trace = cov2d[0, 0] + cov2d[1, 1]
            disc  = ((cov2d[0, 0] - cov2d[1, 1]).pow(2) / 4 + cov2d[0, 1].pow(2)).clamp(0)
            lam_max = trace / 2 + disc.sqrt()
            r = min(int((3 * lam_max.sqrt()).ceil().item()), max_radius_px)

            u_lo = max(0, int(u0) - r);  u_hi = min(W, int(u0) + r + 1)
            v_lo = max(0, int(v0) - r);  v_hi = min(H, int(v0) + r + 1)
            if u_lo >= u_hi or v_lo >= v_hi:
                continue

            us = torch.arange(u_lo, u_hi, device=device, dtype=torch.float32) - u0
            vs = torch.arange(v_lo, v_hi, device=device, dtype=torch.float32) - v0
            vv, uu = torch.meshgrid(vs, us, indexing='ij')           # [pH, pW]
            d = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # [pH*pW, 2]
            mahal = (d @ cov_inv * d).sum(-1).view(vv.shape)          # [pH, pW]
            w = torch.exp(-0.5 * mahal)
            alpha_n = (op[n] * w).clamp(max=0.9999)

            T_patch = T_map[v_lo:v_hi, u_lo:u_hi]
            contrib  = T_patch * alpha_n

            depth_img[0, v_lo:v_hi, u_lo:u_hi] += contrib * z_n
            sem_img[:, v_lo:v_hi, u_lo:u_hi]   += contrib[None] * fe[n, :, None, None]
            alpha_acc[0, v_lo:v_hi, u_lo:u_hi] += contrib
            T_map[v_lo:v_hi, u_lo:u_hi]         = T_patch * (1 - alpha_n)

        depth_list.append(depth_img)
        sem_list.append(sem_img)
        alpha_list.append(alpha_acc)

    return (
        torch.stack(depth_list),   # [B, 1, H, W]
        torch.stack(sem_list),     # [B, C, H, W]
        torch.stack(alpha_list),   # [B, 1, H, W]
    )


def _quat_to_rotmat_single(q: torch.Tensor) -> torch.Tensor:
    """四元数 [w, x, y, z] → 3×3 旋转矩阵（单个 Gaussian）。"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(3, 3)


# ===========================================================================
# 完整的 DLWM 模型（端到端）
# ===========================================================================

class DLWMModel(nn.Module):
    """DLWM 一阶段场景重建模型。

    Args:
        num_classes:    语义类别数 C（含背景 0）
        num_anchor:     初始 Gaussian 数量
        embed_dims:     实例特征维度
        num_decoder:    解码器（精化）层数
        pc_range:       [x_min, y_min, z_min, x_max, y_max, z_max]
        scale_range:    [scale_min, scale_max]
        num_cams:       相机数量
        fpn_out_ch:     FPN 输出通道数
        num_levels:     FPN 层数
        pretrained:     是否加载图像主干预训练权重
        img_backbone_out_indices: backbone 输出层索引
        img_backbone_config:      覆盖默认 backbone 配置
        img_neck_config:          覆盖默认 neck 配置
        img_pretrained_path:      backbone/neck 预训练权重路径
    """

    def __init__(
        self,
        num_classes:  int = 16,
        num_anchor:   int = 6400,
        embed_dims:   int = 128,
        num_decoder:  int = 6,
        pc_range:     List[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        scale_range:  List[float] = (0.05, 5.0),
        num_cams:     int = 7,
        fpn_out_ch:   int = 128,
        num_levels:   int = 4,
        pretrained:   bool = True,
        img_backbone_out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        img_backbone_config: Optional[Dict] = None,
        img_neck_config: Optional[Dict] = None,
        img_pretrained_path: str = "ckpts/r101_dcn_fcos3d_pretrain.pth",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_decoder = num_decoder
        self.num_cams    = num_cams
        self.embed_dims  = embed_dims

        # ── 图像特征提取 ──────────────────────────────────────────────
        self.feature_extractor = ImageFeatureExtractor(
            out_channels=fpn_out_ch,
            num_levels=num_levels,
            pretrained=pretrained,
            img_backbone_out_indices=img_backbone_out_indices,
            img_backbone_config=img_backbone_config,
            img_neck_config=img_neck_config,
            pretrained_path=img_pretrained_path,
        )

        # ── Gaussian 初始化 ────────────────────────────────────────────
        self.lifter = GaussianLifter(
            num_anchor=num_anchor,
            embed_dims=embed_dims,
            num_classes=num_classes,
            pc_range=list(pc_range),
            scale_range=list(scale_range),
        )

        # ── 特征聚合 + 精化头（num_decoder 层共用参数） ────────────────
        self.aggregations = nn.ModuleList([
            SimpleDeformableAggregation(
                embed_dims=embed_dims,
                num_cams=num_cams,
                num_levels=num_levels,
            )
            for _ in range(num_decoder)
        ])
        self.refine_heads = nn.ModuleList([
            GaussianRefinementHead(
                embed_dims=embed_dims,
                num_classes=num_classes,
                pc_range=list(pc_range),
                scale_range=list(scale_range),
            )
            for _ in range(num_decoder)
        ])

        # FPN 通道数对齐到 embed_dims（若不同）
        if fpn_out_ch != embed_dims:
            self.fpn_proj = nn.Conv2d(fpn_out_ch, embed_dims, 1)
        else:
            self.fpn_proj = None

    def init_weights(self) -> None:
        self.lifter.init_weights()
        for head in self.refine_heads:
            head._init_weights()

    def forward(
        self,
        imgs:     torch.Tensor,  # [B, N_cams, 3, H, W]
        metas:    dict,
    ) -> Dict[str, object]:
        """端到端前向。

        Args:
            imgs:  [B, N_cams, 3, H, W]  多视角归一化图像
            metas: 包含 'projection' [B, N_cams, 4, 4] 等相机参数

        Returns:
            dict:
                'all_gaussians'  : List[Gaussians]  各解码层的 Gaussian 参数
                'last_gaussians' : Gaussians         最后一层的 Gaussian 参数
                'ms_img_feats'   : List[Tensor]      多尺度图像特征
        """
        B, N_cams, _, H, W = imgs.shape

        # ── 1. 图像特征提取 ──────────────────────────────────────────
        imgs_flat = imgs.view(B * N_cams, 3, H, W)     # [B*N, 3, H, W]
        ms_feats_flat: List[torch.Tensor] = self.feature_extractor(imgs_flat)
        # 恢复为 [B, N_cams, C, H_i, W_i]
        ms_img_feats: List[torch.Tensor] = []
        for feat in ms_feats_flat:
            BN, C_f, H_f, W_f = feat.shape
            if self.fpn_proj is not None:
                feat = self.fpn_proj(feat)
                C_f = feat.shape[1]
            ms_img_feats.append(feat.view(B, N_cams, C_f, H_f, W_f))

        # ── 2. Gaussian 初始化 ────────────────────────────────────────
        anchor, instance_feature = self.lifter(B)
        # anchor:           [B, num_anchor, D]
        # instance_feature: [B, num_anchor, embed_dims]

        # ── 3. 迭代精化（num_decoder 层） ────────────────────────────
        all_gaussians: List[Gaussians] = []
        for i in range(self.num_decoder):
            # 3a. 多视角特征聚合
            instance_feature = self.aggregations[i](
                instance_feature, anchor, ms_img_feats, metas
            )  # [B, num_anchor, embed_dims]

            # 3b. 精化 Gaussian 参数
            anchor, gaussians = self.refine_heads[i](instance_feature, anchor)
            all_gaussians.append(gaussians)

        return {
            'all_gaussians':  all_gaussians,
            'last_gaussians': all_gaussians[-1],
            'ms_img_feats':   ms_img_feats,
        }
