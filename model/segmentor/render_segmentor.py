"""
RenderSegmentor: 基于 Gaussian Splatting 的深度与语义渲染分割器。

完整 Pipeline：
  7V 图像输入
    → 2D 特征提取（ResNet Backbone + FPN Neck）
    → Gaussian Lifter（生成初始 Gaussian 锚点）
    → GaussianOccEncoder（迭代精化 Gaussians：xyz, scale, rot, opa, semantic[16]）
    → GaussianRenderHead（指定目标相机视角 → 渲染深度图 + 语义特征图）
    → DLWMLoss（L_d + L_pd + L_sem）

关键设计：
  - 语义特征在 3D 空间无激活（raw logits），渲染至 2D 后接 Softmax
  - 支持将任意目标相机视角作为渲染目标（Novel View Synthesis）
  - 默认将输入的 7 个相机视角同时作为渲染目标
"""

import torch

from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class RenderSegmentor(BEVSegmentor):
    """基于 Gaussian Splatting 的渲染分割器。

    继承自 BEVSegmentor，主要差异：
      1. head 替换为 GaussianRenderHead（渲染头），而非 GaussianHead（体素占据头）
      2. forward 流程增加 image_size 传递给渲染头
      3. 支持 forward_train 和 forward_inference 两个接口

    配置示例（config/lt_render_gs.py）：
        model = dict(
            type='RenderSegmentor',
            img_backbone=...,
            img_neck=...,
            lifter=dict(type='GaussianLifter', ...),
            encoder=dict(type='GaussianOccEncoder', ...),
            head=dict(
                type='GaussianRenderHead',
                rasterizer_cfg=dict(
                    type='SemanticDepthRasterizer',
                    num_semantic_classes=16,
                    backend='gsplat',
                ),
                num_classes=16,
                apply_loss_type='all',
            ),
        )
    """

    def forward(
        self,
        imgs=None,
        metas=None,
        **kwargs,
    ) -> dict:
        """前向流程（训练和推理共用）。

        Args:
            imgs:  [B, N, 3, H, W]  多视角图像（N=7）
            metas: dict，批次数据字典，含以下字段：
                   必需：
                     intrinsics  [B, N, 4, 4]  输入相机内参
                     cam2ego     [B, N, 4, 4]  输入相机 cam2ego
                   可选（渲染目标视角，不提供时默认使用输入视角）：
                     target_intrinsics [B, M, 4, 4]
                     target_cam2ego    [B, M, 4, 4]
                   可选（监督信号，训练时需要）：
                     sparse_depth_gt   [B, M, 1, H, W]
                     valid_lidar_mask  [B, M, 1, H, W] bool
                     dense_depth_gt    [B, M, 1, H, W]
                     semantic_gt       [B, M, H, W]    int64

        Returns:
            dict，包含：
                depth_pred    [B, M, 1, H, W]
                semantic_pred [B, M, C, H, W]  (Softmax 前)
                alpha_map     [B, M, 1, H, W]
                ...（其他中间结果）
        """
        # 将图像尺寸注入 metas，供渲染头使用
        if imgs is not None:
            H, W = imgs.shape[-2], imgs.shape[-1]
            if metas is None:
                metas = {}
            metas['image_size'] = (H, W)

        results = {
            'imgs': imgs,
            'metas': metas,
        }
        results.update(kwargs)

        # ── 阶段 1：2D 特征提取 ────────────────────────────────────────
        outs = self.extract_img_feat(**results)
        results.update(outs)

        # ── 阶段 2：Gaussian 初始化（Lifter）──────────────────────────
        outs = self.lifter(**results)
        results.update(outs)

        # ── 阶段 3：GaussianOccEncoder（迭代精化）─────────────────────
        outs = self.encoder(**results)
        results.update(outs)

        # ── 阶段 4：渲染（GaussianRenderHead）────────────────────────
        outs = self.head(
            representation=results['representation'],
            metas=metas,
            **kwargs,
        )
        results.update(outs)

        return results

    @torch.no_grad()
    def forward_inference(
        self,
        imgs: torch.Tensor,
        metas: dict,
        target_intrinsics: torch.Tensor = None,
        target_cam2ego: torch.Tensor = None,
    ) -> dict:
        """推理接口：支持指定任意目标相机视角。

        Args:
            imgs:               [B, N, 3, H, W]  输入图像
            metas:              batch 数据字典
            target_intrinsics:  [B, M, 4, 4]  目标视角内参（None 则使用输入视角）
            target_cam2ego:     [B, M, 4, 4]  目标视角外参

        Returns:
            dict:
                'depth_map':     [B, M, 1, H, W]  渲染深度图（米）
                'semantic_pred': [B, M, H, W]     argmax 预测类别 id
                'semantic_prob': [B, M, C, H, W]  Softmax 概率
                'alpha_map':     [B, M, 1, H, W]  不透明度
        """
        self.eval()

        if target_intrinsics is not None:
            metas['target_intrinsics'] = target_intrinsics
        if target_cam2ego is not None:
            metas['target_cam2ego'] = target_cam2ego

        results = self.forward(imgs=imgs, metas=metas)

        # 语义图后处理：Softmax → argmax
        sem_logits = results['semantic_pred']                     # [B, M, C, H, W]
        sem_prob = sem_logits.softmax(dim=2)                      # [B, M, C, H, W]
        sem_pred = sem_prob.argmax(dim=2)                         # [B, M, H, W]

        return {
            'depth_map':     results['depth_pred'],    # [B, M, 1, H, W]
            'semantic_pred': sem_pred,                 # [B, M, H, W]
            'semantic_prob': sem_prob,                 # [B, M, C, H, W]
            'alpha_map':     results['alpha_map'],     # [B, M, 1, H, W]
        }
