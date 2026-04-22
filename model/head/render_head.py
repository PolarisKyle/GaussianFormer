"""
GaussianRenderHead: 基于 Gaussian Splatting 的深度与语义渲染头。

职责：
  - 接收 GaussianOccEncoder 输出的 Gaussian 表示
  - 构建 GaussianParams 并调用 SemanticDepthRasterizer
  - 支持多解码器层的渲染（all / random / fixed）
  - 输出渲染深度图和语义特征图（语义 logits，Softmax 前）

注意：
  - 与 GaussianHead（体素占据预测）相互独立，可以并行使用或单独使用
  - 语义特征来自 GaussianPrediction.semantics（需使用 semantics_activation='none'）
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS

from .base_head import BaseTaskHead
from ..renderer.semantic_depth_rasterizer import GaussianParams, SemanticDepthRasterizer


@MODELS.register_module()
class GaussianRenderHead(BaseTaskHead):
    """Gaussian Splatting 渲染头。

    从 GaussianOccEncoder 的输出中提取 Gaussian 参数，
    调用 SemanticDepthRasterizer 渲染深度图和语义特征图。

    Args:
        rasterizer_cfg (dict):   SemanticDepthRasterizer 的配置字典
        num_classes (int):       语义类别数（含 empty_label），默认 16
        apply_loss_type (str):   渲染哪些解码器层：
                                   'all'         → 所有层
                                   'random_K'    → 随机 K 层（包含最后一层）
                                   'fixed_i_j_k' → 固定层索引
        init_cfg:                权重初始化配置
    """

    def __init__(
        self,
        rasterizer_cfg: dict,
        num_classes: int = 16,
        apply_loss_type: str = 'all',
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_classes = num_classes

        # 解析 apply_loss_type
        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [
                int(x) for x in apply_loss_type.split('_')[1:]
            ]
        else:
            raise ValueError(
                f"不支持的 apply_loss_type: {apply_loss_type}，"
                "有效值：'all', 'random_K', 'fixed_i_j_...'"
            )

        # 构建渲染器
        self.rasterizer = MODELS.build(rasterizer_cfg)

        # 用于生成零标量梯度的 buffer
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'init_weight'):
                m.init_weight()

    def _get_apply_loss_layers(self, num_decoder: int) -> List[int]:
        """根据策略确定需要计算 Loss 的解码器层索引。"""
        if not self.training:
            return [num_decoder - 1]

        if self.apply_loss_type == 'all':
            return list(range(num_decoder))
        elif self.apply_loss_type == 'random':
            if self.random_apply_loss_layers > 1:
                chosen = np.random.choice(
                    num_decoder - 1,
                    self.random_apply_loss_layers - 1,
                    replace=False,
                ).tolist()
                return chosen + [num_decoder - 1]
            return [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            return self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

    def _build_gaussian_params(self, gaussian_bundle) -> GaussianParams:
        """从 GaussianPrediction NamedTuple 构建 GaussianParams。

        注意：gaussian_bundle.semantics 应为原始 logits（无激活），
        需在 refine_module 中设置 semantics_activation='none'。
        """
        return GaussianParams(
            means=gaussian_bundle.means,
            scales=gaussian_bundle.scales,
            rotations=F.normalize(gaussian_bundle.rotations, dim=-1),
            opacities=gaussian_bundle.opacities,
            semantic_features=gaussian_bundle.semantics,  # raw logits
        )

    def forward(
        self,
        representation,          # List[{'gaussian': GaussianPrediction}]
        metas: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """前向渲染流程。

        Args:
            representation: encoder 输出的多层 Gaussian 表示列表
            metas:          batch 数据字典，需包含：
                            - 'intrinsics'          [B, N, 4, 4]  输入相机内参
                            - 'cam2ego'             [B, N, 4, 4]  输入相机外参
                            - 'target_intrinsics'   [B, M, 4, 4]  目标视角内参（可选）
                            - 'target_cam2ego'      [B, M, 4, 4]  目标视角外参（可选）
                            - 'image_size'          (H, W)         渲染分辨率

        Returns:
            dict:
                'depth_pred':    [B, M, 1, H, W]  渲染深度图（米）
                'semantic_pred': [B, M, C, H, W]  语义 logits（Softmax 前）
                'alpha_map':     [B, M, 1, H, W]  累积不透明度
                'all_renders':   list             各解码器层渲染结果（调试用）
        """
        num_decoder = len(representation)
        apply_loss_layers = self._get_apply_loss_layers(num_decoder)

        # 从 metas 中获取目标视角（若未提供则默认使用输入视角）
        target_intrinsics = metas.get('target_intrinsics', metas['intrinsics'])
        target_cam2ego = metas.get('target_cam2ego', metas['cam2ego'])
        image_size = metas.get('image_size')

        # metas 中的 tensor 可能未移动到 GPU，统一确保设备一致
        device = self.zero_tensor.device
        if isinstance(target_intrinsics, torch.Tensor):
            target_intrinsics = target_intrinsics.to(device)
        if isinstance(target_cam2ego, torch.Tensor):
            target_cam2ego = target_cam2ego.to(device)

        all_renders = []
        for idx in apply_loss_layers:
            gaussian_bundle = representation[idx]['gaussian']
            gaussian_params = self._build_gaussian_params(gaussian_bundle)

            render_out = self.rasterizer(
                gaussians=gaussian_params,
                target_intrinsics=target_intrinsics,
                target_cam2ego=target_cam2ego,
                image_size=image_size,
            )
            all_renders.append(render_out)

        # 以最后一个解码器层的渲染结果作为主输出
        final_render = all_renders[-1]

        return {
            'depth_pred':    final_render['depth_map'],    # [B, M, 1, H, W]
            'semantic_pred': final_render['semantic_map'], # [B, M, C, H, W]
            'alpha_map':     final_render['alpha_map'],    # [B, M, 1, H, W]
            'all_renders':   all_renders,
        }
