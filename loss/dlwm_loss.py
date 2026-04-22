"""
DLWMLoss: DLWM 一阶段范式渲染监督损失函数。

总损失公式：
    L_rec = w_d * L_d + w_pd * L_pd + w_sem * L_sem

- L_d  (稀疏深度损失):   L1 Loss，仅在 valid_lidar_mask 有效像素上计算，权重 1.0
- L_pd (稠密伪深度损失):  L1 Loss，全像素计算，权重 0.05
- L_sem (语义交叉熵损失): Cross-Entropy Loss，权重 1.0，ignore_index=empty_label

参考：DLWM 一阶段范式 Loss 设计
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class DLWMLoss(nn.Module):
    """DLWM 渲染监督总损失模块。

    Args:
        weight_sparse_depth (float): L_d 权重，默认 1.0
        weight_dense_depth  (float): L_pd 权重，默认 0.05
        weight_semantic     (float): L_sem 权重，默认 1.0
        num_classes         (int):   语义类别总数（含 empty_label），默认 16
        empty_label         (int):   空/忽略类别 id，默认 0
        train_classes       (list|None): 仅计算 Loss 的类别子集；
                                        None 表示全部类别均计入 Loss。
                                        例：[6] 仅训练地面类。
        input_dict          (dict):  从 loss_func 输入 dict 到内部字段的映射，
                                    兼容 BaseLoss 风格的 MultiLoss 调用链。
    """

    def __init__(
        self,
        weight_sparse_depth: float = 1.0,
        weight_dense_depth: float = 0.05,
        weight_semantic: float = 1.0,
        num_classes: int = 16,
        empty_label: int = 0,
        train_classes: Optional[List[int]] = None,
        input_dict: Optional[dict] = None,
    ):
        super().__init__()
        self.w_sparse = weight_sparse_depth
        self.w_dense = weight_dense_depth
        self.w_sem = weight_semantic
        self.num_classes = num_classes
        self.empty_label = empty_label
        self.train_classes = train_classes

        # 默认 input_dict：直接从 inputs 字典按相同 key 取值
        if input_dict is None:
            input_dict = {
                'depth_pred':       'depth_pred',
                'sparse_depth_gt':  'sparse_depth_gt',
                'valid_lidar_mask': 'valid_lidar_mask',
                'dense_depth_gt':   'dense_depth_gt',
                'semantic_pred':    'semantic_pred',
                'semantic_gt':      'semantic_gt',
            }
        self.input_dict = input_dict

        # Cross-Entropy Loss，ignore_index 对应 empty_label（不参与梯度）
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=empty_label,
            reduction='mean',
        )

    # ------------------------------------------------------------------
    # 内部计算方法
    # ------------------------------------------------------------------

    def _sparse_depth_loss(
        self,
        depth_pred: torch.Tensor,       # [B, M, 1, H, W]
        sparse_depth_gt: torch.Tensor,  # [B, M, 1, H, W]
        valid_lidar_mask: torch.Tensor, # [B, M, 1, H, W]  bool
    ) -> torch.Tensor:
        """L_d：稀疏深度 L1 Loss，仅在有效 LiDAR 点处计算。"""
        mask = valid_lidar_mask.bool()
        if mask.any():
            return F.l1_loss(
                depth_pred[mask],
                sparse_depth_gt[mask],
                reduction='mean',
            )
        # 无有效点时返回零标量，保持计算图完整
        return depth_pred.sum() * 0.0

    def _dense_depth_loss(
        self,
        depth_pred: torch.Tensor,      # [B, M, 1, H, W]
        dense_depth_gt: torch.Tensor,  # [B, M, 1, H, W]
    ) -> torch.Tensor:
        """L_pd：稠密伪深度 L1 Loss，全像素计算。"""
        return F.l1_loss(depth_pred, dense_depth_gt, reduction='mean')

    def _semantic_loss(
        self,
        semantic_pred: torch.Tensor,  # [B, M, C, H, W]  logits（Softmax 前）
        semantic_gt: torch.Tensor,    # [B, M, H, W]      int64 类别 id
    ) -> torch.Tensor:
        """L_sem：语义 Cross-Entropy Loss，支持类别过滤。"""
        B, M, C, H, W = semantic_pred.shape
        pred_flat = semantic_pred.view(B * M, C, H, W)   # [B*M, C, H, W]
        gt_flat = semantic_gt.view(B * M, H, W).long()   # [B*M, H, W]

        # 类别过滤：只保留 train_classes 中的像素参与 Loss
        if self.train_classes is not None:
            class_mask = torch.zeros_like(gt_flat, dtype=torch.bool)
            for cls_id in self.train_classes:
                class_mask |= (gt_flat == cls_id)
            # 非目标类别的像素 → 设为 ignore_index，不参与梯度
            gt_flat = gt_flat.clone()
            gt_flat[~class_mask] = self.empty_label

        return self.ce_loss(pred_flat, gt_flat)

    # ------------------------------------------------------------------
    # 前向接口
    # ------------------------------------------------------------------

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, dict]:
        """计算总 Loss 并返回各分项。

        Args:
            inputs (dict): 包含以下字段（key 由 self.input_dict 映射）：
                depth_pred       [B, M, 1, H, W]  float  渲染深度（米）
                sparse_depth_gt  [B, M, 1, H, W]  float  稀疏 LiDAR 深度 GT
                valid_lidar_mask [B, M, 1, H, W]  bool   有效 LiDAR 像素 mask
                dense_depth_gt   [B, M, 1, H, W]  float  稠密伪深度 GT
                semantic_pred    [B, M, C, H, W]  float  语义 logits
                semantic_gt      [B, M, H, W]     int64  语义标签

        Returns:
            total_loss (Tensor): 标量总损失
            loss_dict  (dict):   各分项损失值（detach 后的 float）
        """
        # 从 inputs 中提取各字段（支持 input_dict 重映射）
        depth_pred = inputs[self.input_dict['depth_pred']]
        sparse_depth_gt = inputs[self.input_dict['sparse_depth_gt']]
        valid_lidar_mask = inputs[self.input_dict['valid_lidar_mask']]
        dense_depth_gt = inputs[self.input_dict['dense_depth_gt']]
        semantic_pred = inputs[self.input_dict['semantic_pred']]
        semantic_gt = inputs[self.input_dict['semantic_gt']]

        l_d = self._sparse_depth_loss(depth_pred, sparse_depth_gt, valid_lidar_mask)
        l_pd = self._dense_depth_loss(depth_pred, dense_depth_gt)
        l_sem = self._semantic_loss(semantic_pred, semantic_gt)

        total = self.w_sparse * l_d + self.w_dense * l_pd + self.w_sem * l_sem

        loss_dict = {
            'L_d':   l_d.detach().item(),
            'L_pd':  l_pd.detach().item(),
            'L_sem': l_sem.detach().item(),
        }
        return total, loss_dict
