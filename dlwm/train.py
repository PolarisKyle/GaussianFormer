"""
dlwm/train.py
=============
DLWM (arXiv:2604.00969) 一阶段复现 —— 损失函数 & 训练主循环

损失公式：
    L_rec = 1.0 × L_d + 0.05 × L_pd + 1.0 × L_sem

动态类别过滤（train_classes）：
    1. 根据 semantic_gt 生成 class_mask（只有 GT 类别在 train_classes 中时为 True）
    2. L_d  — 仅在 sparse_depth_gt > 0 且 class_mask 为 True 的像素上计算 L1
    3. L_pd — 仅在 dense_depth_gt  > 0 且 class_mask 为 True 的像素上计算 L1
    4. L_sem— CrossEntropyLoss，通过手动 masking 只对 class_mask 内的像素计算梯度

训练流程：
    多视角图像 → DLWMModel → last_gaussians
        → render(gaussians, target_cam)
        → DLWMLoss(depth_pred, semantic_pred, ...)
        → backward + optimizer.step()

使用示例：
    python -m dlwm.train \
        --data_root /path/to/dataset \
        --epochs 24 \
        --batch_size 1 \
        --train_classes 1 2       # 只训练 road(1) 和 vehicle(2)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import DLWMDataset
from .model import Camera, DLWMModel, Gaussians, render


# ===========================================================================
# DLWMLoss
# ===========================================================================

class DLWMLoss(nn.Module):
    """DLWM 总损失：L_rec = w_d * L_d + w_pd * L_pd + w_sem * L_sem。

    动态类别过滤机制：
      - train_classes=None  → 所有像素均参与（深度有效 & 非背景）
      - train_classes=[1,2] → 仅 semantic_gt ∈ {1, 2} 的像素参与 depth/sem loss

    Args:
        weight_sparse_depth: L_d 权重，默认 1.0
        weight_dense_depth:  L_pd 权重，默认 0.05
        weight_semantic:     L_sem 权重，默认 1.0
        num_classes:         总类别数（含背景 0），用于 CE Loss ignore_index
        train_classes:       参与训练的类别 id 列表；None 表示全部非背景类别
        depth_ignore_value:  深度图中代表"无效"的值（通常为 0.0）
    """

    def __init__(
        self,
        weight_sparse_depth: float = 1.0,
        weight_dense_depth:  float = 0.05,
        weight_semantic:     float = 1.0,
        num_classes:         int   = 16,
        train_classes:       Optional[List[int]] = None,
        depth_ignore_value:  float = 0.0,
    ) -> None:
        super().__init__()
        self.w_sparse  = weight_sparse_depth
        self.w_dense   = weight_dense_depth
        self.w_sem     = weight_semantic
        self.num_classes = num_classes
        self.train_classes = train_classes
        self.depth_ignore = depth_ignore_value

        # CE Loss（不设 ignore_index，通过手动 masking 实现动态过滤）
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    # ------------------------------------------------------------------
    # 内部：生成类别 mask
    # ------------------------------------------------------------------

    def _build_class_mask(
        self,
        semantic_gt: torch.Tensor,   # [B, N, H, W]  int64
    ) -> torch.Tensor:               # [B, N, H, W]  bool
        """根据 train_classes 生成像素级类别过滤 mask。

        Returns:
            mask: True 表示该像素属于 train_classes 中的类别，参与损失计算。
                  若 train_classes 为 None，则所有非背景（非 0）像素均为 True。
        """
        if self.train_classes is None:
            # 所有非背景像素
            return semantic_gt > 0    # [B, N, H, W]

        mask = torch.zeros_like(semantic_gt, dtype=torch.bool)
        for cls_id in self.train_classes:
            mask = mask | (semantic_gt == cls_id)
        return mask  # [B, N, H, W]

    # ------------------------------------------------------------------
    # L_d：稀疏深度 L1
    # ------------------------------------------------------------------

    def _loss_sparse_depth(
        self,
        depth_pred:      torch.Tensor,   # [B, N, 1, H, W]
        sparse_depth_gt: torch.Tensor,   # [B, N, 1, H, W]
        class_mask:      torch.Tensor,   # [B, N, H, W]  bool
    ) -> torch.Tensor:
        """L_d：仅在 sparse_depth_gt > 0 且 class_mask == True 的像素计算 L1。

        Returns:
            scalar loss
        """
        valid_depth_mask = (sparse_depth_gt > self.depth_ignore).squeeze(2)  # [B, N, H, W]
        combined_mask = valid_depth_mask & class_mask                         # [B, N, H, W]

        if not combined_mask.any():
            return depth_pred.new_tensor(0.0)

        pred_masked = depth_pred.squeeze(2)[combined_mask]    # [M]
        gt_masked   = sparse_depth_gt.squeeze(2)[combined_mask]  # [M]
        return F.l1_loss(pred_masked, gt_masked, reduction='mean')

    # ------------------------------------------------------------------
    # L_pd：稠密伪深度 L1
    # ------------------------------------------------------------------

    def _loss_dense_depth(
        self,
        depth_pred:     torch.Tensor,   # [B, N, 1, H, W]
        dense_depth_gt: torch.Tensor,   # [B, N, 1, H, W]
        class_mask:     torch.Tensor,   # [B, N, H, W]  bool
    ) -> torch.Tensor:
        """L_pd：仅在 dense_depth_gt > 0 且 class_mask == True 的像素计算 L1。

        Returns:
            scalar loss
        """
        valid_mask = (dense_depth_gt > self.depth_ignore).squeeze(2)  # [B, N, H, W]
        combined_mask = valid_mask & class_mask                        # [B, N, H, W]

        if not combined_mask.any():
            return depth_pred.new_tensor(0.0)

        pred_masked = depth_pred.squeeze(2)[combined_mask]       # [M]
        gt_masked   = dense_depth_gt.squeeze(2)[combined_mask]   # [M]
        return F.l1_loss(pred_masked, gt_masked, reduction='mean')

    # ------------------------------------------------------------------
    # L_sem：语义交叉熵（动态 mask）
    # ------------------------------------------------------------------

    def _loss_semantic(
        self,
        semantic_pred: torch.Tensor,   # [B, N, C, H, W]  logits（Softmax 前）
        semantic_gt:   torch.Tensor,   # [B, N, H, W]     int64
        class_mask:    torch.Tensor,   # [B, N, H, W]     bool
    ) -> torch.Tensor:
        """L_sem：仅对 class_mask 内的像素计算 CrossEntropyLoss。

        实现方式：
          1. 展平为 [B*N*H*W]
          2. 用 class_mask 筛选有效像素
          3. 对筛选后的子集计算 CE Loss

        Returns:
            scalar loss
        """
        B, N, C, H, W = semantic_pred.shape

        # [B*N, C, H*W]  — 方便按像素 mask
        logits_flat = semantic_pred.view(B * N, C, H * W)     # [B*N, C, H*W]
        gt_flat     = semantic_gt.view(B * N, H * W).long()   # [B*N, H*W]
        mask_flat   = class_mask.view(B * N, H * W)           # [B*N, H*W]  bool

        # 只保留 class_mask 为 True 的像素
        # 将 [B*N, H*W] mask 展平为 [B*N*H*W]
        logits_all  = logits_flat.permute(0, 2, 1).reshape(-1, C)   # [B*N*H*W, C]
        gt_all      = gt_flat.reshape(-1)                            # [B*N*H*W]
        mask_all    = mask_flat.reshape(-1)                          # [B*N*H*W]

        if not mask_all.any():
            return semantic_pred.new_tensor(0.0)

        # 仅对 mask 内的像素计算 CE
        logits_sel = logits_all[mask_all]   # [M, C]
        gt_sel     = gt_all[mask_all]       # [M]  int64

        # CE Loss（reduction='mean'）
        return F.cross_entropy(logits_sel, gt_sel, reduction='mean')

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(
        self,
        depth_pred:      torch.Tensor,   # [B, N, 1, H, W]  渲染深度（米）
        semantic_pred:   torch.Tensor,   # [B, N, C, H, W]  语义 logits
        sparse_depth_gt: torch.Tensor,   # [B, N, 1, H, W]  稀疏 LiDAR 深度 GT
        dense_depth_gt:  torch.Tensor,   # [B, N, 1, H, W]  稠密伪深度 GT
        semantic_gt:     torch.Tensor,   # [B, N, H, W]     语义标签（int64）
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失。

        Returns:
            total_loss: 标量总损失（可直接 .backward()）
            loss_dict:  各分项损失值字典（float，已 detach）
        """
        # 1. 生成类别 mask  [B, N, H, W]
        class_mask = self._build_class_mask(semantic_gt)

        # 2. 各分项损失
        l_d   = self._loss_sparse_depth(depth_pred, sparse_depth_gt, class_mask)
        l_pd  = self._loss_dense_depth(depth_pred, dense_depth_gt, class_mask)
        l_sem = self._loss_semantic(semantic_pred, semantic_gt, class_mask)

        total = self.w_sparse * l_d + self.w_dense * l_pd + self.w_sem * l_sem

        loss_dict: Dict[str, float] = {
            'L_d':    l_d.detach().mean().item(),
            'L_pd':   l_pd.detach().mean().item(),
            'L_sem':  l_sem.detach().mean().item(),
            'L_total': total.detach().mean().item(),
        }
        return total, loss_dict


# ===========================================================================
# 单 epoch 训练函数
# ===========================================================================

def train_one_epoch(
    model:        DLWMModel,
    dataloader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    loss_fn:      DLWMLoss,
    device:       torch.device,
    epoch:        int,
    print_freq:   int = 50,
    use_amp:      bool = False,
    target_cam_idx: int = 0,
) -> Dict[str, float]:
    """执行一个 epoch 的训练。

    NVS 策略（Novel View Synthesis）：
      在每个 batch 中，随机或固定地选取其中一个相机视角作为"目标视角"，
      将对应的深度 GT 和语义 GT 作为渲染监督。
      其余相机视角用于特征提取（输入图像）。

    Args:
        model:           DLWMModel
        dataloader:      训练数据加载器
        optimizer:       优化器
        loss_fn:         DLWMLoss
        device:          计算设备
        epoch:           当前 epoch 号（0-based）
        print_freq:      日志打印频率（iteration）
        use_amp:         是否使用混合精度训练（torch.cuda.amp）
        target_cam_idx:  目标相机视角索引（用于 NVS 监督），-1 = 所有视角

    Returns:
        avg_losses: 本 epoch 各项损失的平均值字典
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    running_losses: Dict[str, float] = {
        'L_d': 0.0, 'L_pd': 0.0, 'L_sem': 0.0, 'L_total': 0.0
    }
    num_iters = 0
    t0 = time.perf_counter()

    for i_iter, batch in enumerate(dataloader):
        # ── 数据搬移到 device ────────────────────────────────────────
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        imgs            = batch['imgs']             # [B, N_cams, 3, H, W]
        intrinsics      = batch['intrinsics']       # [B, N_cams, 4, 4]
        ego2cam         = batch['ego2cam']          # [B, N_cams, 4, 4]
        sparse_depth_gt = batch['sparse_depth_gt']  # [B, N_cams, 1, H, W]
        dense_depth_gt  = batch['dense_depth_gt']   # [B, N_cams, 1, H, W]
        semantic_gt     = batch['semantic_gt']      # [B, N_cams, H, W]

        B, N_cams, _, H, W = imgs.shape

        # ── 构建 metas（供模型内部特征聚合使用） ─────────────────────
        metas: Dict[str, object] = {
            'projection': batch.get('projection'),  # [B, N_cams, 4, 4]
            'intrinsics': intrinsics,
            'ego2cam':    ego2cam,
        }

        # ── 前向传播 ─────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=use_amp):
            model_out = model(imgs=imgs, metas=metas)

        gaussians: Gaussians = model_out['last_gaussians']

        # ── 选取目标视角并渲染 ────────────────────────────────────────
        # 确定需要渲染的相机视角索引
        if target_cam_idx == -1:
            render_cam_indices = list(range(N_cams))   # 所有视角
        else:
            render_cam_indices = [target_cam_idx]

        all_depth_preds:    List[torch.Tensor] = []    # 各视角 [B, 1, H, W]
        all_semantic_preds: List[torch.Tensor] = []    # 各视角 [B, C, H, W]

        for cam_idx in render_cam_indices:
            target_cam = Camera(
                K       = intrinsics[:, cam_idx],   # [B, 4, 4]
                ego2cam = ego2cam[:, cam_idx],       # [B, 4, 4]
                width   = W,
                height  = H,
                near    = 0.1,
                far     = 100.0,
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                depth_pred, semantic_pred, _ = render(gaussians, target_cam)
                # depth_pred:    [B, 1, H, W]
                # semantic_pred: [B, C, H, W]

            all_depth_preds.append(depth_pred)
            all_semantic_preds.append(semantic_pred)

        # ── 堆叠为 [B, N_render, 1/C, H, W] ─────────────────────────
        n_render = len(render_cam_indices)
        # [B, n_render, 1, H, W]
        depth_preds_cat   = torch.stack(all_depth_preds,    dim=1)
        # [B, n_render, C, H, W]
        semantic_preds_cat = torch.stack(all_semantic_preds, dim=1)

        # 取对应相机的 GT（只保留渲染的视角）
        render_idx_tensor = torch.tensor(render_cam_indices, device=device)
        sp_gt_sel  = sparse_depth_gt[:, render_idx_tensor]   # [B, n_render, 1, H, W]
        dn_gt_sel  = dense_depth_gt[:, render_idx_tensor]    # [B, n_render, 1, H, W]
        sem_gt_sel = semantic_gt[:, render_idx_tensor]       # [B, n_render, H, W]

        # ── 计算损失 ─────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=use_amp):
            total_loss, loss_dict = loss_fn(
                depth_pred      = depth_preds_cat,
                semantic_pred   = semantic_preds_cat,
                sparse_depth_gt = sp_gt_sel,
                dense_depth_gt  = dn_gt_sel,
                semantic_gt     = sem_gt_sel,
            )

        # ── 反向传播 & 梯度更新 ────────────────────────────────────────
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()

        # ── 统计 ────────────────────────────────────────────────────
        for k in running_losses:
            running_losses[k] += loss_dict.get(k, 0.0)
        num_iters += 1

        if i_iter % print_freq == 0:
            elapsed = time.perf_counter() - t0
            lr = optimizer.param_groups[0]['lr']
            loss_str = '  '.join(
                f"{k}={loss_dict.get(k, 0.0):.4f}" for k in loss_dict
            )
            print(
                f"[TRAIN] Epoch {epoch:3d}  Iter {i_iter:5d}/{len(dataloader):5d}  "
                f"lr={lr:.6f}  t={elapsed:.1f}s  {loss_str}"
            )
            t0 = time.perf_counter()

    avg_losses = {k: v / max(num_iters, 1) for k, v in running_losses.items()}
    return avg_losses


# ===========================================================================
# 验证（无梯度）
# ===========================================================================

@torch.no_grad()
def evaluate(
    model:          DLWMModel,
    dataloader:     DataLoader,
    loss_fn:        DLWMLoss,
    device:         torch.device,
    target_cam_idx: int = 0,
) -> Dict[str, float]:
    """在验证集上评估模型，返回平均损失字典。"""
    model.eval()

    running_losses: Dict[str, float] = {
        'L_d': 0.0, 'L_pd': 0.0, 'L_sem': 0.0, 'L_total': 0.0
    }
    num_iters = 0

    for batch in dataloader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        imgs            = batch['imgs']
        intrinsics      = batch['intrinsics']
        ego2cam         = batch['ego2cam']
        sparse_depth_gt = batch['sparse_depth_gt']
        dense_depth_gt  = batch['dense_depth_gt']
        semantic_gt     = batch['semantic_gt']
        B, N_cams, _, H, W = imgs.shape

        metas = {
            'projection': batch.get('projection'),
            'intrinsics': intrinsics,
            'ego2cam':    ego2cam,
        }

        model_out = model(imgs=imgs, metas=metas)
        gaussians = model_out['last_gaussians']

        cam_idx = target_cam_idx % N_cams
        target_cam = Camera(
            K       = intrinsics[:, cam_idx],
            ego2cam = ego2cam[:, cam_idx],
            width   = W,
            height  = H,
        )
        depth_pred, semantic_pred, _ = render(gaussians, target_cam)

        # [B, 1, 1, H, W] → [B, 1, H, W] 使用 unsqueeze(1) 对齐维度
        total_loss, loss_dict = loss_fn(
            depth_pred      = depth_pred.unsqueeze(1),
            semantic_pred   = semantic_pred.unsqueeze(1),
            sparse_depth_gt = sparse_depth_gt[:, cam_idx:cam_idx+1],
            dense_depth_gt  = dense_depth_gt[:, cam_idx:cam_idx+1],
            semantic_gt     = semantic_gt[:, cam_idx:cam_idx+1],
        )
        for k in running_losses:
            running_losses[k] += loss_dict.get(k, 0.0)
        num_iters += 1

    return {k: v / max(num_iters, 1) for k, v in running_losses.items()}


# ===========================================================================
# 训练主入口
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    """DLWM 训练主函数。"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备：{device}")

    # ── 数据集 ────────────────────────────────────────────────────────
    train_classes = args.train_classes if args.train_classes else None
    print(f"[INFO] 训练类别：{train_classes or '全部'}")

    train_dataset = DLWMDataset(
        data_root       = args.data_root,
        target_size     = tuple(args.img_size),
        start_timestep  = args.start_timestep,
        end_timestep    = args.end_timestep,
        num_classes     = args.num_classes,
        phase           = 'train',
    )
    val_dataset = DLWMDataset(
        data_root       = args.val_root or args.data_root,
        target_size     = tuple(args.img_size),
        start_timestep  = 0,
        end_timestep    = -1,
        num_classes     = args.num_classes,
        phase           = 'val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    print(f"[INFO] 训练帧数：{len(train_dataset)}，验证帧数：{len(val_dataset)}")

    # ── 模型 ──────────────────────────────────────────────────────────
    model = DLWMModel(
        num_classes  = args.num_classes,
        num_anchor   = args.num_anchor,
        embed_dims   = args.embed_dims,
        num_decoder  = args.num_decoder,
        pc_range     = args.pc_range,
        scale_range  = args.scale_range,
        num_cams     = len(train_dataset.cam_keys),
        pretrained   = not args.no_pretrain,
    ).to(device)
    model.init_weights()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 可训练参数量：{n_params / 1e6:.2f}M")

    # ── 损失函数 ───────────────────────────────────────────────────────
    loss_fn = DLWMLoss(
        weight_sparse_depth = args.w_sparse,
        weight_dense_depth  = args.w_dense,
        weight_semantic     = args.w_sem,
        num_classes         = args.num_classes,
        train_classes       = train_classes,
    ).to(device)

    # ── 优化器 & 调度器 ───────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs * len(train_loader),
        eta_min = args.lr * 0.1,
    )

    # ── 断点续训 ──────────────────────────────────────────────────────
    start_epoch = 0
    os.makedirs(args.work_dir, exist_ok=True)
    ckpt_latest = os.path.join(args.work_dir, 'latest.pth')
    if os.path.isfile(ckpt_latest):
        ckpt = torch.load(ckpt_latest, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f"[INFO] 从 epoch {start_epoch} 恢复训练")

    # ── 训练循环 ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # 调整调度器（基于 iteration 更新，在 train_one_epoch 内部不调用）
        # 这里改为 epoch 级更新，简化代码
        train_losses = train_one_epoch(
            model          = model,
            dataloader     = train_loader,
            optimizer      = optimizer,
            loss_fn        = loss_fn,
            device         = device,
            epoch          = epoch,
            print_freq     = args.print_freq,
            use_amp        = args.amp,
            target_cam_idx = args.target_cam_idx,
        )
        scheduler.step()

        train_str = '  '.join(f"{k}={v:.4f}" for k, v in train_losses.items())
        print(f"[TRAIN] Epoch {epoch:3d} 平均损失：{train_str}")

        # ── 保存 checkpoint ─────────────────────────────────────────
        ckpt_dict = {
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch':     epoch + 1,
        }
        epoch_path = os.path.join(args.work_dir, f'epoch_{epoch+1:03d}.pth')
        torch.save(ckpt_dict, epoch_path)
        torch.save(ckpt_dict, ckpt_latest)
        print(f"[INFO] 已保存：{epoch_path}")

        # ── 验证 ─────────────────────────────────────────────────────
        if (epoch + 1) % args.eval_every == 0:
            val_losses = evaluate(
                model          = model,
                dataloader     = val_loader,
                loss_fn        = loss_fn,
                device         = device,
                target_cam_idx = args.target_cam_idx,
            )
            val_str = '  '.join(f"{k}={v:.4f}" for k, v in val_losses.items())
            print(f"[EVAL]  Epoch {epoch+1:3d} 验证损失：{val_str}")


# ===========================================================================
# CLI 参数解析
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DLWM 一阶段场景重建训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────────────────
    p.add_argument('--data_root',       type=str,   required=True,
                   help='训练集根目录（含 DEPTH_GT_PNG、SEMANTIC_GT_PNG 等子目录）')
    p.add_argument('--val_root',        type=str,   default='',
                   help='验证集根目录；留空则与 data_root 相同')
    p.add_argument('--img_size',        type=int,   nargs=2, default=[512, 1408],
                   metavar=('H', 'W'),  help='目标图像尺寸')
    p.add_argument('--start_timestep',  type=int,   default=0,
                   help='训练集起始帧序号（0-based）')
    p.add_argument('--end_timestep',    type=int,   default=-1,
                   help='训练集结束帧序号（不含），-1 = 末尾')
    p.add_argument('--num_workers',     type=int,   default=4,
                   help='DataLoader 工作进程数')

    # ── 模型 ──────────────────────────────────────────────────────────
    p.add_argument('--num_classes',     type=int,   default=16,
                   help='语义类别总数（含背景 0）')
    p.add_argument('--num_anchor',      type=int,   default=6400,
                   help='初始 Gaussian 锚点数量')
    p.add_argument('--embed_dims',      type=int,   default=128,
                   help='实例特征维度')
    p.add_argument('--num_decoder',     type=int,   default=6,
                   help='Gaussian 精化解码器层数')
    p.add_argument('--pc_range',        type=float, nargs=6,
                   default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                   metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'))
    p.add_argument('--scale_range',     type=float, nargs=2, default=[0.05, 5.0],
                   metavar=('S_MIN', 'S_MAX'))
    p.add_argument('--no_pretrain',     action='store_true',
                   help='不加载 ImageNet 预训练权重')

    # ── 损失 ──────────────────────────────────────────────────────────
    p.add_argument('--train_classes',   type=int,   nargs='+', default=None,
                   help='只训练这些类别 id（如 1 2）；不指定则训练所有非背景类别')
    p.add_argument('--w_sparse',        type=float, default=1.0,
                   help='稀疏深度损失权重')
    p.add_argument('--w_dense',         type=float, default=0.05,
                   help='稠密深度损失权重')
    p.add_argument('--w_sem',           type=float, default=1.0,
                   help='语义损失权重')

    # ── 训练 ──────────────────────────────────────────────────────────
    p.add_argument('--epochs',          type=int,   default=24)
    p.add_argument('--batch_size',      type=int,   default=1)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--weight_decay',    type=float, default=0.01)
    p.add_argument('--amp',             action='store_true',
                   help='使用混合精度训练（torch.cuda.amp）')
    p.add_argument('--device',          type=str,   default='cuda')
    p.add_argument('--work_dir',        type=str,   default='./work_dirs/dlwm_run1',
                   help='输出目录（checkpoint、日志）')
    p.add_argument('--print_freq',      type=int,   default=50)
    p.add_argument('--eval_every',      type=int,   default=1,
                   help='每隔多少 epoch 做一次验证')
    p.add_argument('--target_cam_idx',  type=int,   default=0,
                   help='NVS 目标相机索引；-1 = 所有相机同时作为渲染目标')

    return p


if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()
    main(args)
