"""
LTDatasetRender: 扩展 LTDataset，新增深度与语义 2D 监督信号。

在原有 LTDataset 的基础上，增加以下监督数据的加载：

1. sparse_depth_gt    [N, 1, H, W]  float32
   - LiDAR 点云投影到各相机图像平面的稀疏深度图
   - 需预先离线生成并存储为 .npy 格式
   - 无效像素值为 0.0

2. valid_lidar_mask   [N, 1, H, W]  bool
   - 稀疏深度图的有效像素 mask（True=有 LiDAR 点）

3. dense_depth_gt     [N, 1, H, W]  float32
   - 由深度补全算法生成的稠密伪深度（如 NLSPN / DySPN）
   - 需预先离线生成并存储为 .npy 格式

4. semantic_gt        [N, H, W]     int64
   - LiDAR 语义点云投影到各相机图像平面的 2D 语义标签图
   - 无效像素标记为 empty_label（默认 0）
   - 需预先离线生成并存储为 .npy 格式

目录结构约定（均在 data_root 下）：
    data_root/
    ├── SPARSE_DEPTH/
    │   ├── <lidar_timestamp_ns>_<cam_name>.npy    # float32 [H_orig, W_orig]
    ├── DENSE_DEPTH/
    │   ├── <lidar_timestamp_ns>_<cam_name>.npy    # float32 [H_orig, W_orig]
    └── SEMANTIC_2D/
        ├── <lidar_timestamp_ns>_<cam_name>.npy    # int16 [H_orig, W_orig]

离线生成工具可参考 get_occ_gt.py 的 LiDAR 投影逻辑。
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF

from . import OPENOCC_DATASET
from .lt_dataset import LTDataset


@OPENOCC_DATASET.register_module()
class LTDatasetRender(LTDataset):
    """扩展数据集：在 LTDataset 基础上增加渲染监督信号。

    Args:
        data_root (str):          数据集根目录
        target_size (tuple):      目标图像尺寸 (H, W)
        sparse_depth_dir (str):   稀疏深度 .npy 文件目录（相对或绝对路径）
        dense_depth_dir (str):    稠密伪深度 .npy 文件目录
        semantic_2d_dir (str):    2D 语义标签 .npy 文件目录
        num_classes (int):        语义类别总数（含 empty_label），默认 16
        empty_label (int):        空/忽略类别 id，默认 0
        use_target_views (bool):  是否额外加载"目标视角"（不同于输入视角）。
                                  若为 False，则 target_intrinsics = intrinsics，
                                  target_cam2ego = cam2ego（与输入视角相同）。
        **kwargs:                 透传给 LTDataset 的其余参数
    """

    def __init__(
        self,
        data_root: str,
        target_size: tuple = (512, 1408),
        sparse_depth_dir: str = 'SPARSE_DEPTH',
        dense_depth_dir: str = 'DENSE_DEPTH',
        semantic_2d_dir: str = 'SEMANTIC_2D',
        num_classes: int = 16,
        empty_label: int = 0,
        use_target_views: bool = False,
        **kwargs,
    ):
        # 先调用父类初始化（包含标定、相机样本、数据索引构建）
        super().__init__(data_root=data_root, target_size=target_size, **kwargs)

        self.num_classes = num_classes
        self.empty_label = empty_label
        self.use_target_views = use_target_views

        # 解析监督数据目录（支持相对路径和绝对路径）
        self.sparse_depth_dir = self._resolve_dir(sparse_depth_dir)
        self.dense_depth_dir = self._resolve_dir(dense_depth_dir)
        self.semantic_2d_dir = self._resolve_dir(semantic_2d_dir)

    def _resolve_dir(self, dir_path: str) -> Optional[str]:
        """将相对路径转换为绝对路径，目录不存在时返回 None 并打印警告。"""
        if dir_path is None:
            return None
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(self.data_root, dir_path)
        if not os.path.isdir(dir_path):
            print(
                f"[LTDatasetRender] 警告：目录不存在，相关监督将使用零填充：{dir_path}"
            )
            return None
        return dir_path

    def _npy_path(self, base_dir: Optional[str], lidar_ts_ns: int, cam: str) -> Optional[str]:
        """构建 .npy 文件路径。"""
        if base_dir is None:
            return None
        path = os.path.join(base_dir, f'{lidar_ts_ns}_{cam}.npy')
        return path if os.path.isfile(path) else None

    def _load_depth_map(
        self,
        npy_path: Optional[str],
        target_h: int,
        target_w: int,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """加载深度图并缩放至目标尺寸。

        Args:
            npy_path:  .npy 文件路径；为 None 时返回全零张量
            target_h, target_w: 目标尺寸
            orig_h, orig_w:     原始图像尺寸（用于验证）

        Returns:
            depth: [1, H, W]  float32
        """
        if npy_path is not None:
            arr = np.load(npy_path).astype(np.float32)  # [H_orig, W_orig]
        else:
            arr = np.zeros((orig_h, orig_w), dtype=np.float32)

        # NEAREST 插值缩放：保证无效像素（0.0）不被双线性插值污染
        t = torch.from_numpy(arr)[None, None]  # [1, 1, H, W]
        t = TF.resize(t.squeeze(0), [target_h, target_w], interpolation=TF.InterpolationMode.NEAREST)
        return t.float()  # [1, H, W]

    def _load_semantic_map(
        self,
        npy_path: Optional[str],
        target_h: int,
        target_w: int,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """加载语义标签图并缩放至目标尺寸。

        Args:
            npy_path:  .npy 文件路径；为 None 时返回全 empty_label 张量
            target_h, target_w: 目标尺寸
            orig_h, orig_w:     原始图像尺寸

        Returns:
            semantic: [H, W]  int64，类别 id
        """
        if npy_path is not None:
            arr = np.load(npy_path).astype(np.int16)  # [H_orig, W_orig]
        else:
            arr = np.full((orig_h, orig_w), self.empty_label, dtype=np.int16)

        t = torch.from_numpy(arr.astype(np.int32))[None, None]  # [1, 1, H, W]
        t = TF.resize(t.squeeze(0), [target_h, target_w], interpolation=TF.InterpolationMode.NEAREST)
        return t.long()  # [H, W]

    def __getitem__(self, index: int) -> dict:
        """获取单个样本，包含图像、标定、OCC GT 及渲染监督信号。

        Returns:
            dict 包含所有 LTDataset 字段，以及：
                sparse_depth_gt   [N, 1, H, W]  float32  稀疏 LiDAR 深度
                valid_lidar_mask  [N, 1, H, W]  bool     有效像素 mask
                dense_depth_gt    [N, 1, H, W]  float32  稠密伪深度
                semantic_gt       [N, H, W]     int64    2D 语义标签
                target_intrinsics [M, 4, 4]     float32  目标视角内参
                target_cam2ego    [M, 4, 4]     float32  目标视角外参
        """
        # 先获取父类的基础样本
        sample = super().__getitem__(index)

        info = self.data_infos[index]
        lidar_ts_ns = info['lidar_timestamp_ns']
        target_h, target_w = self.target_size

        sparse_depth_list: List[torch.Tensor] = []
        dense_depth_list: List[torch.Tensor] = []
        semantic_gt_list: List[torch.Tensor] = []

        for cam in self.cams:
            # 原始图像尺寸（用于正确缩放）
            orig_h = self.calibs[cam]['orig_H']
            orig_w = self.calibs[cam]['orig_W']

            # ----- 稀疏深度（LiDAR 投影） -----
            sp_path = self._npy_path(self.sparse_depth_dir, lidar_ts_ns, cam)
            sparse_depth = self._load_depth_map(
                sp_path, target_h, target_w, orig_h, orig_w
            )  # [1, H, W]
            sparse_depth_list.append(sparse_depth)

            # ----- 稠密伪深度 -----
            dn_path = self._npy_path(self.dense_depth_dir, lidar_ts_ns, cam)
            dense_depth = self._load_depth_map(
                dn_path, target_h, target_w, orig_h, orig_w
            )  # [1, H, W]
            dense_depth_list.append(dense_depth)

            # ----- 2D 语义标签 -----
            sem_path = self._npy_path(self.semantic_2d_dir, lidar_ts_ns, cam)
            semantic_map = self._load_semantic_map(
                sem_path, target_h, target_w, orig_h, orig_w
            )  # [H, W]
            semantic_gt_list.append(semantic_map)

        # 堆叠为批次维度 [N, ...]
        sparse_depth_gt = torch.stack(sparse_depth_list)   # [N, 1, H, W]
        valid_lidar_mask = (sparse_depth_gt > 0.0)         # [N, 1, H, W] bool
        dense_depth_gt = torch.stack(dense_depth_list)     # [N, 1, H, W]
        semantic_gt = torch.stack(semantic_gt_list)         # [N, H, W]

        sample['sparse_depth_gt'] = sparse_depth_gt
        sample['valid_lidar_mask'] = valid_lidar_mask
        sample['dense_depth_gt'] = dense_depth_gt
        sample['semantic_gt'] = semantic_gt

        # 目标视角（默认与输入视角相同，M = N = 7）
        sample['target_intrinsics'] = sample['intrinsics'].clone()   # [N, 4, 4]
        sample['target_cam2ego'] = sample['cam2ego'].clone()         # [N, 4, 4]

        return sample
