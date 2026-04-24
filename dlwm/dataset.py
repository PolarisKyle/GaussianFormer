"""
dlwm/dataset.py
===============
DLWM (arXiv:2604.00969) 一阶段复现 —— 数据加载器

目录结构约定（data_root 下）：
    undistorted_img/
        JPG_2K_CAM_FN/              ← 多视角去畸变图像
            frame-XXXXXX_<13位ms时间戳>.jpg
            JPG_2K_CAM_FN_mask/     ← 每帧对应的二值 mask（可选）
                frame-XXXXXX_<ts>_road.png
                frame-XXXXXX_<ts>_vehicle.png
                ...
            cam0_params.yml         ← OpenCV 风格标定文件
        JPG_2K_CAM_FW/ ...          ← 其余相机同理
    DEPTH_GT_PNG/                   ← ★ 主时间轴：LiDAR 投影稀疏深度（16-bit PNG）
        frame-XXXXXX_<ts>_<cam>.png
    SEMANTIC_GT_PNG/                ← LiDAR 投影语义标签（8-bit PNG）
        frame-XXXXXX_<ts>_<cam>.png
    metric3d_output/
        dense_depth/                ← Metric3D 稠密深度（16-bit PNG）
            frame-XXXXXX_<ts>_<cam>.png
    calib_param/                    ← 相机标定 YAML（备选位置，若 img 目录内无 yaml）

命名约定（DEPTH_GT_PNG 文件名基准）：
    frame-XXXXXX_<13位ms时间戳>_<cam_key>.png
    例：frame-000001_1701234567890_JPG_2K_CAM_FN.png

设计说明：
  - 以 DEPTH_GT_PNG 中的文件名为主时间轴（LiDAR 对齐后的图像时间戳）
  - start_timestep / end_timestep 支持整数（帧序号）或字符串（时间戳）范围切片
  - 16-bit 深度 PNG 读取后除以 1000.0 还原为米
  - 支持将多个二值 mask 合并为单张类别索引图
  - 输出字典与 GaussianFormer 训练链路兼容
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 图像标准化均值 / 标准差（ImageNet）
# ---------------------------------------------------------------------------
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# 深度图缩放因子（16-bit PNG → 米）
# ---------------------------------------------------------------------------
_DEPTH_SCALE: float = 1000.0


# ---------------------------------------------------------------------------
# 相机 key → PCO 标定名称映射（与 lt_dataset.py 保持一致）
# ---------------------------------------------------------------------------
SENSOR_TO_PCO_NAME: Dict[str, str] = {
    'JPG_2K_CAM_FN': 'cam0',
    'JPG_2K_CAM_FW': 'cam1',
    'JPG_CAM_FL':    'cam4',
    'JPG_CAM_RL':    'cam5',
    'JPG_CAM_RN':    'cam6',
    'JPG_CAM_RR':    'cam7',
    'JPG_CAM_FR':    'cam8',
}

# 二值 mask 类别名称 → 类别 id 映射（0 = 背景）
DEFAULT_MASK_CLASSES: Dict[str, int] = {
    'road':      1,
    'vehicle':   2,
    'pedestrian':3,
    'cyclist':   4,
    'cone':      5,
    'obstacle':  6,
}


# ===========================================================================
# 标定 YAML 解析工具
# ===========================================================================

def _load_calib_yaml(yaml_path: str) -> dict:
    """读取带有 '%YAML:1.0' 文件头的 OpenCV 风格标定 YAML。"""
    assert os.path.isfile(yaml_path), f"标定文件不存在：{yaml_path}"
    with open(yaml_path, 'r', encoding='utf-8') as fh:
        raw = fh.read()
    cleaned = '\n'.join(
        line for line in raw.splitlines()
        if not line.strip().startswith('%YAML')
    )
    data = yaml.safe_load(cleaned)
    assert data is not None and 'camera' in data, (
        f"YAML 格式错误，缺少 'camera' 字段：{yaml_path}"
    )
    return data['camera']


def _parse_intrinsic(cam_dict: dict) -> np.ndarray:
    """提取 4×4 齐次内参矩阵。"""
    flat = np.asarray(cam_dict['intrinsic'], dtype=np.float64)
    assert flat.size == 16, f"intrinsic 长度应为 16，实际为 {flat.size}"
    return flat.reshape(4, 4)


def _parse_distortion(cam_dict: dict) -> np.ndarray:
    """提取去畸变系数（支持 4/5/8/12/14 参数 OpenCV 模型）。"""
    raw = cam_dict.get('distorted')
    if raw is None:
        return np.zeros(5, dtype=np.float64)
    coeffs = np.asarray(raw, dtype=np.float64).reshape(-1)
    valid = [4, 5, 8, 12, 14]
    if coeffs.size in valid:
        return coeffs
    if coeffs.size >= 14:
        return coeffs[:14]
    if coeffs.size >= 8:
        return coeffs[:8]
    if coeffs.size >= 5:
        return coeffs[:5]
    out = np.zeros(5, dtype=np.float64)
    out[:coeffs.size] = coeffs
    return out


def _parse_cam2ego(cam_dict: dict) -> np.ndarray:
    """提取 4×4 cam2ego 变换矩阵。"""
    R_flat = np.asarray(cam_dict['cam2ego_R'], dtype=np.float64)
    t      = np.asarray(cam_dict['cam2ego_t'], dtype=np.float64).reshape(-1)
    assert R_flat.size == 9,  f"cam2ego_R 长度应为 9，实际 {R_flat.size}"
    assert t.size      == 3,  f"cam2ego_t 长度应为 3，实际 {t.size}"
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_flat.reshape(3, 3)
    mat[:3,  3] = t
    return mat


def _resolve_calib_yaml(calib_dir: str, pco_name: str) -> str:
    """在 calib_dir 下搜索 pco_name 对应的 YAML 文件。"""
    for suffix in (
        f'{pco_name}_params.yml',
        f'{pco_name}.yml',
        f'{pco_name}_params.yaml',
        f'{pco_name}.yaml',
    ):
        path = os.path.join(calib_dir, suffix)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"未找到 {pco_name} 的标定文件，已搜索目录：{calib_dir}"
    )


# ===========================================================================
# 文件名时间戳解析
# ===========================================================================

def _parse_timestamp_from_stem(stem: str) -> int:
    """从形如 'frame-XXXXXX_<13位ms时间戳>[_<cam_key>]' 的文件名主干中提取 13 位毫秒时间戳。

    Examples:
        'frame-000001_1701234567890'             → 1701234567890
        'frame-000001_1701234567890_JPG_2K_CAM_FN' → 1701234567890
    """
    # 提取第一个 13 位纯数字段
    matches = re.findall(r'\b(\d{13})\b', stem)
    assert matches, f"无法从文件名主干 '{stem}' 中提取 13 位毫秒时间戳"
    return int(matches[0])


def _parse_cam_key_from_depth_stem(stem: str, cam_keys: List[str]) -> Optional[str]:
    """从深度图文件名主干中识别所属相机 key。

    深度/语义文件命名约定：
        frame-XXXXXX_<ts>_<cam_key>.png
    """
    for key in cam_keys:
        # 用最长匹配，避免 'JPG_2K_CAM_FN' 与 'JPG_2K_CAM_FW' 混淆
        if stem.endswith(f'_{key}') or f'_{key}_' in stem:
            return key
    return None


# ===========================================================================
# 主数据集类
# ===========================================================================

class DLWMDataset(Dataset):
    """DLWM 一阶段复现数据集。

    以 ``DEPTH_GT_PNG`` 目录中的文件作为主时间轴（对应 LiDAR 对齐后的图像时间戳），
    再分别查找 RGB 图像、语义 GT 和稠密深度。

    Args:
        data_root:         数据集根目录
        cam_keys:          需要加载的相机目录名称列表；默认使用 SENSOR_TO_PCO_NAME 中所有键
        target_size:       输出图像尺寸 ``(H, W)``，默认 ``(512, 1408)``
        img_ext:           图像文件扩展名，默认 ``'.jpg'``
        depth_gt_dir:      稀疏深度子目录，默认 ``'DEPTH_GT_PNG'``
        semantic_gt_dir:   语义 GT 子目录，默认 ``'SEMANTIC_GT_PNG'``
        dense_depth_dir:   稠密深度子目录，默认 ``'metric3d_output/dense_depth'``
        calib_dir:         标定 YAML 目录，默认 ``'calib_param'``
        start_timestep:    起始帧序号（含），0-based 整数；或 13 位毫秒时间戳字符串
        end_timestep:      结束帧序号（不含），-1 表示到末尾；或 13 位毫秒时间戳字符串
        num_classes:       语义类别总数（含背景 0），默认 16
        mask_classes:      二值 mask 类别名称 → id 映射；为 None 则直接使用 SEMANTIC_GT_PNG
        use_undistort:     是否对 RGB 图像执行去畸变，默认 True
        phase:             'train' / 'val' / 'test'
    """

    def __init__(
        self,
        data_root: str,
        cam_keys: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (512, 1408),
        img_ext: str = '.jpg',
        depth_gt_dir: str = 'DEPTH_GT_PNG',
        semantic_gt_dir: str = 'SEMANTIC_GT_PNG',
        dense_depth_dir: str = 'metric3d_output/dense_depth',
        calib_dir: str = 'calib_param',
        start_timestep: Union[int, str] = 0,
        end_timestep: Union[int, str] = -1,
        num_classes: int = 16,
        mask_classes: Optional[Dict[str, int]] = None,
        use_undistort: bool = True,
        phase: str = 'train',
    ) -> None:
        assert os.path.isdir(data_root), f"data_root 不存在：{data_root}"

        self.data_root = data_root
        self.target_size = tuple(target_size)   # (H, W)
        self.img_ext = img_ext
        self.num_classes = num_classes
        self.mask_classes = mask_classes or DEFAULT_MASK_CLASSES
        self.use_undistort = use_undistort
        self.phase = phase

        # ── 解析目录路径 ──────────────────────────────────────────────
        def _abs(sub: str) -> str:
            return sub if os.path.isabs(sub) else os.path.join(data_root, sub)

        self.depth_gt_dir    = _abs(depth_gt_dir)
        self.semantic_gt_dir = _abs(semantic_gt_dir)
        self.dense_depth_dir = _abs(dense_depth_dir)
        self.calib_dir       = _abs(calib_dir)

        # ── 确定可用相机 ──────────────────────────────────────────────
        if cam_keys is None:
            cam_keys = [
                k for k in SENSOR_TO_PCO_NAME
                if os.path.isdir(os.path.join(data_root, 'undistorted_img', k))
            ]
        assert cam_keys, (
            f"在 {data_root}/undistorted_img 下未找到任何相机目录，"
            f"支持的键：{list(SENSOR_TO_PCO_NAME.keys())}"
        )
        self.cam_keys = cam_keys

        # ── 加载标定参数 ──────────────────────────────────────────────
        self.calibs: Dict[str, dict] = self._load_calibs()

        # ── 构建相机图像索引（时间戳 → 图像路径） ──────────────────────
        self.cam_img_index: Dict[str, Dict[int, str]] = self._build_cam_img_index()

        # ── 以 DEPTH_GT_PNG 为主时间轴构建数据 infos ──────────────────
        self.data_infos: List[dict] = self._build_data_infos(
            start_timestep, end_timestep
        )

    # ------------------------------------------------------------------
    # 初始化辅助方法
    # ------------------------------------------------------------------

    def _load_calibs(self) -> Dict[str, dict]:
        """加载所有相机的标定参数。"""
        calibs: Dict[str, dict] = {}
        for cam in self.cam_keys:
            pco = SENSOR_TO_PCO_NAME[cam]

            # 优先从相机目录下查找 yml，其次从 calib_dir
            img_dir = os.path.join(self.data_root, 'undistorted_img', cam)
            yaml_path: Optional[str] = None
            for suffix in (
                f'{pco}_params.yml', f'{pco}.yml',
                f'{pco}_params.yaml', f'{pco}.yaml',
            ):
                for base_dir in (img_dir, self.calib_dir):
                    cand = os.path.join(base_dir, suffix)
                    if os.path.isfile(cand):
                        yaml_path = cand
                        break
                if yaml_path:
                    break

            if yaml_path is None:
                # 最后尝试 _resolve_calib_yaml 会抛出更清晰的错误
                yaml_path = _resolve_calib_yaml(self.calib_dir, pco)

            cam_dict = _load_calib_yaml(yaml_path)
            orig_h = int(
                cam_dict['image_height'][0]
                if isinstance(cam_dict['image_height'], (list, tuple))
                else cam_dict['image_height']
            )
            orig_w = int(
                cam_dict['image_width'][0]
                if isinstance(cam_dict['image_width'], (list, tuple))
                else cam_dict['image_width']
            )
            calibs[cam] = {
                'K':       _parse_intrinsic(cam_dict),   # [4, 4]
                'D':       _parse_distortion(cam_dict),  # [n]
                'cam2ego': _parse_cam2ego(cam_dict),      # [4, 4]
                'orig_H':  orig_h,
                'orig_W':  orig_w,
            }
        return calibs

    def _build_cam_img_index(self) -> Dict[str, Dict[int, str]]:
        """为每个相机构建 {毫秒时间戳 → 图像路径} 的快速索引。"""
        index: Dict[str, Dict[int, str]] = {}
        for cam in self.cam_keys:
            img_dir = os.path.join(self.data_root, 'undistorted_img', cam)
            assert os.path.isdir(img_dir), f"相机图像目录不存在：{img_dir}"
            paths = sorted(glob.glob(os.path.join(img_dir, f'*{self.img_ext}')))
            assert paths, f"相机 {cam} 目录下未找到 {self.img_ext} 文件"
            ts_to_path: Dict[int, str] = {}
            for p in paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                ts = _parse_timestamp_from_stem(stem)
                ts_to_path[ts] = p
            index[cam] = ts_to_path
        return index

    def _build_data_infos(
        self,
        start_timestep: Union[int, str],
        end_timestep: Union[int, str],
    ) -> List[dict]:
        """以 DEPTH_GT_PNG 目录为主时间轴扫描所有可用帧，并按 [start, end) 切片。

        DEPTH_GT_PNG 文件命名约定：
            frame-XXXXXX_<13位ms时间戳>_<cam_key>.png

        一帧 = 一个时间戳（同一时间戳可能对应多个相机，但我们按时间戳聚合）。
        """
        assert os.path.isdir(self.depth_gt_dir), (
            f"DEPTH_GT_PNG 目录不存在：{self.depth_gt_dir}"
        )
        depth_files = sorted(glob.glob(os.path.join(self.depth_gt_dir, '*.png')))
        assert depth_files, f"DEPTH_GT_PNG 目录下未找到 PNG 文件：{self.depth_gt_dir}"

        # 以 (时间戳, cam_key) 为 key，按时间戳聚合
        # 每个时间戳对应一个完整的"帧"（包含所有相机）
        ts_to_cam_depth: Dict[int, Dict[str, str]] = {}
        for fp in depth_files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            ts = _parse_timestamp_from_stem(stem)
            cam = _parse_cam_key_from_depth_stem(stem, self.cam_keys)
            if cam is None:
                continue   # 不属于任何配置相机，跳过
            if ts not in ts_to_cam_depth:
                ts_to_cam_depth[ts] = {}
            ts_to_cam_depth[ts][cam] = fp

        # 只保留所有相机均有深度 GT 的时间戳
        full_timestamps = sorted([
            ts for ts, cams in ts_to_cam_depth.items()
            if all(c in cams for c in self.cam_keys)
        ])
        assert full_timestamps, (
            "DEPTH_GT_PNG 中没有包含所有相机的完整帧，请检查文件命名约定。"
        )

        # ── 时间戳切片 ──────────────────────────────────────────────
        total = len(full_timestamps)
        start_idx, end_idx = self._resolve_slice(
            full_timestamps, start_timestep, end_timestep, total
        )
        sliced_ts = full_timestamps[start_idx:end_idx]
        assert sliced_ts, (
            f"切片后没有可用帧：start={start_timestep}, end={end_timestep}"
        )

        # ── 构建 data_infos ────────────────────────────────────────
        infos: List[dict] = []
        for ts in sliced_ts:
            # 对每个相机找最近邻图像
            cam_img_paths: Dict[str, str] = {}
            for cam in self.cam_keys:
                ts_arr = np.asarray(
                    sorted(self.cam_img_index[cam].keys()), dtype=np.int64
                )
                nearest_ts = int(ts_arr[np.argmin(np.abs(ts_arr - ts))])
                cam_img_paths[cam] = self.cam_img_index[cam][nearest_ts]

            # 语义 GT 路径（与深度同命名规则）
            cam_sem_paths: Dict[str, str] = {}
            if os.path.isdir(self.semantic_gt_dir):
                for cam in self.cam_keys:
                    # 从深度路径推导语义路径（替换目录）
                    depth_basename = os.path.basename(ts_to_cam_depth[ts][cam])
                    sem_path = os.path.join(self.semantic_gt_dir, depth_basename)
                    if os.path.isfile(sem_path):
                        cam_sem_paths[cam] = sem_path

            # 稠密深度路径
            cam_dense_paths: Dict[str, str] = {}
            if os.path.isdir(self.dense_depth_dir):
                for cam in self.cam_keys:
                    depth_basename = os.path.basename(ts_to_cam_depth[ts][cam])
                    dense_path = os.path.join(self.dense_depth_dir, depth_basename)
                    if os.path.isfile(dense_path):
                        cam_dense_paths[cam] = dense_path

            infos.append({
                'timestamp_ms':       ts,
                'cam_img_paths':      cam_img_paths,
                'cam_depth_paths':    ts_to_cam_depth[ts],   # 稀疏深度
                'cam_sem_paths':      cam_sem_paths,
                'cam_dense_paths':    cam_dense_paths,
            })
        return infos

    @staticmethod
    def _resolve_slice(
        timestamps: List[int],
        start: Union[int, str],
        end: Union[int, str],
        total: int,
    ) -> Tuple[int, int]:
        """将 start/end 解析为 [start_idx, end_idx) 帧序号区间。

        若 start/end 为字符串且为纯数字，则视为毫秒时间戳，做最近邻匹配；
        否则视为整数帧序号（0-based，-1 = 末尾）。
        """
        def _to_idx(v: Union[int, str], default_end: int) -> int:
            if isinstance(v, str) and v.isdigit() and len(v) == 13:
                # 13 位时间戳字符串
                ts_arr = np.asarray(timestamps, dtype=np.int64)
                return int(np.argmin(np.abs(ts_arr - int(v))))
            iv = int(v)
            if iv < 0:
                return default_end
            return iv

        start_idx = _to_idx(start, 0)
        end_idx   = _to_idx(end, total)
        assert 0 <= start_idx <= end_idx <= total, (
            f"切片范围非法：start_idx={start_idx}, end_idx={end_idx}, total={total}"
        )
        return start_idx, end_idx

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, index: int) -> dict:
        """返回单帧数据字典。

        Returns:
            dict with keys:
                imgs           : Tensor[N, 3, H, W]  归一化 RGB
                intrinsics     : Tensor[N, 4, 4]     齐次内参（已缩放至 target_size）
                extrinsics     : Tensor[N, 4, 4]     cam2ego
                ego2cam        : Tensor[N, 4, 4]     ego2cam（extrinsics 的逆）
                projection     : Tensor[N, 4, 4]     intrinsic @ ego2cam
                sparse_depth_gt: Tensor[N, 1, H, W]  稀疏 LiDAR 深度（米），无效=0
                dense_depth_gt : Tensor[N, 1, H, W]  稠密伪深度（米），无效=0
                semantic_gt    : Tensor[N, H, W]     类别索引（int64），背景=0
                image_wh       : Tensor[N, 2]        [W, H]
                timestamp_ms   : int
                frame_id       : str
        """
        info    = self.data_infos[index]
        tgt_h, tgt_w = self.target_size

        imgs_list:         List[torch.Tensor] = []
        intrinsics_list:   List[torch.Tensor] = []
        cam2ego_list:      List[torch.Tensor] = []
        ego2cam_list:      List[torch.Tensor] = []
        projection_list:   List[torch.Tensor] = []
        sparse_depth_list: List[torch.Tensor] = []
        dense_depth_list:  List[torch.Tensor] = []
        semantic_list:     List[torch.Tensor] = []
        image_wh_list:     List[torch.Tensor] = []

        for cam in self.cam_keys:
            # ── RGB 图像 ─────────────────────────────────────────────
            img_path = info['cam_img_paths'][cam]
            with Image.open(img_path) as raw:
                pil_img = raw.convert('RGB')
                orig_w, orig_h = pil_img.size          # PIL: (W, H)

            if self.use_undistort:
                pil_img = self._undistort(pil_img, cam)

            pil_img = TF.resize(pil_img, [tgt_h, tgt_w])
            img_t = TF.to_tensor(pil_img)              # [3, H, W]  float32 in [0,1]
            img_t = TF.normalize(img_t, _IMG_MEAN, _IMG_STD)
            imgs_list.append(img_t)

            # ── 内参（随图像缩放比例更新 fx, fy, cx, cy） ───────────
            scale_w = tgt_w / orig_w
            scale_h = tgt_h / orig_h
            K = self.calibs[cam]['K'].copy()            # [4, 4]
            K[0, 0] *= scale_w;  K[0, 2] *= scale_w
            K[1, 1] *= scale_h;  K[1, 2] *= scale_h
            intrinsics_list.append(torch.from_numpy(K).float())

            # ── 外参 ────────────────────────────────────────────────
            c2e = self.calibs[cam]['cam2ego']           # [4, 4]
            e2c = np.linalg.inv(c2e)
            proj = K @ e2c                              # [4, 4]
            cam2ego_list.append(torch.from_numpy(c2e).float())
            ego2cam_list.append(torch.from_numpy(e2c).float())
            projection_list.append(torch.from_numpy(proj).float())

            image_wh_list.append(
                torch.tensor([tgt_w, tgt_h], dtype=torch.float32)
            )

            # ── 稀疏深度 GT ──────────────────────────────────────────
            depth_path = info['cam_depth_paths'].get(cam)
            if depth_path and os.path.isfile(depth_path):
                sp_depth = self._load_depth_png(depth_path, orig_h, orig_w, tgt_h, tgt_w)
            else:
                sp_depth = torch.zeros(1, tgt_h, tgt_w, dtype=torch.float32)
            sparse_depth_list.append(sp_depth)          # [1, H, W]

            # ── 稠密深度 GT ──────────────────────────────────────────
            dense_path = info['cam_dense_paths'].get(cam)
            if dense_path and os.path.isfile(dense_path):
                dn_depth = self._load_depth_png(dense_path, orig_h, orig_w, tgt_h, tgt_w)
            else:
                dn_depth = torch.zeros(1, tgt_h, tgt_w, dtype=torch.float32)
            dense_depth_list.append(dn_depth)           # [1, H, W]

            # ── 语义 GT ──────────────────────────────────────────────
            sem_path = info['cam_sem_paths'].get(cam)
            if sem_path and os.path.isfile(sem_path):
                sem = self._load_semantic(sem_path, orig_h, orig_w, tgt_h, tgt_w, cam, info)
            else:
                sem = torch.zeros(tgt_h, tgt_w, dtype=torch.long)
            semantic_list.append(sem)                   # [H, W]

        # ── 堆叠为批次维度 [N, ...] ────────────────────────────────
        imgs          = torch.stack(imgs_list)           # [N, 3, H, W]
        intrinsics    = torch.stack(intrinsics_list)     # [N, 4, 4]
        extrinsics    = torch.stack(cam2ego_list)        # [N, 4, 4]
        ego2cam       = torch.stack(ego2cam_list)        # [N, 4, 4]
        projection    = torch.stack(projection_list)     # [N, 4, 4]
        sparse_depth  = torch.stack(sparse_depth_list)  # [N, 1, H, W]
        dense_depth   = torch.stack(dense_depth_list)   # [N, 1, H, W]
        semantic_gt   = torch.stack(semantic_list)       # [N, H, W]
        image_wh      = torch.stack(image_wh_list)       # [N, 2]

        return {
            'imgs':             imgs,
            'intrinsics':       intrinsics,
            'extrinsics':       extrinsics,          # cam2ego
            'ego2cam':          ego2cam,
            'projection':       projection,           # intrinsic @ ego2cam
            'sparse_depth_gt':  sparse_depth,
            'dense_depth_gt':   dense_depth,
            'semantic_gt':      semantic_gt,
            'image_wh':         image_wh,
            'timestamp_ms':     info['timestamp_ms'],
            'frame_id':         str(info['timestamp_ms']),
        }

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _load_depth_png(
        path: str,
        orig_h: int,
        orig_w: int,
        tgt_h: int,
        tgt_w: int,
    ) -> torch.Tensor:
        """读取 16-bit 深度 PNG，返回 float32 米深度图 [1, H, W]。

        无效像素（像素值=0）在缩放后仍保持 0.0（使用 NEAREST 插值）。
        """
        # cv2.IMREAD_ANYDEPTH 以 uint16 读取（保留 16 位精度）
        arr_u16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)  # [H_orig, W_orig]  uint16
        assert arr_u16 is not None, f"深度图读取失败：{path}"
        arr_f32 = arr_u16.astype(np.float32) / _DEPTH_SCALE  # → 米

        # NEAREST 插值：保证 0.0 无效像素不被线性插值污染
        arr_resized = cv2.resize(
            arr_f32, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
        )
        t = torch.from_numpy(arr_resized.copy()).float()   # [H, W]
        return t.unsqueeze(0)                              # [1, H, W]

    def _load_semantic(
        self,
        path: str,
        orig_h: int,
        orig_w: int,
        tgt_h: int,
        tgt_w: int,
        cam: str,
        info: dict,
    ) -> torch.Tensor:
        """读取 8-bit 语义标签图，返回 int64 类别索引 [H, W]。

        若 path 存在则直接读取；否则尝试从二值 mask 目录合并。
        """
        arr_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # [H_orig, W_orig]
        if arr_u8 is None:
            # 尝试从 mask 目录合并
            arr_u8 = self._merge_masks(cam, info, orig_h, orig_w)

        arr_resized = cv2.resize(
            arr_u8.astype(np.uint8), (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
        )
        t = torch.from_numpy(arr_resized.astype(np.int64))  # [H, W]
        return t                                             # [H, W]  int64

    def _merge_masks(
        self,
        cam: str,
        info: dict,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """将 <img_dir>/<cam>_mask/ 下的多个二值 mask 合并为单张类别索引图。

        文件命名约定：
            <frame_stem>_road.png     → 1
            <frame_stem>_vehicle.png  → 2
            ...（按 self.mask_classes 中的映射）

        Returns:
            label_map: [H, W]  uint8，0=背景，其余为对应类别 id
        """
        img_path = info['cam_img_paths'][cam]
        img_stem = os.path.splitext(os.path.basename(img_path))[0]

        mask_dir = os.path.join(
            self.data_root, 'undistorted_img', cam,
            f'{cam}_mask'
        )
        label_map = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if not os.path.isdir(mask_dir):
            return label_map

        # 按类别 id 升序处理，id 较大的类别覆盖较小的（前景优先）
        for cls_name, cls_id in sorted(self.mask_classes.items(), key=lambda x: x[1]):
            mask_path = os.path.join(mask_dir, f'{img_stem}_{cls_name}.png')
            if not os.path.isfile(mask_path):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(
                    mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
            # 二值化：非零像素 → 赋予 cls_id
            label_map[mask > 0] = cls_id

        return label_map

    def _undistort(self, pil_img: Image.Image, cam: str) -> Image.Image:
        """使用 cv2.undistort 对 PIL 图像执行去畸变。"""
        calib = self.calibs[cam]
        D = calib['D']
        if D.size == 0 or not np.any(D != 0):
            return pil_img

        K3 = calib['K'][:3, :3].astype(np.float64)
        bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        bgr_ud = cv2.undistort(bgr, K3, D, None, K3)
        return Image.fromarray(cv2.cvtColor(bgr_ud, cv2.COLOR_BGR2RGB))
