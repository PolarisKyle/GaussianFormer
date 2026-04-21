"""
LTDataset: 私有数据集加载器，适配 GaussianFormer 模型。

核心特性：
- 支持多机位、异构分辨率相机（前视主摄、侧视广角、鱼眼等）
- 强制 Resize 到统一目标尺寸，并同步修正相机内参矩阵
- 输出与 NuScenes 管线兼容的字典格式
"""

import os
import glob
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import yaml

from . import OPENOCC_DATASET


# ---------------------------------------------------------------------------
# YAML 解析辅助函数
# ---------------------------------------------------------------------------

def _load_calib_yaml(yaml_path: str) -> dict:
    """
    读取带有 '%YAML:1.0' 文件头的 OpenCV 风格标定 YAML 文件。

    Args:
        yaml_path: .yml 文件的绝对路径。

    Returns:
        解析后的 Python 字典。
    """
    assert os.path.isfile(yaml_path), f"标定文件不存在：{yaml_path}"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # 去掉 OpenCV YAML 特有的版本声明行，使标准 PyYAML 可正常解析
    cleaned = '\n'.join(
        line for line in raw.splitlines()
        if not line.strip().startswith('%YAML')
    )
    data = yaml.safe_load(cleaned)
    assert data is not None and 'camera' in data, \
        f"YAML 文件格式错误，缺少 'camera' 字段：{yaml_path}"
    return data['camera']


def _parse_intrinsic(cam_dict: dict) -> np.ndarray:
    """
    从标定字典中提取内参，构造 4×4 齐次内参矩阵 K。

    期望 cam_dict['intrinsic'] 是长度为 16 的列表（行优先 4×4 矩阵）。

    Returns:
        K (np.ndarray): shape (4, 4), dtype float64
    """
    intrinsic_flat = np.array(cam_dict['intrinsic'], dtype=np.float64)
    assert intrinsic_flat.size == 16, \
        f"intrinsic 长度应为 16，实际为 {intrinsic_flat.size}"
    K = intrinsic_flat.reshape(4, 4)
    return K


def _parse_sensor2ego(cam_dict: dict) -> np.ndarray:
    """
    从标定字典中提取旋转矩阵 R 和平移向量 t，
    拼装为 4×4 齐次变换矩阵 sensor2ego（相机坐标系 → 车体坐标系）。

    期望：
        cam_dict['cam2ego_R']: 长度为 9 的列表，行优先 3×3 旋转矩阵
        cam_dict['cam2ego_t']: 长度为 3 的列表，[x, y, z] 平移向量

    Returns:
        T (np.ndarray): shape (4, 4), dtype float64
    """
    R_flat = np.array(cam_dict['cam2ego_R'], dtype=np.float64)
    t = np.array(cam_dict['cam2ego_t'], dtype=np.float64).flatten()

    assert R_flat.size == 9, \
        f"cam2ego_R 长度应为 9，实际为 {R_flat.size}"
    assert t.size == 3, \
        f"cam2ego_t 长度应为 3，实际为 {t.size}"

    R = R_flat.reshape(3, 3)

    # 拼装 4×4 齐次变换矩阵
    # [ R  t ]
    # [ 0  1 ]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# 图像归一化参数（ImageNet 均值/标准差，与 NuScenes 管线保持一致）
# ---------------------------------------------------------------------------
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# LTDataset 主类
# ---------------------------------------------------------------------------

@OPENOCC_DATASET.register_module()
class LTDataset(Dataset):
    """
    私有数据集加载器，支持多机位异构分辨率图像的统一化处理。

    目录结构示例::

        data_root/
        ├── calib_param/
        │   ├── cam0_params.yml   # CAM_FN 对应的标定文件
        │   ├── cam1_params.yml
        │   └── ...
        ├── JPG_2K_CAM_FN/        # 前视主摄图像文件夹
        │   ├── 000000.jpg
        │   └── ...
        ├── JPG_CAM_FL/           # 左前视图像文件夹
        └── ...

    Args:
        data_root (str): 数据集根目录路径。
        target_size (tuple[int, int]): 目标图像尺寸 (H, W)，默认 (512, 1408)。
        img_ext (str): 图像文件扩展名，默认 '.jpg'。
        phase (str): 'train' 或 'val'/'test'，当前保留供扩展使用。
    """

    # ------------------------------------------------------------------
    # 相机逻辑名列表（类似 NuScenes sensor_types）
    # ------------------------------------------------------------------
    CAMERA_NAMES = [
        'CAM_FN',    # 前视主摄 (Front Narrow)
        'CAM_FL',    # 左前视 (Front Left)
        'CAM_FR',    # 右前视 (Front Right)
        'CAM_SLW',   # 侧左广角 (Side Left Wide)
        'CAM_SRW',   # 侧右广角 (Side Right Wide)
        'CAM_BN',    # 后视 (Back Narrow)
    ]

    # ------------------------------------------------------------------
    # 相机逻辑名 → (图像文件夹名, YAML 标定文件名) 的映射
    # 请根据实际数据集目录结构修改此映射
    # ------------------------------------------------------------------
    CAM_CONFIG = {
        'CAM_FN':  ('JPG_2K_CAM_FN',  'cam0_params.yml'),
        'CAM_FL':  ('JPG_CAM_FL',      'cam1_params.yml'),
        'CAM_FR':  ('JPG_CAM_FR',      'cam2_params.yml'),
        'CAM_SLW': ('JPG_CAM_SLW',     'cam3_params.yml'),
        'CAM_SRW': ('JPG_CAM_SRW',     'cam4_params.yml'),
        'CAM_BN':  ('JPG_CAM_BN',      'cam5_params.yml'),
    }

    def __init__(
        self,
        data_root: str,
        target_size: tuple = (512, 1408),
        img_ext: str = '.jpg',
        phase: str = 'train',
    ):
        """
        初始化数据集。

        Args:
            data_root: 数据集根目录，必须存在。
            target_size: 统一目标尺寸 (H, W)。
            img_ext: 图像后缀，如 '.jpg' 或 '.png'。
            phase: 训练/验证阶段标识（保留参数）。
        """
        assert os.path.isdir(data_root), f"数据集根目录不存在：{data_root}"
        self.data_root = data_root
        self.target_size = target_size  # (H, W)
        self.img_ext = img_ext
        self.phase = phase
        self.cams = self.CAMERA_NAMES

        # ------------------------------------------------------------------
        # 预加载所有相机的标定参数（内参 K 和外参 sensor2ego）
        # ------------------------------------------------------------------
        self.calibs: Dict[str, dict] = {}
        calib_dir = os.path.join(data_root, 'calib_param')
        assert os.path.isdir(calib_dir), \
            f"标定参数目录不存在：{calib_dir}"

        for cam in self.cams:
            _, yaml_name = self.CAM_CONFIG[cam]
            yaml_path = os.path.join(calib_dir, yaml_name)
            cam_dict = _load_calib_yaml(yaml_path)
            self.calibs[cam] = {
                'K': _parse_intrinsic(cam_dict),          # (4, 4)
                'sensor2ego': _parse_sensor2ego(cam_dict), # (4, 4)
                'orig_H': int(cam_dict['image_height'][0])
                    if isinstance(cam_dict['image_height'], (list, tuple))
                    else int(cam_dict['image_height']),
                'orig_W': int(cam_dict['image_width'][0])
                    if isinstance(cam_dict['image_width'], (list, tuple))
                    else int(cam_dict['image_width']),
            }

        # ------------------------------------------------------------------
        # 构建帧索引：以第一个相机目录中的有序文件列表为基准
        # ------------------------------------------------------------------
        ref_cam = self.cams[0]
        ref_img_dir = os.path.join(data_root, self.CAM_CONFIG[ref_cam][0])
        assert os.path.isdir(ref_img_dir), \
            f"参考相机图像目录不存在：{ref_img_dir}"

        pattern = os.path.join(ref_img_dir, f'*{self.img_ext}')
        self.frame_ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(pattern)
        )
        assert len(self.frame_ids) > 0, \
            f"未在 {ref_img_dir} 中找到扩展名为 {self.img_ext} 的图像文件"

    # ------------------------------------------------------------------
    # Dataset 协议方法
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> dict:
        """
        加载第 index 帧的多视角图像及对应标定参数。

        Returns:
            dict 包含:
                imgs (torch.Tensor): shape (N, 3, target_H, target_W)，
                    已归一化（ImageNet 均值/方差）。
                intrinsics (torch.Tensor): shape (N, 4, 4)，
                    已根据 Resize 缩放比例修正。
                sensor2ego (torch.Tensor): shape (N, 4, 4)，
                    相机坐标系到车体坐标系的齐次变换矩阵。
                frame_id (str): 当前帧的文件名（不含扩展名）。
        """
        assert 0 <= index < len(self), \
            f"索引越界：index={index}, 数据集长度={len(self)}"

        frame_id = self.frame_ids[index]
        target_H, target_W = self.target_size

        imgs_list = []
        intrinsics_list = []
        sensor2ego_list = []

        for cam in self.cams:
            img_dir, _ = self.CAM_CONFIG[cam]
            img_path = os.path.join(
                self.data_root, img_dir, f'{frame_id}{self.img_ext}'
            )
            assert os.path.isfile(img_path), \
                f"图像文件不存在：{img_path}"

            # ----------------------------------------------------------
            # 1. 读取原始图像
            # ----------------------------------------------------------
            with Image.open(img_path) as raw_img:
                pil_img = raw_img.convert('RGB')
                orig_W, orig_H = pil_img.size  # PIL.size 返回 (W, H)

            # ----------------------------------------------------------
            # 2. Resize 到目标尺寸
            # ----------------------------------------------------------
            # torchvision resize 接收 (H, W) 顺序
            pil_img = TF.resize(pil_img, [target_H, target_W])

            # ----------------------------------------------------------
            # 3. 图像转 Tensor 并归一化（CHW, float32, [0,1] → 归一化）
            # ----------------------------------------------------------
            img_tensor = TF.to_tensor(pil_img)             # (3, H, W) float32
            img_tensor = TF.normalize(
                img_tensor, mean=_IMG_MEAN, std=_IMG_STD
            )
            imgs_list.append(img_tensor)

            # ----------------------------------------------------------
            # 4. 内参随 Resize 等比例缩放（核心！）
            #    scale_w = target_W / orig_W  ← 水平方向缩放比
            #    scale_h = target_H / orig_H  ← 垂直方向缩放比
            #
            #    需要修正的内参分量：
            #      K[0, 0] = fx  →  fx * scale_w
            #      K[0, 2] = cx  →  cx * scale_w
            #      K[1, 1] = fy  →  fy * scale_h
            #      K[1, 2] = cy  →  cy * scale_h
            #    其余分量（skew、齐次行）保持不变。
            # ----------------------------------------------------------
            scale_w = target_W / orig_W
            scale_h = target_H / orig_H

            K = self.calibs[cam]['K'].copy()  # (4, 4) float64，深拷贝避免污染

            # 水平方向相关的内参分量乘以 scale_w
            K[0, 0] *= scale_w   # fx
            K[0, 2] *= scale_w   # cx

            # 垂直方向相关的内参分量乘以 scale_h
            K[1, 1] *= scale_h   # fy
            K[1, 2] *= scale_h   # cy

            intrinsics_list.append(
                torch.from_numpy(K).float()  # (4, 4)
            )

            # ----------------------------------------------------------
            # 5. 外参矩阵（sensor2ego）无需随图像缩放修改，直接复用
            # ----------------------------------------------------------
            T = self.calibs[cam]['sensor2ego']  # (4, 4) float64
            sensor2ego_list.append(
                torch.from_numpy(T).float()  # (4, 4)
            )

        # ------------------------------------------------------------------
        # 6. 将各相机数据堆叠为批量张量
        # ------------------------------------------------------------------
        imgs = torch.stack(imgs_list, dim=0)           # (N, 3, H, W)
        intrinsics = torch.stack(intrinsics_list, dim=0)  # (N, 4, 4)
        sensor2ego = torch.stack(sensor2ego_list, dim=0)  # (N, 4, 4)

        N = len(self.cams)
        assert imgs.shape == (N, 3, target_H, target_W), \
            f"imgs 形状异常：{imgs.shape}"
        assert intrinsics.shape == (N, 4, 4), \
            f"intrinsics 形状异常：{intrinsics.shape}"
        assert sensor2ego.shape == (N, 4, 4), \
            f"sensor2ego 形状异常：{sensor2ego.shape}"

        return {
            'imgs': imgs,           # (N, 3, target_H, target_W)
            'intrinsics': intrinsics,  # (N, 4, 4)
            'sensor2ego': sensor2ego,  # (N, 4, 4)
            'frame_id': frame_id,
        }
