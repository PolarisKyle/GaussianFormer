"""
LTDataset: 私有数据集加载器，适配 GaussianFormer 模型。

核心特性：
- 以 OCC_GT_NPZ 中的 LiDAR 真值文件作为主时间轴
- 在初始化阶段完成 LiDAR(10Hz) 与多相机(30Hz) 的最近邻时间戳对齐
- 支持按时间范围切片，仅构建指定主帧区间的数据索引
- 支持按标定参数对图像执行去畸变，再进行统一尺寸缩放
- 输出与 GaussianFormer 训练链路兼容的字典格式
"""

import glob
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import Dataset

from . import OPENOCC_DATASET


def _load_calib_yaml(yaml_path: str) -> dict:
    """读取带有 '%YAML:1.0' 文件头的 OpenCV 风格标定 YAML 文件。"""
    assert os.path.isfile(yaml_path), f"标定文件不存在：{yaml_path}"
    with open(yaml_path, 'r', encoding='utf-8') as file:
        raw = file.read()
    cleaned = '\n'.join(
        line for line in raw.splitlines()
        if not line.strip().startswith('%YAML')
    )
    data = yaml.safe_load(cleaned)
    assert data is not None and 'camera' in data, (
        f"YAML 文件格式错误，缺少 'camera' 字段：{yaml_path}"
    )
    return data['camera']


def _parse_intrinsic(cam_dict: dict) -> np.ndarray:
    """从标定字典中提取 4x4 齐次内参矩阵。"""
    intrinsic_flat = np.asarray(cam_dict['intrinsic'], dtype=np.float64)
    assert intrinsic_flat.size == 16, (
        f"intrinsic 长度应为 16，实际为 {intrinsic_flat.size}"
    )
    return intrinsic_flat.reshape(4, 4)

def _parse_distortion(cam_dict: dict) -> np.ndarray:
    """提取去畸变系数，支持 4, 5, 8, 12, 14 个参数的 OpenCV 模型。"""
    distorted = cam_dict.get('distorted')
    
    if distorted is None:
        return np.zeros(5, dtype=np.float64)
        
    coeffs = np.asarray(distorted, dtype=np.float64).reshape(-1)
    
    # OpenCV 常见合法长度: 4, 5, 8, 12, 14
    valid_lengths = [4, 5, 8, 12, 14]
    if coeffs.size in valid_lengths:
        return coeffs
        
    if coeffs.size >= 14:
        return coeffs[:14]
    elif coeffs.size >= 8:
        return coeffs[:8]
    elif coeffs.size >= 5:
        return coeffs[:5]
    
    padded = np.zeros(5, dtype=np.float64)
    padded[:coeffs.size] = coeffs
    return padded

def _parse_cam2ego(cam_dict: dict) -> np.ndarray:
    """从标定字典中提取 4x4 cam2ego 齐次变换矩阵。"""
    rotation_flat = np.asarray(cam_dict['cam2ego_R'], dtype=np.float64)
    translation = np.asarray(cam_dict['cam2ego_t'], dtype=np.float64).reshape(-1)

    assert rotation_flat.size == 9, (
        f"cam2ego_R 长度应为 9，实际为 {rotation_flat.size}"
    )
    assert translation.size == 3, (
        f"cam2ego_t 长度应为 3，实际为 {translation.size}"
    )

    cam2ego = np.eye(4, dtype=np.float64)
    cam2ego[:3, :3] = rotation_flat.reshape(3, 3)
    cam2ego[:3, 3] = translation
    return cam2ego


def _extract_img_timestamp_ms(file_path: str) -> int:
    """从 frame-XXXXXX_<13位毫秒时间戳>.jpg 中解析毫秒时间戳。"""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = stem.rsplit('_', 1)
    assert len(parts) == 2, (
        f"无法从图像文件名 '{stem}' 中提取时间戳，期望格式：frame-XXXXXX_<timestamp>"
    )
    timestamp_ms = int(parts[1])
    assert len(parts[1]) == 13, (
        f"图像时间戳应为 13 位毫秒，实际为 {parts[1]}"
    )
    return timestamp_ms


def _extract_occ_timestamp_ns(file_path: str) -> int:
    """从 {19位纳秒时间戳}_occ.npz 中解析纳秒时间戳。"""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    if stem.endswith('_occ'):
        stem = stem[:-4]
    timestamp_ns = int(stem)
    assert len(str(timestamp_ns)) == 19, (
        f"OCC_GT_NPZ 主帧时间戳应为 19 位纳秒，实际为 {timestamp_ns}"
    )
    return timestamp_ns


_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]


@OPENOCC_DATASET.register_module()
class LTDataset(Dataset):
    """私有多机位数据集，使用 LiDAR 主帧驱动多模态时空对齐。"""

    SENSOR_TO_PCO_NAME = {
        'JPG_2K_CAM_FN': 'cam0',
        'JPG_2K_CAM_FW': 'cam1',
        'JPG_CAM_FL': 'cam4',
        'JPG_CAM_RL': 'cam5',
        'JPG_CAM_RN': 'cam6',
        'JPG_CAM_RR': 'cam7',
        'JPG_CAM_FR': 'cam8',
        # 'JPG_CAM_SFW': 'cam_front_fisheye',
        # 'JPG_CAM_SLW': 'cam_left_fisheye',
        # 'JPG_CAM_SRCW': 'cam_rear_fisheye',
        # 'JPG_CAM_SRW': 'cam_right_fisheye',
    }

    def __init__(
        self,
        data_root: str,
        target_size: tuple = (512, 1408),
        img_ext: str = '.jpg',
        occ_dir: str = 'OCC_GT_NPZ',
        phase: str = 'train',
        start_timestep: int = 0,
        end_timestep: Optional[int] = -1,
        **kwargs,
    ):
        assert os.path.isdir(data_root), f"数据集根目录不存在：{data_root}"
        assert len(target_size) == 2, f"target_size 应为 (H, W)，实际为 {target_size}"
        assert start_timestep >= 0, f"start_timestep 必须 >= 0，实际为 {start_timestep}"
        assert isinstance(occ_dir, str) and occ_dir, f"occ_dir 必须是非空字符串，实际为 {occ_dir}"

        self.data_root = data_root
        self.target_size = tuple(target_size)
        self.img_ext = img_ext
        self.occ_dir = occ_dir
        self.phase = phase
        self.extra_kwargs = kwargs

        self.cams = [
            folder_name for folder_name in self.SENSOR_TO_PCO_NAME
            if os.path.isdir(os.path.join(self.data_root, folder_name))
        ]
        assert self.cams, (
            f"在 {self.data_root} 下未找到任何已配置相机目录，"
            f"支持目录：{list(self.SENSOR_TO_PCO_NAME.keys())}"
        )

        self.calibs = self._load_all_calibs()
        self.cam_samples = self._build_camera_samples()
        self.data_infos = self._build_data_infos(
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )

    def _load_all_calibs(self) -> Dict[str, dict]:
        calib_dir = os.path.join(self.data_root, 'calib_param')
        assert os.path.isdir(calib_dir), f"标定参数目录不存在：{calib_dir}"

        calibs: Dict[str, dict] = {}
        for cam in self.cams:
            pco_name = self.SENSOR_TO_PCO_NAME[cam]
            yaml_path = self._resolve_calib_yaml_path(calib_dir, pco_name)
            cam_dict = _load_calib_yaml(yaml_path)
            calibs[cam] = {
                'yaml_path': yaml_path,
                'K': _parse_intrinsic(cam_dict),
                'D': _parse_distortion(cam_dict),
                'cam2ego': _parse_cam2ego(cam_dict),
                'orig_H': int(cam_dict['image_height'][0])
                if isinstance(cam_dict['image_height'], (list, tuple))
                else int(cam_dict['image_height']),
                'orig_W': int(cam_dict['image_width'][0])
                if isinstance(cam_dict['image_width'], (list, tuple))
                else int(cam_dict['image_width']),
            }
        return calibs

    def _build_camera_samples(self) -> Dict[str, dict]:
        cam_samples: Dict[str, dict] = {}
        for cam in self.cams:
            cam_dir = os.path.join(self.data_root, cam)
            image_paths = sorted(glob.glob(os.path.join(cam_dir, f'*{self.img_ext}')))
            assert image_paths, f"相机 {cam} 目录下未找到扩展名为 {self.img_ext} 的图像文件"

            timestamps_ms = np.asarray(
                [_extract_img_timestamp_ms(path) for path in image_paths],
                dtype=np.int64,
            )
            assert np.all(timestamps_ms[:-1] <= timestamps_ms[1:]), (
                f"相机 {cam} 的图像时间戳未按升序排列，请检查文件命名"
            )
            cam_samples[cam] = {
                'timestamps_ms': timestamps_ms,
                'paths': image_paths,
            }
        return cam_samples

    def _build_data_infos(
        self,
        start_timestep: int,
        end_timestep: Optional[int],
    ) -> List[dict]:
        occ_dir = self.occ_dir
        if not os.path.isabs(occ_dir):
            occ_dir = os.path.join(self.data_root, occ_dir)
        assert os.path.isdir(occ_dir), f"OCC_GT_NPZ 目录不存在：{occ_dir}"

        occ_paths = sorted(glob.glob(os.path.join(occ_dir, '*_occ.npz')))
        assert occ_paths, f"在 {occ_dir} 下未找到任何 *_occ.npz 文件"

        occ_paths = sorted(occ_paths, key=_extract_occ_timestamp_ns)
        sliced_occ_paths = self._slice_occ_paths(occ_paths, start_timestep, end_timestep)
        assert sliced_occ_paths, (
            f"切片后没有可用主帧，请检查 start_timestep={start_timestep}, end_timestep={end_timestep}"
        )

        data_infos: List[dict] = []
        for occ_path in sliced_occ_paths:
            lidar_ts_ns = _extract_occ_timestamp_ns(occ_path)
            lidar_ts_ms = lidar_ts_ns // 1_000_000
            camera_paths: Dict[str, str] = {}
            time_diffs_ms: Dict[str, int] = {}

            for cam in self.cams:
                cam_timestamps_ms = self.cam_samples[cam]['timestamps_ms']
                diffs = np.abs(cam_timestamps_ms - lidar_ts_ms)
                nearest_idx = int(np.argmin(diffs))
                camera_paths[cam] = self.cam_samples[cam]['paths'][nearest_idx]
                time_diffs_ms[cam] = int(diffs[nearest_idx])

            data_infos.append({
                'occ_path': occ_path,
                'lidar_timestamp_ns': lidar_ts_ns,
                'lidar_timestamp_ms': lidar_ts_ms,
                'camera_paths': camera_paths,
                'time_diffs_ms': time_diffs_ms,
            })

        return data_infos

    @staticmethod
    def _slice_occ_paths(
        occ_paths: List[str],
        start_timestep: int,
        end_timestep: Optional[int],
    ) -> List[str]:
        total = len(occ_paths)
        if end_timestep in (-1, None):
            end_index = total
        else:
            end_index = end_timestep
        assert start_timestep <= total, (
            f"start_timestep 超出范围：{start_timestep} > {total}"
        )
        assert end_index <= total, (
            f"end_timestep 超出范围：{end_index} > {total}"
        )
        assert start_timestep <= end_index, (
            f"切片范围非法：start_timestep={start_timestep}, end_timestep={end_index}"
        )
        return occ_paths[start_timestep:end_index]

    @staticmethod
    def _resolve_calib_yaml_path(calib_dir: str, pco_name: str) -> str:
        candidates = [
            f'{pco_name}_params.yml',
            f'{pco_name}.yml',
            f'{pco_name}_params.yaml',
            f'{pco_name}.yaml',
        ]
        for name in candidates:
            path = os.path.join(calib_dir, name)
            if os.path.isfile(path):
                return path
        raise AssertionError(
            f"未找到相机 {pco_name} 的标定文件，已尝试：{candidates}"
        )

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, index: int) -> dict:
        assert 0 <= index < len(self), f"索引越界：index={index}, 数据集长度={len(self)}"

        info = self.data_infos[index]
        target_h, target_w = self.target_size

        with np.load(info['occ_path']) as occ_data:
            occ_xyz = torch.from_numpy(occ_data['occ_xyz']).float()
            occ_label = torch.from_numpy(occ_data['occ_label']).long()
            occ_cam_mask = torch.from_numpy(occ_data['occ_cam_mask']).bool()

        imgs_list = []
        ori_imgs_list = []
        intrinsics_list = []
        cam2ego_list = []
        projection_list = []
        image_wh_list = []
        cam_positions_list = []
        focal_positions_list = []
        focal_depth = 0.0055

        for cam in self.cams:
            img_path = info['camera_paths'][cam]
            assert os.path.isfile(img_path), f"图像文件不存在：{img_path}"

            with Image.open(img_path) as raw_img:
                pil_img = raw_img.convert('RGB')
                orig_w, orig_h = pil_img.size

            pil_img = self._undistort_image(pil_img, cam)
            # Keep an undistorted original image for visualization.
            ori_imgs_list.append(np.asarray(pil_img)[..., ::-1].copy())
            pil_img = TF.resize(pil_img, [target_h, target_w])

            img_tensor = TF.to_tensor(pil_img)
            img_tensor = TF.normalize(img_tensor, mean=_IMG_MEAN, std=_IMG_STD)
            imgs_list.append(img_tensor)

            scale_w = target_w / orig_w
            scale_h = target_h / orig_h

            intrinsic = self.calibs[cam]['K'].copy()
            intrinsic[0, 0] *= scale_w
            intrinsic[0, 2] *= scale_w
            intrinsic[1, 1] *= scale_h
            intrinsic[1, 2] *= scale_h
            intrinsics_list.append(torch.from_numpy(intrinsic).float())

            cam2ego = self.calibs[cam]['cam2ego']
            cam2ego_list.append(torch.from_numpy(cam2ego).float())

            ego2cam = np.linalg.inv(cam2ego)
            projection = intrinsic @ ego2cam
            projection_list.append(torch.from_numpy(projection).float())

            image_wh_list.append(torch.tensor([target_w, target_h], dtype=torch.float32))

            cam_position = (cam2ego @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))[:3]
            focal_position = (
                cam2ego @ np.array([0.0, 0.0, focal_depth, 1.0], dtype=np.float64)
            )[:3]
            cam_positions_list.append(torch.from_numpy(cam_position).float())
            focal_positions_list.append(torch.from_numpy(focal_position).float())

        imgs = torch.stack(imgs_list, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)
        cam2ego = torch.stack(cam2ego_list, dim=0)
        projection_mat = torch.stack(projection_list, dim=0)
        image_wh = torch.stack(image_wh_list, dim=0)
        cam_positions = torch.stack(cam_positions_list, dim=0)
        focal_positions = torch.stack(focal_positions_list, dim=0)

        num_cams = len(self.cams)
        assert imgs.shape == (num_cams, 3, target_h, target_w), f"imgs 形状异常：{imgs.shape}"
        assert intrinsics.shape == (num_cams, 4, 4), f"intrinsics 形状异常：{intrinsics.shape}"
        assert cam2ego.shape == (num_cams, 4, 4), f"cam2ego 形状异常：{cam2ego.shape}"

        return {
            'img': imgs,
            'ori_img': ori_imgs_list,
            'intrinsics': intrinsics,
            'cam2ego': cam2ego,
            'projection_mat': projection_mat,
            'image_wh': image_wh,
            'cam_positions': cam_positions,
            'focal_positions': focal_positions,
            'occ_xyz': occ_xyz,
            'occ_label': occ_label,
            'occ_cam_mask': occ_cam_mask,
            'frame_id': str(info['lidar_timestamp_ns']),
            'lidar_timestamp_ns': info['lidar_timestamp_ns'],
            'lidar_timestamp_ms': info['lidar_timestamp_ms'],
            'time_diffs_ms': info['time_diffs_ms'],
        }

    def _undistort_image(self, pil_img: Image.Image, cam: str) -> Image.Image:
        """根据 yml 中的 intrinsic + distorted 参数进行去畸变。"""
        calib = self.calibs[cam]
        distortion = calib['D']
        if distortion.size == 0 or not np.any(distortion != 0):
            return pil_img

        intrinsic_3x3 = calib['K'][:3, :3]
        rgb_image = np.asarray(pil_img)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        undistorted_bgr = cv2.undistort(
            bgr_image,
            intrinsic_3x3,
            distortion,
            None,
            intrinsic_3x3,
        )
        undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(undistorted_rgb)
