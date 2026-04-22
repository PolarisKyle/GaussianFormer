import os
import glob
import yaml
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import Counter

# ==========================================
# 1. 核心参数配置
# ==========================================
data_root = "/J6P-perception/advc/test/wzx_gt_traj_data/2042440754647289856/LGB810008_20260118T150624_E0H5QH/2042265419897131009/"
lidar_dir = "LIDAR_RECONSTRUCT_ROI/undistort_pcd/LGB810008_20260118T150624_E0H5QH/2042265419897131009/SLAM_Output/undistort_results/undistort_pcd/"
PCD_DIR = os.path.join(data_root, lidar_dir)  
OUTPUT_DIR = os.path.join(data_root, "OCC_GT_NPZ")          
IMG_DIR = os.path.join(data_root, "JPG_2K_CAM_FN")          
CALIB_DIR = os.path.join(data_root, "calib_param") # 标定文件目录

VOXEL_SIZE = np.array([0.5, 0.5, 0.5])
PC_RANGE = np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
GRID_SIZE = np.round((PC_RANGE[3:] - PC_RANGE[:3]) / VOXEL_SIZE).astype(int)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 11V 相机映射表
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

# ==========================================
# 2. YAML 解析与 11V Mask 计算 (新增核心逻辑)
# ==========================================
def load_lt_yaml(yaml_path):
    """解析带有 %YAML:1.0 头的标定文件"""
    with open(yaml_path, 'r') as f:
        lines = f.readlines()[1:] # 跳过 %YAML:1.0
        data = yaml.safe_load("".join(lines))
    
    cam_data = data['camera']
    intrinsic = np.array(cam_data['intrinsic']).reshape(4, 4)[:3, :3]
    
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = np.array(cam_data['cam2ego_R']).reshape(3, 3)
    cam2ego[:3, 3] = np.array(cam_data['cam2ego_t']).reshape(3)
    
    return {
        'intrinsic': intrinsic,
        'ego2cam': np.linalg.inv(cam2ego), # 需要自车到相机的逆矩阵进行投影
        'width': cam_data['image_width'][0],
        'height': cam_data['image_height'][0]
    }

def generate_11v_occ_mask(occ_xyz, calib_dir, sensor_mapping):
    """静态计算 11 路相机的 3D 覆盖掩码"""
    W, H, D, _ = occ_xyz.shape
    occ_cam_mask = np.zeros((W, H, D), dtype=bool)
    
    pts_ego = occ_xyz.reshape(-1, 3)
    pts_ego_homo = np.concatenate([pts_ego, np.ones((pts_ego.shape[0], 1))], axis=-1)
    
    print("正在根据 YAML 标定计算 11V 全景视锥掩码...")
    for cam_dir_name, yml_prefix in tqdm(sensor_mapping.items(), desc="Projecting 11 Cameras"):
        yaml_path = os.path.join(calib_dir, f"{yml_prefix}_params.yml")
        if not os.path.exists(yaml_path):
            print(f"  [警告] 找不到标定文件: {yaml_path}，跳过此机位。")
            continue
            
        cam_cfg = load_lt_yaml(yaml_path)
        
        # 1. 坐标系转换 (Ego -> Cam)
        pts_cam = (cam_cfg['ego2cam'] @ pts_ego_homo.T).T
        # 2. 剔除相机后方的点
        valid_depth = pts_cam[:, 2] > 0.1
        
        # 3. 投影到 2D 像素平面
        pts_cam_valid = pts_cam[valid_depth]
        pts_img = (cam_cfg['intrinsic'] @ pts_cam_valid[:, :3].T).T
        u = pts_img[:, 0] / pts_img[:, 2]
        v = pts_img[:, 1] / pts_img[:, 2]
        
        # 4. 判断是否在有效像素范围内
        in_fov = (u >= 0) & (u < cam_cfg['width']) & (v >= 0) & (v < cam_cfg['height'])
        
        # 5. 更新 Mask
        current_mask = np.zeros(pts_ego.shape[0], dtype=bool)
        current_mask[valid_depth] = in_fov
        occ_cam_mask = occ_cam_mask | current_mask.reshape(W, H, D)
        
    return occ_cam_mask

# ==========================================
# 3. 时间戳同步与占据栅格生成
# ==========================================
def build_timestamp_index(pcd_dir, img_dir):
    print("正在建立传感器时间戳同步索引...")
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    cam_timestamps = np.array([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in img_files])
    sync_mapping = {}
    
    for pcd_path in pcd_files:
        lidar_ts_ns = int(os.path.basename(pcd_path).split('.')[0])
        lidar_ts_ms = lidar_ts_ns // 1000000
        
        time_diffs = np.abs(cam_timestamps - lidar_ts_ms)
        best_match_idx = np.argmin(time_diffs)
        
        sync_mapping[lidar_ts_ns] = {
            'pcd_path': pcd_path,
            'matched_cam_path': img_files[best_match_idx],
            'time_diff_ms': time_diffs[best_match_idx]
        }
    print(f"成功同步 {len(sync_mapping)} 帧数据。")
    return sync_mapping

def generate_occ_xyz(pc_range, voxel_size, grid_size):
    x = np.linspace(pc_range[0] + voxel_size[0]/2, pc_range[3] - voxel_size[0]/2, grid_size[0])
    y = np.linspace(pc_range[1] + voxel_size[1]/2, pc_range[4] - voxel_size[1]/2, grid_size[1])
    z = np.linspace(pc_range[2] + voxel_size[2]/2, pc_range[5] - voxel_size[2]/2, grid_size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.stack([X, Y, Z], axis=-1)

def load_point_semantic_from_pcd(pcd_tensor):
    labels = pcd_tensor.point['pred'].numpy().reshape(-1).astype(np.int64)
    return labels


def update_class_counter(counter, labels):
    if labels is None or labels.size == 0:
        return 0
    labels = labels.astype(np.int64).reshape(-1)
    valid = labels >= 0
    if np.any(valid):
        uniq, cnt = np.unique(labels[valid], return_counts=True)
        for cls_id, n in zip(uniq.tolist(), cnt.tolist()):
            counter[int(cls_id)] += int(n)
    return int((~valid).sum())


def format_class_counter(counter):
    if len(counter) == 0:
        return "{}"
    items = [f"{int(i)}:{int(counter[i])}" for i in sorted(counter.keys()) if counter[i] > 0]
    if len(items) == 0:
        return "{}"
    return "{" + ", ".join(items) + "}"


def generate_occ_label_semantic(points, point_labels, pc_range, voxel_size, grid_size):
    # Use raw PCD semantics directly; empty voxels are 0 by default.
    occ_label = np.zeros(grid_size, dtype=np.int32)

    in_range = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )
    valid_points = points[in_range]
    if valid_points.shape[0] == 0:
        return occ_label

    voxel_indices = np.floor((valid_points - pc_range[:3]) / voxel_size).astype(np.int32)

    valid_labels = point_labels[in_range]
    sem_mask = (valid_labels >= 0)
    voxel_indices = voxel_indices[sem_mask]
    valid_labels = valid_labels[sem_mask].astype(np.int32)
    if voxel_indices.shape[0] == 0:
        return occ_label

    w, h, d = grid_size
    linear_voxel = voxel_indices[:, 0] * (h * d) + voxel_indices[:, 1] * d + voxel_indices[:, 2]
    voxel_class = np.stack([linear_voxel, valid_labels], axis=1)
    uniq_pair, counts = np.unique(voxel_class, axis=0, return_counts=True)

    voxel_ids = uniq_pair[:, 0]
    class_ids = uniq_pair[:, 1]
    for vid in np.unique(voxel_ids):
        mask = voxel_ids == vid
        best_cls = class_ids[mask][np.argmax(counts[mask])]
        x = vid // (h * d)
        yz = vid % (h * d)
        y = yz // d
        z = yz % d
        occ_label[x, y, z] = np.int32(best_cls)

    return occ_label

# ==========================================
# 4. 主干流水线
# ==========================================
def main():
    sync_mapping = build_timestamp_index(PCD_DIR, IMG_DIR)
    
    # 生成 3D 网格坐标
    print("正在预计算全局 occ_xyz...")
    global_occ_xyz = generate_occ_xyz(PC_RANGE, VOXEL_SIZE, GRID_SIZE)
    
    # 🌟 关键修改：调用 11V 投影函数生成真实 Mask 🌟
    global_occ_cam_mask = generate_11v_occ_mask(global_occ_xyz, CALIB_DIR, SENSOR_TO_PCO_NAME)

    global_class_counter = Counter()
    global_invalid_label_count = 0
    is_first_frame = True
    
    print("开始并行/循环生成每帧的真值文件...")
    for lidar_ts_ns, info in tqdm(sync_mapping.items(), desc="Processing Frames"):
        pcd_path = info['pcd_path']

        pcd_tensor = o3d.t.io.read_point_cloud(pcd_path)
        points = pcd_tensor.point['positions'].numpy()
        point_semantic = load_point_semantic_from_pcd(pcd_tensor)
        
        if len(points) == 0:
            continue

        invalid_count = update_class_counter(global_class_counter, point_semantic)
        global_invalid_label_count += invalid_count
        if is_first_frame:
            print(
                f"[语义统计][首帧] source=pcd_pred, labels={format_class_counter(global_class_counter)}, "
                f"invalid={global_invalid_label_count}"
            )
            is_first_frame = False

        occ_label = generate_occ_label_semantic(
            points,
            point_semantic,
            PC_RANGE,
            VOXEL_SIZE,
            GRID_SIZE,
        )
        
        # 将静态的真实 11V Mask 写入每一个帧的 npz 文件中
        save_path = os.path.join(OUTPUT_DIR, f"{lidar_ts_ns}_occ.npz")
        np.savez_compressed(
            save_path, 
            occ_xyz=global_occ_xyz.astype(np.float32), 
            occ_label=occ_label, 
            occ_cam_mask=global_occ_cam_mask,  # <--- 这里存进去的就是 11V 的真实覆盖范围了
            matched_img_path=info['matched_cam_path']
        )

    print(
        f"[语义统计][全局] source=pcd_pred, labels={format_class_counter(global_class_counter)}, "
        f"invalid={global_invalid_label_count}"
    )

if __name__ == "__main__":
    main()