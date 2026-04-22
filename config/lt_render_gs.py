"""
lt_render_gs.py: LT 数据集 7V 相机 Gaussian Splatting 渲染训练配置。

架构：
  2D 特征提取（ResNet-50 + FPN）
  → Gaussian Lifter（初始化 Gaussian 锚点）
  → GaussianOccEncoder（6 层迭代精化，输出携带 16 维语义特征的 3D Gaussians）
  → GaussianRenderHead（调用 SemanticDepthRasterizer 渲染深度图 + 语义图）
  → DLWMLoss（L_d + 0.05 * L_pd + L_sem）

数据集：LTDatasetRender（7V 无鱼眼，16 类语义）
"""

# ================== 数据集与相机 =====================
num_cams = 7
num_classes = 16      # 含 empty_label=0
empty_label = 0

# ================== 模型超参数 ========================
embed_dims = 128
num_groups = 4
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True     # 需要 model/encoder/.../ops/setup.py 已编译
num_levels = 4
drop_out = 0.1

# 点云范围（笛卡尔坐标系，单位：米）
# [x_min, y_min, z_min, x_max, y_max, z_max]
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
scale_range = [0.05, 5.0]

# Gaussian 属性配置
include_opa = True
semantics = True
semantic_dim = num_classes      # 16 维语义特征（无激活，raw logits）
xyz_activation = 'sigmoid'
scale_activation = 'sigmoid'

# ================== 模型配置 ==========================
model = dict(
    type='RenderSegmentor',

    # ── 2D Backbone：ResNet-50 ─────────────────────────
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        pretrained='ckpt/resnet50-19c8e357.pth',
    ),

    # ── 2D Neck：FPN ───────────────────────────────────
    img_neck=dict(
        type='FPN',
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),

    # ── Gaussian Lifter：学习式 Gaussian 锚点初始化 ────
    lifter=dict(
        type='GaussianLifter',
        num_anchor=6400,             # 初始 Gaussian 数量
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        xyz_activation=xyz_activation,
        scale_activation=scale_activation,
    ),

    # ── GaussianOccEncoder：迭代精化 Gaussians ─────────
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim,
        ),
        norm_layer=dict(type='LN', normalized_shape=embed_dims),
        ffn=dict(
            type='AsymmetricFFN',
            in_channels=embed_dims * 2,
            pre_norm=dict(type='LN'),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type='ReLU', inplace=True),
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,              # 7V
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode='cat',
            kps_generator=dict(
                type='SparseGaussian3DKeyPointsGenerator',
                embed_dims=embed_dims,
                num_learnable_pts=6,
                fix_scale=[
                    [0,    0,    0   ],
                    [0.45, 0,    0   ],
                    [-0.45, 0,   0   ],
                    [0,    0.45, 0   ],
                    [0,    -0.45, 0  ],
                    [0,    0,    0.45],
                    [0,    0,    -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range,
                # 笛卡尔坐标系，xyz_coordinate='cartesian'
                xyz_coordinate='cartesian',
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=False,
            unit_xyz=None,
            refine_manual=list(range(10 + int(include_opa))),  # xyz+scale+rot+opa
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            # !! 关键：保留原始 logits，渲染后再接 Softmax
            semantics_activation='none',
            xyz_activation=xyz_activation,
            scale_activation=scale_activation,
        ),
        spconv_layer=None,
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=None,
    ),

    # ── GaussianRenderHead：渲染头 ──────────────────────
    head=dict(
        type='GaussianRenderHead',
        rasterizer_cfg=dict(
            type='SemanticDepthRasterizer',
            num_semantic_classes=num_classes,  # 16
            near=0.1,
            far=100.0,
            backend='auto',       # 自动检测：有 gsplat 用 gsplat，否则用 pytorch
            max_radius_px=32,
        ),
        num_classes=num_classes,
        apply_loss_type='all',    # 所有解码器层均计算渲染 Loss
    ),
)

# ================== 损失函数 ==========================
loss = dict(
    type='DLWMLoss',
    weight_sparse_depth=1.0,
    weight_dense_depth=0.05,
    weight_semantic=1.0,
    num_classes=num_classes,
    empty_label=empty_label,
    train_classes=None,         # None 表示所有非 empty 类别均参与 Loss
    # 若只训练地面类：train_classes=[1]（根据实际类别 id 调整）
)

# loss_input_convertion 定义 result_dict → loss_input 的字段映射
# 格式：{loss_input_key: result_dict_key}
loss_input_convertion = {
    'depth_pred':       'depth_pred',
    'semantic_pred':    'semantic_pred',
    'sparse_depth_gt':  'sparse_depth_gt',
    'valid_lidar_mask': 'valid_lidar_mask',
    'dense_depth_gt':   'dense_depth_gt',
    'semantic_gt':      'semantic_gt',
}

# ================== 数据集配置 ========================
_base_target_size = (512, 1408)  # (H, W)

train_dataset_config = dict(
    type='LTDatasetRender',
    data_root='/path/to/lt_dataset/train',      # TODO: 替换为实际路径
    target_size=_base_target_size,
    img_ext='.jpg',
    occ_dir='OCC_GT_NPZ',
    sparse_depth_dir='SPARSE_DEPTH',            # 相对于 data_root
    dense_depth_dir='DENSE_DEPTH',
    semantic_2d_dir='SEMANTIC_2D',
    num_classes=num_classes,
    empty_label=empty_label,
    phase='train',
    start_timestep=0,
    end_timestep=-1,
)

val_dataset_config = dict(
    type='LTDatasetRender',
    data_root='/path/to/lt_dataset/val',        # TODO: 替换为实际路径
    target_size=_base_target_size,
    img_ext='.jpg',
    occ_dir='OCC_GT_NPZ',
    sparse_depth_dir='SPARSE_DEPTH',
    dense_depth_dir='DENSE_DEPTH',
    semantic_2d_dir='SEMANTIC_2D',
    num_classes=num_classes,
    empty_label=empty_label,
    phase='val',
    start_timestep=0,
    end_timestep=-1,
)

# ================== DataLoader 配置 ===================
train_loader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=True,
)

val_loader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
)

# ================== 优化器配置 ========================
optimizer = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        },
    ),
)

# ================== 训练调度 ==========================
max_epochs = 24
warmup_iters = 500
min_lr_ratio = 0.1
grad_max_norm = 35
amp = False                # 混合精度训练（建议开启以节省显存：True）
print_freq = 50
eval_every_epochs = 1

# ================== 预训练权重 ========================
load_from = ''             # 可指定预训练 checkpoint 路径

# ================== 其他配置 ==========================
syncBN = True
find_unused_parameters = False
debug_projection_before_train = True
debug_projection_num_samples = 2
debug_projection_num_points = 256
