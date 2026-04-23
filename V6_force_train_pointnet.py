# -*- coding: utf-8 -*-
"""
V6_force_train_pointnet.py — 基于 PointNet 的触觉传感器力标定训练
输入: 60个marker点的位移 dxyz (T,60,3)
输出: 六维力 force (T,6) → fx,fy,fz,mx,my,mz
"""
import os
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


# ─────────────────────── T-Net (Input/Feature Transform) ───────────────────────

class TNet(nn.Module):
    """Transformation Network for input or feature alignment"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 初始化为单位矩阵
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).flatten().unsqueeze(0)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# ─────────────────────── PointNet Backbone ───────────────────────

class PointNetBackbone(nn.Module):
    """PointNet feature extractor"""
    def __init__(self, use_input_transform=True, use_feature_transform=True):
        super().__init__()
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

        # Input transform
        if use_input_transform:
            self.input_transform = TNet(k=3)

        # Shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        # Feature transform
        if use_feature_transform:
            self.feature_transform = TNet(k=128)

        # Shared MLP 2
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) point cloud
        Returns:
            global_feat: (B, 2048) global feature
            trans_feat: (B, 128, 128) feature transform matrix (for regularization)
        """
        batch_size, _, n_pts = x.size()

        # Input transform
        if self.use_input_transform:
            trans_input = self.input_transform(x)  # (B, 3, 3)
            x = torch.bmm(trans_input, x)

        # Shared MLP 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        trans_feat = None
        if self.use_feature_transform:
            trans_feat = self.feature_transform(x)  # (B, 128, 128)
            x = torch.bmm(trans_feat, x)

        # Shared MLP 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling
        global_feat = torch.max(x, 2, keepdim=False)[0]  # (B, 2048)

        return global_feat, trans_feat


# ─────────────────────── PointNet for Regression ───────────────────────

class ForcePointNet(nn.Module):
    """PointNet for force prediction"""
    def __init__(self, output_dim=6, use_input_transform=True, use_feature_transform=True):
        super().__init__()
        self.output_dim = output_dim
        self.use_feature_transform = use_feature_transform

        self.backbone = PointNetBackbone(use_input_transform, use_feature_transform)

        # Regression head
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) or (B, 3, N) point cloud
        Returns:
            out: (B, output_dim) predicted force
            trans_feat: feature transform matrix for regularization
        """
        if x.dim() == 3 and x.size(2) == 3:
            x = x.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)

        global_feat, trans_feat = self.backbone(x)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, trans_feat

    def feature_transform_regularizer(self, trans_feat):
        """Regularization loss for feature transform matrix"""
        if trans_feat is None:
            return 0
        d = trans_feat.size(1)
        I = torch.eye(d, device=trans_feat.device).unsqueeze(0)
        loss = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I, dim=(1, 2)))
        return loss


# ─────────────────────── 数据集 ───────────────────────

class ForceDataset(Dataset):
    """从多个 processed HDF5 文件加载 dxyz → force 数据"""

    def __init__(self, h5_files, force_dims=6):
        self.X = []  # dxyz (N, 60, 3)
        self.y = []  # force
        self.file_info = []
        self.segment_info = []  # 记录每个segment的起止索引: [(start_idx, end_idx, file_idx), ...]

        for file_idx, fpath in enumerate(h5_files):
            with h5py.File(fpath, 'r') as f:
                if 'vision/dxyz' not in f or 'force/values' not in f:
                    print(f"[跳过] {fpath}: 缺少数据")
                    continue

                dxyz = f['vision/dxyz'][:]
                force = f['force/values'][:]
                T = min(dxyz.shape[0], force.shape[0])
                dxyz = dxyz[:T].astype(np.float32)
                force = force[:T, :force_dims].astype(np.float32)

                # 读取segments信息
                segments = []
                if 'segments' in f:
                    sg = f['segments']
                    count = int(sg.attrs.get('count', 0))
                    for i in range(count):
                        s = float(sg.attrs.get(f'seg_{i}_start', 0))
                        e = float(sg.attrs.get(f'seg_{i}_end', 0))
                        segments.append((s, e))
                    print(f"[加载] {os.path.basename(fpath)}: 找到 {count} 个有效段")

                # 读取时间戳（使用force/timestamp作为基准，因为segment是相对于它的）
                timestamps = None
                if 'force/timestamp' in f:
                    timestamps = f['force/timestamp'][:]
                elif 'vision/timestamp' in f:
                    timestamps = f['vision/timestamp'][:]
                elif 'vision/timestamps' in f:
                    timestamps = f['vision/timestamps'][:]
                elif 'timestamps' in f:
                    timestamps = f['timestamps'][:]

                if timestamps is None:
                    print(f"  [警告] 未找到timestamps，无法进行segment映射")
                else:
                    print(f"  timestamps范围: {timestamps.min():.3f}s ~ {timestamps.max():.3f}s")
                    # segment时间是相对于timestamps[0]的
                    t_rel = timestamps - timestamps[0]

                offset = sum(info[1] for info in self.file_info)
                self.X.append(dxyz)
                self.y.append(force)
                self.file_info.append((os.path.basename(fpath), T, offset))

                # 记录每个segment对应的帧索引范围
                if segments and timestamps is not None:
                    for seg_idx, (seg_start_time, seg_end_time) in enumerate(segments):
                        # 找到时间戳在segment范围内的帧（使用相对时间）
                        mask = (t_rel >= seg_start_time) & (t_rel <= seg_end_time)
                        indices = np.where(mask)[0]
                        if len(indices) > 0:
                            start_idx = offset + indices[0]
                            end_idx = offset + indices[-1] + 1
                            self.segment_info.append((start_idx, end_idx, file_idx))
                        else:
                            print(f"  [警告] Segment {seg_idx} ({seg_start_time:.3f}~{seg_end_time:.3f}s) 未找到匹配帧")

                print(f"[加载] {os.path.basename(fpath)}: {T} 帧")

        if len(self.X) == 0:
            raise RuntimeError("未加载到任何有效数据")

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        print(f"\n总样本数: {len(self.X)}")
        print(f"输入shape: {self.X.shape}, 输出shape: {self.y.shape}")
        print(f"力范围: {self.y.min(axis=0)} ~ {self.y.max(axis=0)}")
        print(f"总segment数: {len(self.segment_info)}")

    def get_segment_based_split(self, train_ratio=0.8, group_size=10, min_frames_for_split=100):
        """基于segment划分训练集和验证集

        Args:
            train_ratio: 训练集比例（默认0.8，即每10个segment前8个训练）
            group_size: 分组大小（默认10）
            min_frames_for_split: 单segment内部划分的最小帧数阈值（默认100）

        Returns:
            train_indices: 训练集索引列表
            val_indices: 验证集索引列表
        """
        if len(self.segment_info) == 0:
            print("[警告] 未找到segment信息，使用随机划分")
            total = len(self.X)
            indices = np.arange(total)
            np.random.shuffle(indices)
            split = int(total * train_ratio)
            return indices[:split].tolist(), indices[split:].tolist()

        train_indices = []
        val_indices = []

        # 按group_size分组处理segments
        num_segments = len(self.segment_info)
        train_per_group = int(group_size * train_ratio)

        for group_start in range(0, num_segments, group_size):
            group_end = min(group_start + group_size, num_segments)
            group_segments = self.segment_info[group_start:group_end]

            # 当前组的训练/验证分界点
            train_end_in_group = min(train_per_group, len(group_segments))

            for i, (start_idx, end_idx, file_idx) in enumerate(group_segments):
                seg_frames = end_idx - start_idx

                # 特殊处理：如果整个组只有1个segment且帧数足够，内部划分
                if len(group_segments) == 1 and seg_frames >= min_frames_for_split:
                    split_point = start_idx + int(seg_frames * train_ratio)
                    train_indices.extend(range(start_idx, split_point))
                    val_indices.extend(range(split_point, end_idx))
                    print(f"  [单segment内部划分] Segment {group_start}: "
                          f"训练={split_point-start_idx}帧, 验证={end_idx-split_point}帧")
                else:
                    # 正常按segment整体划分
                    indices = list(range(start_idx, end_idx))
                    if i < train_end_in_group:
                        train_indices.extend(indices)
                    else:
                        val_indices.extend(indices)

        print(f"\n[Segment划分] 总segment数: {num_segments}")
        print(f"  训练集: {len(train_indices)} 帧 ({len(train_indices)/len(self.X)*100:.1f}%)")
        print(f"  验证集: {len(val_indices)} 帧 ({len(val_indices)/len(self.X)*100:.1f}%)")

        return train_indices, val_indices

    def compute_scaler(self, indices=None):
        if indices is None:
            indices = list(range(len(self.X)))
        X_sub = self.X[indices].reshape(len(indices), -1)
        y_sub = self.y[indices]

        self.x_mean = X_sub.mean(axis=0).reshape(60, 3)
        self.x_std = X_sub.std(axis=0).reshape(60, 3)
        self.x_std[self.x_std < 1e-8] = 1.0

        self.y_mean = y_sub.mean(axis=0)
        self.y_std = y_sub.std(axis=0)
        self.y_std[self.y_std < 1e-8] = 1.0

        print(f"Scaler 基于 {len(indices)} 个样本计算")

    def set_scaler(self, scaler):
        self.x_mean = scaler['x_mean']
        self.x_std = scaler['x_std']
        self.y_mean = scaler['y_mean']
        self.y_std = scaler['y_std']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.x_mean is None:
            raise RuntimeError("未设置 scaler")
        x_norm = (self.X[idx] - self.x_mean) / self.x_std
        y_norm = (self.y[idx] - self.y_mean) / self.y_std
        return (torch.tensor(x_norm, dtype=torch.float32),
                torch.tensor(y_norm, dtype=torch.float32))

    def get_scaler(self):
        if self.x_mean is None:
            raise RuntimeError("未设置 scaler")
        return {
            'x_mean': self.x_mean, 'x_std': self.x_std,
            'y_mean': self.y_mean, 'y_std': self.y_std,
        }


# ─────────────────────── 数据增强 ───────────────────────

class AugmentedSubset(Dataset):
    """数据增强：随机旋转 + 高斯噪声"""
    def __init__(self, subset, noise_std=0.01, rotation_std=0.1):
        self.subset = subset
        self.noise_std = noise_std
        self.rotation_std = rotation_std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]  # (60, 3), (6,)

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if self.rotation_std > 0:
            # 随机小角度旋转（绕z轴）
            angle = torch.randn(1).item() * self.rotation_std
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_z = torch.tensor([[cos_a, -sin_a, 0],
                                  [sin_a, cos_a, 0],
                                  [0, 0, 1]], dtype=x.dtype)
            x = torch.matmul(x, rot_z.T)

        return x, y


# ─────────────────────── 训练逻辑 ───────────────────────

def select_files_by_number(data_dir):
    """扫描目录下的processed文件，让用户通过编号选择

    Returns:
        train_val_files: 用于训练+验证的文件列表
        test_files: 用于测试的文件列表
        data_dir: 数据所在目录
    """
    # 扫描文件
    h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))

    if not h5_files:
        print(f"在 {data_dir} 中未找到 processed_*.h5 文件")
        return [], [], data_dir

    # 显示文件列表
    print("\n" + "="*60)
    print(f"在 {data_dir} 中找到以下文件:")
    for i, f in enumerate(h5_files):
        print(f"  [{i}] {os.path.basename(f)}")
    print("="*60)

    # 获取用户输入
    while True:
        try:
            train_val_input = input("\n请输入训练+验证文件的编号 (例如: 0-4 或 0,1,2,3,4): ").strip()
            if not train_val_input:
                print("输入不能为空")
                continue

            # 解析输入
            train_val_indices = parse_indices(train_val_input, len(h5_files))
            if train_val_indices is None:
                continue

            test_input = input("请输入测试文件的编号 (例如: 5 或 5,6，留空表示无测试集): ").strip()
            if test_input:
                test_indices = parse_indices(test_input, len(h5_files))
                if test_indices is None:
                    continue
            else:
                test_indices = []

            # 检查是否有重复
            if set(train_val_indices) & set(test_indices):
                print("错误: 训练和测试文件不能重复")
                continue

            break
        except KeyboardInterrupt:
            print("\n用户取消")
            return [], [], data_dir

    train_val_files = [h5_files[i] for i in train_val_indices]
    test_files = [h5_files[i] for i in test_indices]

    print(f"\n训练+验证文件 ({len(train_val_files)}):")
    for f in train_val_files:
        print(f"  {os.path.basename(f)}")
    print(f"测试文件 ({len(test_files)}):")
    for f in test_files:
        print(f"  {os.path.basename(f)}")

    return train_val_files, test_files, data_dir


def parse_indices(input_str, max_index):
    """解析用户输入的索引字符串

    支持格式:
    - 单个数字: "5"
    - 逗号分隔: "0,1,2,3"
    - 范围: "0-4"
    - 混合: "0-2,5,7-9"

    Returns:
        list of int or None if invalid
    """
    indices = []
    parts = input_str.split(',')

    try:
        for part in parts:
            part = part.strip()
            if '-' in part:
                # 范围
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start < 0 or end >= max_index or start > end:
                    print(f"错误: 范围 {part} 无效 (有效范围: 0-{max_index-1})")
                    return None
                indices.extend(range(start, end + 1))
            else:
                # 单个数字
                idx = int(part)
                if idx < 0 or idx >= max_index:
                    print(f"错误: 索引 {idx} 超出范围 (有效范围: 0-{max_index-1})")
                    return None
                indices.append(idx)

        return sorted(set(indices))  # 去重并排序
    except ValueError:
        print(f"错误: 输入格式无效，请使用数字、逗号或连字符")
        return None


def split_by_contiguous(dataset, train_val_fnames, test_fnames, val_ratio=0.15):
    train_ratio = 1.0 - val_ratio
    train_idx, val_idx, test_idx = [], [], []

    for fname, n_frames, offset in dataset.file_info:
        if fname in test_fnames:
            test_idx.extend(range(offset, offset + n_frames))
            print(f"  {fname}: test={n_frames} (整段)")
        elif fname in train_val_fnames:
            train_end = int(n_frames * train_ratio)
            train_idx.extend(range(offset, offset + train_end))
            val_idx.extend(range(offset + train_end, offset + n_frames))
            print(f"  {fname}: train={train_end}, val={n_frames - train_end}")
        else:
            print(f"  {fname}: 未选中，跳过")

    print(f"数据划分合计: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train(args):
    # 选择文件
    if args.no_interact:
        data_dir = args.data_dir
        h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))
        if not h5_files:
            print(f"在 {data_dir} 中未找到文件")
            sys.exit(1)
        train_val_files = h5_files
        test_files = []
    else:
        # 使用编号选择方式
        data_dir = args.data_dir
        train_val_files, test_files, data_dir = select_files_by_number(data_dir)

    if not train_val_files:
        print("未选择任何训练文件，退出")
        sys.exit(1)

    if data_dir is None:
        data_dir = os.path.dirname(train_val_files[0])

    # 加载数据
    all_files = train_val_files + [f for f in test_files if f not in train_val_files]
    dataset = ForceDataset(all_files, force_dims=args.force_dims)

    # 划分数据（使用基于segment的划分方式）
    train_idx, val_idx = dataset.get_segment_based_split(train_ratio=0.8, group_size=10)

    # 处理测试集
    test_idx = []
    if test_files:
        test_fnames = set(os.path.basename(f) for f in test_files)
        for fname, n_frames, offset in dataset.file_info:
            if fname in test_fnames:
                test_idx.extend(range(offset, offset + n_frames))
                print(f"  {fname}: test={n_frames} (整段)")

    # 计算scaler
    dataset.compute_scaler(train_idx)
    scaler = dataset.get_scaler()
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # 数据增强
    if args.augment:
        print(f"数据增强: noise_std={args.noise_std}, rotation_std={args.rotation_std}")
        train_set = AugmentedSubset(train_set, noise_std=args.noise_std,
                                     rotation_std=args.rotation_std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = ForcePointNet(output_dim=args.force_dims,
                          use_input_transform=args.use_input_transform,
                          use_feature_transform=args.use_feature_transform).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = nn.MSELoss()
    reg_weight = args.reg_weight

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    save_dir = os.path.join(data_dir, "model_output_pointnet")
    os.makedirs(save_dir, exist_ok=True)

    y_mean_t = torch.tensor(scaler['y_mean'], device=device)
    y_std_t = torch.tensor(scaler['y_std'], device=device)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, trans_feat = model(xb)

            # MSE loss
            loss_mse = criterion(pred, yb)

            # Feature transform regularization
            loss_reg = model.feature_transform_regularizer(trans_feat)

            # 大力加权
            fz_true_raw = yb[:, 2] * y_std_t[2] + y_mean_t[2]
            weight = torch.where(torch.abs(fz_true_raw) > 5, 2.0, 1.0).mean()

            loss = loss_mse * weight + reg_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_set)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_set)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}, best val_loss={best_val_loss:.6f}")
                break

    # 零点偏置校正
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth")))
    model.eval()

    zero_threshold = {'fx': 0.3, 'fy': 0.3, 'fz': 0.5, 'mx': 0.01, 'my': 0.01, 'mz': 0.005}
    zero_mask = np.ones(len(val_idx), dtype=bool)
    for i, dim_name in enumerate(['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:args.force_dims]):
        zero_mask &= (np.abs(dataset.y[val_idx, i]) < zero_threshold[dim_name])

    if zero_mask.sum() > 10:
        zero_indices = np.array(val_idx)[zero_mask]
        zero_loader = DataLoader(Subset(dataset, zero_indices.tolist()),
                                batch_size=args.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in zero_loader:
                pred, _ = model(xb.to(device))
                pred = pred.cpu().numpy() * scaler['y_std'] + scaler['y_mean']
                preds.append(pred)

        bias = np.concatenate(preds, axis=0).mean(axis=0)
        print(f"\n零点偏置校正 (基于{zero_mask.sum()}个零点样本):")
        for i, name in enumerate(['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:args.force_dims]):
            print(f"  {name}: {bias[i]:.4f}")
    else:
        bias = np.zeros(args.force_dims)
        print(f"\n零点样本不足({zero_mask.sum()}个)，跳过偏置校正")

    # 保存
    np.savez(os.path.join(save_dir, "scaler.npz"), **scaler, bias=bias)

    config = {
        'model_type': 'PointNet',
        'output_dim': args.force_dims,
        'use_input_transform': args.use_input_transform,
        'use_feature_transform': args.use_feature_transform,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs_trained': epoch,
        'best_val_loss': float(best_val_loss),
        'device': str(device),
        'augment': args.augment,
        'noise_std': args.noise_std,
        'rotation_std': args.rotation_std,
        'reg_weight': args.reg_weight,
        'train_val_files': [os.path.basename(f) for f in train_val_files],
        'test_files': [os.path.basename(f) for f in test_files],
        'total_samples': len(dataset),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    with open(os.path.join(save_dir, "train_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    np.savez(os.path.join(save_dir, "train_history.npz"),
             train_loss=history['train_loss'],
             val_loss=history['val_loss'],
             lr=history['lr'])

    np.savez(os.path.join(save_dir, "split_indices.npz"),
             train=train_idx, val=val_idx, test=test_idx)

    print(f"\n训练完成，模型已保存到: {save_dir}")
    return save_dir


# ─────────────────────── 入口 ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="V6 触觉传感器力标定 — PointNet 训练")
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "force_calibration"),
                        help="processed_*.h5 所在目录")
    parser.add_argument('--force_dims', type=int, default=6,
                        help="输出力维度 (3=仅力, 6=力+力矩)")
    parser.add_argument('--use_input_transform', action='store_true', default=True,
                        help="使用输入变换网络")
    parser.add_argument('--use_feature_transform', action='store_true', default=True,
                        help="使用特征变换网络")
    parser.add_argument('--lr', type=float, default=1e-3, help="学习率")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="权重衰减")
    parser.add_argument('--batch_size', type=int, default=256, help="批大小")
    parser.add_argument('--epochs', type=int, default=500, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping 耐心值")
    parser.add_argument('--augment', action='store_true', default=True,
                        help="启用数据增强")
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help="训练时输入噪声标准差")
    parser.add_argument('--rotation_std', type=float, default=0.1,
                        help="训练时随机旋转角度标准差(弧度)")
    parser.add_argument('--reg_weight', type=float, default=0.001,
                        help="特征变换正则化权重")
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'], help="学习率调度器")
    parser.add_argument('--no_interact', action='store_true',
                        help="非交互模式：所有文件用于训练+验证")

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

