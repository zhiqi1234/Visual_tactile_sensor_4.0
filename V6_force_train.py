# -*- coding: utf-8 -*-
"""
V6_force_train.py — 基于 MLP 的触觉传感器力标定训练
输入: 60个marker点的位移 dxyz (T,60,3) → 展平 (T,180)
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
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# ─────────────────────── 数据集 ───────────────────────

class ForceDataset(Dataset):
    """从多个 processed HDF5 文件加载 dxyz → force 数据。

    注意: 标准化参数(scaler)不在此处计算，需要通过 set_scaler() 外部设置。
    这样可以确保 scaler 只基于训练集计算，避免数据泄漏。
    """

    def __init__(self, h5_files, force_dims=6, contact_threshold=0.02):
        self.X = []  # dxyz 展平 + 接触特征
        self.y = []  # force
        self.file_info = []  # 记录每个样本来源及其在总数组中的起止索引
        self.segment_info = []  # 记录每个segment的起止索引: [(start_idx, end_idx, file_idx), ...]
        self.contact_threshold = contact_threshold

        for file_idx, fpath in enumerate(h5_files):
            with h5py.File(fpath, 'r') as f:
                if 'vision/dxyz' not in f or 'force/values' not in f:
                    print(f"[跳过] {fpath}: 缺少 vision/dxyz 或 force/values")
                    continue

                dxyz = f['vision/dxyz'][:]       # (T, 60, 3)
                force = f['force/values'][:]      # (T, 6)
                T = min(dxyz.shape[0], force.shape[0])
                dxyz = dxyz[:T]
                force = force[:T, :force_dims]

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

                # 提取接触特征
                contact_features = self._extract_contact_features(dxyz)  # (T, 7)

                # 拼接：dxyz展平 + 接触特征
                x_flat = dxyz.reshape(T, -1).astype(np.float32)  # (T, 180)
                x_combined = np.concatenate([x_flat, contact_features], axis=1)  # (T, 187)
                y_flat = force.astype(np.float32)                 # (T, 6)

                offset = sum(info[1] for info in self.file_info)
                self.X.append(x_combined)
                self.y.append(y_flat)
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

                print(f"[加载] {os.path.basename(fpath)}: {T} 帧, "
                      f"dxyz shape={dxyz.shape}, force shape={force.shape}")

        if len(self.X) == 0:
            raise RuntimeError("未加载到任何有效数据，请检查文件路径")

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        # scaler 初始化为 None，需要通过 set_scaler() 设置
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        print(f"\n总样本数: {len(self.X)}")
        print(f"输入维度: {self.X.shape[1]}, 输出维度: {self.y.shape[1]}")
        print(f"力范围: {self.y.min(axis=0)} ~ {self.y.max(axis=0)}")
        print(f"总segment数: {len(self.segment_info)}")

    def _extract_contact_features(self, dxyz):
        """提取接触面积相关特征

        Args:
            dxyz: (T, 60, 3) marker位移

        Returns:
            features: (T, 9) 接触特征
        """
        T = dxyz.shape[0]
        features = np.zeros((T, 9), dtype=np.float32)

        for t in range(T):
            disp_norm = np.linalg.norm(dxyz[t], axis=1)  # (60,)

            # 有效接触点数量（归一化到0-1）
            n_contact = (disp_norm > self.contact_threshold).sum()
            features[t, 0] = n_contact / 60.0

            # 位移的空间方差（反映分布集中度）
            features[t, 1] = disp_norm.std()

            # 最大位移
            features[t, 2] = disp_norm.max()

            # 平均位移
            features[t, 3] = disp_norm.mean()

            # 位移总和
            features[t, 4] = disp_norm.sum()

            # 集中度 = 最大位移 / (平均位移+eps)
            features[t, 5] = disp_norm.max() / (disp_norm.mean() + 1e-8)

            # 平均接触位移 = 总位移 / 接触点数
            features[t, 6] = disp_norm.sum() / max(n_contact, 1)

            # ── 新增：压强相关特征 ──
            # 有效接触面积估计（接触点数 * 单位面积）
            contact_area = max(n_contact, 1) * 0.01  # 假设每个marker代表0.01单位面积

            # 压强指标 = 总位移 / 接触面积（位移越大、面积越小，压强越大）
            features[t, 7] = disp_norm.sum() / contact_area

            # 位移熵（反映分布均匀性，平面物体熵高）
            disp_prob = (disp_norm + 1e-8) / (disp_norm.sum() + 1e-8)
            features[t, 8] = -np.sum(disp_prob * np.log(disp_prob + 1e-8))

        return features

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
        """根据指定索引（通常为训练集）计算标准化参数并设置"""
        if indices is None:
            indices = list(range(len(self.X)))
        X_sub = self.X[indices]
        y_sub = self.y[indices]

        self.x_mean = X_sub.mean(axis=0)
        self.x_std = X_sub.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0

        self.y_mean = y_sub.mean(axis=0)
        self.y_std = y_sub.std(axis=0)
        self.y_std[self.y_std < 1e-8] = 1.0

        print(f"Scaler 基于 {len(indices)} 个样本计算")

    def set_scaler(self, scaler):
        """从外部字典设置 scaler（用于评估时加载）"""
        self.x_mean = scaler['x_mean']
        self.x_std = scaler['x_std']
        self.y_mean = scaler['y_mean']
        self.y_std = scaler['y_std']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.x_mean is None:
            raise RuntimeError("未设置 scaler，请先调用 compute_scaler() 或 set_scaler()")
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


# ─────────────────────── 噪声数据增强 Wrapper ───────────────────────

class NoisySubset(Dataset):
    """对 Subset 的输入添加高斯噪声，用于训练时数据增强。"""

    def __init__(self, subset, noise_std=0.01):
        self.subset = subset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y


# ─────────────────────── MLP 模型 ───────────────────────

class ForceMLP(nn.Module):
    def __init__(self, input_dim=189, output_dim=6,
                 hidden_dims=(512, 256, 128), dropout=(0.2, 0.1, 0.0)):
        super().__init__()
        self.point_dim = 180
        self.contact_dim = input_dim - 180

        # 增强接触特征处理
        self.contact_net = nn.Sequential(
            nn.Linear(self.contact_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # 主干网络
        layers = []
        in_d = self.point_dim + 32
        for i, h_d in enumerate(hidden_dims):
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU(inplace=True))
            if i < len(dropout) and dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            in_d = h_d
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        point_feat = x[:, :self.point_dim]
        contact_feat = x[:, self.point_dim:]
        contact_processed = self.contact_net(contact_feat)
        x_combined = torch.cat([point_feat, contact_processed], dim=1)
        return self.net(x_combined)


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
    """
    按文件角色划分 train/val/test，杜绝相邻帧泄漏。

    - train_val_fnames 中的文件: 文件内按连续段切割为 train(85%) / val(15%)
    - test_fnames 中的文件: 整段用于测试

    这保证训练/验证/测试集之间不存在跨文件泄漏。
    """
    train_ratio = 1.0 - val_ratio
    train_idx = []
    val_idx = []
    test_idx = []

    for fname, n_frames, offset in dataset.file_info:
        if fname in test_fnames:
            # 测试文件：整段用于测试
            test_idx.extend(range(offset, offset + n_frames))
            print(f"  {fname}: test={n_frames} (整段)")
        elif fname in train_val_fnames:
            # 训练+验证文件：连续切割
            train_end = int(n_frames * train_ratio)
            train_idx.extend(range(offset, offset + train_end))
            val_idx.extend(range(offset + train_end, offset + n_frames))
            print(f"  {fname}: train={train_end}, val={n_frames - train_end}")
        else:
            print(f"  {fname}: 未选中，跳过")

    print(f"数据划分合计: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train(args):
    # ── 选择文件 ──
    if args.no_interact:
        data_dir = args.data_dir
        h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))
        if not h5_files:
            print(f"在 {data_dir} 中未找到 processed_*.h5 文件")
            sys.exit(1)
        train_val_files = h5_files
        test_files = []
        print(f"[非交互模式] 所有 {len(h5_files)} 个文件用于训练+验证")
    else:
        # 使用编号选择方式
        data_dir = args.data_dir
        train_val_files, test_files, data_dir = select_files_by_number(data_dir)

    if not train_val_files:
        print("未选择任何训练文件，退出")
        sys.exit(1)

    if data_dir is None:
        data_dir = os.path.dirname(train_val_files[0])

    # ── 加载数据（加载所有选中的文件） ──
    all_files = train_val_files + [f for f in test_files if f not in train_val_files]
    dataset = ForceDataset(all_files, force_dims=args.force_dims)

    # ── 划分数据（先划分，再算 scaler，杜绝泄漏） ──
    # 使用基于segment的划分方式
    train_idx, val_idx = dataset.get_segment_based_split(train_ratio=0.8, group_size=10)

    # 处理测试集
    test_idx = []
    if test_files:
        test_fnames = set(os.path.basename(f) for f in test_files)
        for fname, n_frames, offset in dataset.file_info:
            if fname in test_fnames:
                test_idx.extend(range(offset, offset + n_frames))
                print(f"  {fname}: test={n_frames} (整段)")

    # ── 只在训练集上计算标准化参数 ──
    dataset.compute_scaler(train_idx)
    scaler = dataset.get_scaler()
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # 训练集加噪声增强
    if args.noise_std > 0:
        print(f"训练噪声增强: std={args.noise_std}")
        train_set = NoisySubset(train_set, noise_std=args.noise_std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── 模型 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    input_dim = dataset.X.shape[1]
    output_dim = dataset.y.shape[1]
    model = ForceMLP(input_dim=input_dim, output_dim=output_dim,
                     hidden_dims=args.hidden_dims, dropout=args.dropout).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 训练配置 ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss(reduction='none')  # 改为不自动求均值

    # ── 训练循环 ──
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    save_dir = os.path.join(data_dir, "model_output")
    os.makedirs(save_dir, exist_ok=True)

    # 预计算scaler的tensor版本（用于加权）
    y_mean_t = torch.tensor(scaler['y_mean'], device=device)
    y_std_t = torch.tensor(scaler['y_std'], device=device)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb).mean()

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
                pred = model(xb)
                loss = criterion(pred, yb).mean()
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
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={current_lr:.2e}")

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

    # ── 计算零点偏置校正 ──
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth")))
    model.eval()

    # 在验证集上找零点样本（真实力接近0）
    zero_threshold = {'fx': 0.3, 'fy': 0.3, 'fz': 0.5, 'mx': 0.01, 'my': 0.01, 'mz': 0.005}
    zero_mask = np.ones(len(val_idx), dtype=bool)
    for i, dim_name in enumerate(['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:output_dim]):
        zero_mask &= (np.abs(dataset.y[val_idx, i]) < zero_threshold[dim_name])

    if zero_mask.sum() > 10:
        zero_indices = np.array(val_idx)[zero_mask]
        zero_loader = DataLoader(Subset(dataset, zero_indices.tolist()),
                                batch_size=args.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for xb, _ in zero_loader:
                pred = model(xb.to(device))
                pred = pred.cpu().numpy() * scaler['y_std'] + scaler['y_mean']
                preds.append(pred)

        bias = np.concatenate(preds, axis=0).mean(axis=0)
        print(f"\n零点偏置校正 (基于{zero_mask.sum()}个零点样本):")
        for i, name in enumerate(['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:output_dim]):
            print(f"  {name}: {bias[i]:.4f}")
    else:
        bias = np.zeros(output_dim)
        print(f"\n零点样本不足({zero_mask.sum()}个)，跳过偏置校正")

    # ── 保存 ──
    # scaler + bias
    np.savez(os.path.join(save_dir, "scaler.npz"), **scaler, bias=bias)

    # 训练配置
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': list(args.hidden_dims),
        'dropout': list(args.dropout),
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs_trained': epoch,
        'best_val_loss': float(best_val_loss),
        'device': str(device),
        'noise_std': args.noise_std,
        'scheduler': args.scheduler,
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

    # 训练历史
    np.savez(os.path.join(save_dir, "train_history.npz"),
             train_loss=history['train_loss'],
             val_loss=history['val_loss'],
             lr=history['lr'])

    # 保存测试集索引供评估使用
    np.savez(os.path.join(save_dir, "split_indices.npz"),
             train=train_idx, val=val_idx, test=test_idx)

    print(f"\n训练完成，模型已保存到: {save_dir}")
    print(f"  model.pth        — 模型权重")
    print(f"  scaler.npz       — 标准化参数")
    print(f"  train_config.json — 训练配置")
    print(f"  train_history.npz — 训练曲线数据")
    print(f"  split_indices.npz — 数据划分索引")

    return save_dir


# ─────────────────────── 入口 ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="V6 触觉传感器力标定 — MLP 训练")
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "force_calibration"),
                        help="processed_*.h5 所在目录")
    parser.add_argument('--force_dims', type=int, default=6,
                        help="输出力维度 (3=仅力, 6=力+力矩)")
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help="隐藏层维度")
    parser.add_argument('--dropout', type=float, nargs='+', default=[0.2, 0.1, 0.0],
                        help="各隐藏层 dropout 率")
    parser.add_argument('--lr', type=float, default=5e-4, help="学习率")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="权重衰减")
    parser.add_argument('--batch_size', type=int, default=64, help="批大小")
    parser.add_argument('--epochs', type=int, default=500, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping 耐心值")
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help="训练时输入噪声标准差 (0=不加噪声)")
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'], help="学习率调度器")
    parser.add_argument('--no_interact', action='store_true',
                        help="非交互模式：所有文件用于训练+验证，无独立测试文件")

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
