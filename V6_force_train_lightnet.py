# -*- coding: utf-8 -*-
"""
V6_force_train_lightnet.py — 基于 LightNet 的触觉传感器力标定训练
轻量级点云网络：去掉TNet，缩小通道数，增加mean pooling
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


# ─────────────────────── LightNet 模型 ───────────────────────

class LightNet(nn.Module):
    """轻量级点云网络：per-point Conv1d + dual pooling (max+mean)"""
    def __init__(self, output_dim=6):
        super().__init__()
        self.output_dim = output_dim

        # Per-point feature extraction (轻量级)
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Regression head (128*2=256 from dual pooling)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) or (B, 3, N) point cloud
        Returns:
            out: (B, output_dim) predicted force
        """
        if x.dim() == 3 and x.size(2) == 3:
            x = x.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)

        # Per-point features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, N)

        # Dual pooling: max (尖锐物体) + mean (平面物体)
        max_feat = torch.max(x, 2)[0]  # (B, 128)
        mean_feat = torch.mean(x, 2)   # (B, 128)
        global_feat = torch.cat([max_feat, mean_feat], dim=1)  # (B, 256)

        # Regression
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ─────────────────────── 数据集 ───────────────────────

class ForceDataset(Dataset):
    """从多个 processed HDF5 文件加载 dxyz → force 数据"""

    def __init__(self, h5_files, force_dims=6, contact_threshold=0.02):
        self.X = []  # dxyz (N, 60, 3)
        self.y = []  # force
        self.file_info = []
        self.contact_threshold = contact_threshold

        for fpath in h5_files:
            with h5py.File(fpath, 'r') as f:
                if 'vision/dxyz' not in f or 'force/values' not in f:
                    print(f"[跳过] {fpath}: 缺少数据")
                    continue

                dxyz = f['vision/dxyz'][:]
                force = f['force/values'][:]
                T = min(dxyz.shape[0], force.shape[0])
                dxyz = dxyz[:T].astype(np.float32)
                force = force[:T, :force_dims].astype(np.float32)

                offset = sum(info[1] for info in self.file_info)
                self.X.append(dxyz)
                self.y.append(force)
                self.file_info.append((os.path.basename(fpath), T, offset))
                print(f"[加载] {os.path.basename(fpath)}: {T} 帧")

        if len(self.X) == 0:
            raise RuntimeError("未加载到任何有效数据")

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        # 计算集中度（用于损失加权，尖锐物体权重更高）
        self.concentrations = self._compute_concentrations()

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        print(f"\n总样本数: {len(self.X)}")
        print(f"输入shape: {self.X.shape}, 输出shape: {self.y.shape}")
        print(f"力范围: {self.y.min(axis=0)} ~ {self.y.max(axis=0)}")

    def _compute_concentrations(self):
        """计算每个样本的位移集中度（max/mean，尖锐物体集中度高）"""
        T = self.X.shape[0]
        concentrations = np.zeros(T, dtype=np.float32)
        for t in range(T):
            disp_norm = np.linalg.norm(self.X[t], axis=1)
            concentrations[t] = disp_norm.max() / (disp_norm.mean() + 1e-8)
        return concentrations

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
        concentration = self.concentrations[idx]
        return (torch.tensor(x_norm, dtype=torch.float32),
                torch.tensor(y_norm, dtype=torch.float32),
                torch.tensor(concentration, dtype=torch.float32))

    def get_scaler(self):
        if self.x_mean is None:
            raise RuntimeError("未设置 scaler")
        return {
            'x_mean': self.x_mean, 'x_std': self.x_std,
            'y_mean': self.y_mean, 'y_std': self.y_std,
        }


# ─────────────────────── 数据增强 ───────────────────────

class AugmentedSubset(Dataset):
    """数据增强：高斯噪声"""
    def __init__(self, subset, noise_std=0.01):
        self.subset = subset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y, concentration = self.subset[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y, concentration


# ─────────────────────── 训练逻辑 ───────────────────────

def select_files_interactive():
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    train_val_files = list(filedialog.askopenfilenames(
        title="选择 训练+验证 文件",
        initialdir=os.path.dirname(os.path.abspath(__file__)),
        filetypes=[("HDF5 文件", "*.h5"), ("所有文件", "*.*")],
    ))

    if not train_val_files:
        root.destroy()
        return [], [], None

    data_dir = os.path.dirname(train_val_files[0])
    test_files = list(filedialog.askopenfilenames(
        title="选择 测试 文件",
        initialdir=data_dir,
        filetypes=[("HDF5 文件", "*.h5"), ("所有文件", "*.*")],
    ))

    root.destroy()
    return train_val_files, test_files, data_dir


def split_by_contiguous(dataset, train_val_fnames, test_fnames, val_ratio=0.15):
    train_ratio = 1.0 - val_ratio
    train_idx, val_idx, test_idx = [], [], []

    for fname, n_frames, offset in dataset.file_info:
        if fname in test_fnames:
            test_idx.extend(range(offset, offset + n_frames))
        elif fname in train_val_fnames:
            train_end = int(n_frames * train_ratio)
            train_idx.extend(range(offset, offset + train_end))
            val_idx.extend(range(offset + train_end, offset + n_frames))

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
        train_val_files, test_files, data_dir = select_files_interactive()

    if not train_val_files:
        print("未选择任何训练文件，退出")
        sys.exit(1)

    if data_dir is None:
        data_dir = os.path.dirname(train_val_files[0])

    # 加载数据
    all_files = train_val_files + [f for f in test_files if f not in train_val_files]
    dataset = ForceDataset(all_files, force_dims=args.force_dims)

    # 划分数据
    train_val_fnames = set(os.path.basename(f) for f in train_val_files)
    test_fnames = set(os.path.basename(f) for f in test_files)
    train_idx, val_idx, test_idx = split_by_contiguous(dataset, train_val_fnames, test_fnames)

    # 计算scaler
    dataset.compute_scaler(train_idx)
    scaler = dataset.get_scaler()
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # 数据增强
    if args.noise_std > 0:
        print(f"数据增强: noise_std={args.noise_std}")
        train_set = AugmentedSubset(train_set, noise_std=args.noise_std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = LightNet(output_dim=args.force_dims).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = nn.MSELoss(reduction='none')

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    save_dir = os.path.join(data_dir, "model_output_lightnet")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb, concentration in train_loader:
            xb, yb, concentration = xb.to(device), yb.to(device), concentration.to(device)
            pred = model(xb)

            # 集中度加权：尖锐物体(高集中度)权重更高，平衡训练
            loss_per_sample = criterion(pred, yb).mean(dim=1)
            # 归一化集中度到[0,1]，然后反转（低集中度=平面=低权重）
            conc_norm = (concentration - 1.0) / 2.0  # 假设集中度范围1-3
            conc_norm = torch.clamp(conc_norm, 0, 1)
            weight = 1.0 + args.concentration_weight * conc_norm
            loss = (loss_per_sample * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_set)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
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
            for xb, _, _ in zero_loader:
                pred = model(xb.to(device))
                pred = pred.cpu().numpy() * scaler['y_std'] + scaler['y_mean']
                preds.append(pred)

        bias = np.concatenate(preds, axis=0).mean(axis=0)
        print(f"\n零点偏置校正 (基于{zero_mask.sum()}个零点样本):")
        for i, name in enumerate(['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:args.force_dims]):
            print(f"  {name}: {bias[i]:.4f}")
    else:
        bias = np.zeros(args.force_dims)

    # 保存
    np.savez(os.path.join(save_dir, "scaler.npz"), **scaler, bias=bias)

    config = {
        'model_type': 'LightNet',
        'output_dim': args.force_dims,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs_trained': epoch,
        'best_val_loss': float(best_val_loss),
        'noise_std': args.noise_std,
        'concentration_weight': args.concentration_weight,
        'train_val_files': [os.path.basename(f) for f in train_val_files],
        'test_files': [os.path.basename(f) for f in test_files],
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


def main():
    parser = argparse.ArgumentParser(description="V6 触觉传感器力标定 — LightNet 训练")
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "force_calibration_2"))
    parser.add_argument('--force_dims', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--concentration_weight', type=float, default=1.0,
                        help="集中度加权系数(尖锐物体权重增加)")
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--no_interact', action='store_true')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
