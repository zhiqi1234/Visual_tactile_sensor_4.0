# -*- coding: utf-8 -*-
"""
V6_force_predict_pointnet.py — PointNet力标定模型实时推理接口
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────── 复用训练代码中的模型定义 ───────────────────────

class TNet(nn.Module):
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
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).flatten().unsqueeze(0)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetBackbone(nn.Module):
    def __init__(self, use_input_transform=True, use_feature_transform=True):
        super().__init__()
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

        if use_input_transform:
            self.input_transform = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        if use_feature_transform:
            self.feature_transform = TNet(k=128)

        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

    def forward(self, x):
        if self.use_input_transform:
            trans_input = self.input_transform(x)
            x = torch.bmm(trans_input, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = None
        if self.use_feature_transform:
            trans_feat = self.feature_transform(x)
            x = torch.bmm(trans_feat, x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feat = torch.max(x, 2, keepdim=False)[0]
        return global_feat, trans_feat


class ForcePointNet(nn.Module):
    def __init__(self, output_dim=6, use_input_transform=True, use_feature_transform=True):
        super().__init__()
        self.output_dim = output_dim
        self.use_feature_transform = use_feature_transform
        self.backbone = PointNetBackbone(use_input_transform, use_feature_transform)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        if x.dim() == 3 and x.size(2) == 3:
            x = x.transpose(2, 1)
        global_feat, trans_feat = self.backbone(x)
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, trans_feat

    def feature_transform_regularizer(self, trans_feat):
        if trans_feat is None:
            return 0
        d = trans_feat.size(1)
        I = torch.eye(d, device=trans_feat.device).unsqueeze(0)
        loss = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I, dim=(1, 2)))
        return loss


# ─────────────────────── 预测器 ───────────────────────

class ForcePredictor:
    """PointNet力预测器"""

    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载配置
        with open(os.path.join(model_dir, "train_config.json"), 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载scaler
        sc = np.load(os.path.join(model_dir, "scaler.npz"))
        self.x_mean = sc['x_mean'].astype(np.float32)
        self.x_std = sc['x_std'].astype(np.float32)
        self.y_mean = sc['y_mean'].astype(np.float32)
        self.y_std = sc['y_std'].astype(np.float32)
        self.bias = sc['bias'].astype(np.float32) if 'bias' in sc else np.zeros(6, dtype=np.float32)

        # 加载模型
        self.model = ForcePointNet(
            output_dim=self.config['output_dim'],
            use_input_transform=self.config.get('use_input_transform', True),
            use_feature_transform=self.config.get('use_feature_transform', True)
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pth"),
                       map_location=self.device, weights_only=True))
        self.model.eval()

        # 预计算tensor版scaler
        self._x_mean_t = torch.tensor(self.x_mean, device=self.device)
        self._x_std_t = torch.tensor(self.x_std, device=self.device)
        self._y_mean_t = torch.tensor(self.y_mean, device=self.device)
        self._y_std_t = torch.tensor(self.y_std, device=self.device)
        self._bias_t = torch.tensor(self.bias, device=self.device)

        self.columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:self.config['output_dim']]
        print(f"[ForcePredictor] PointNet模型已加载, 设备={self.device}, 输出={self.columns}")

    def predict(self, dxyz):
        """单帧预测
        Args:
            dxyz: (60, 3) 或 (180,)
        Returns:
            (output_dim,)
        """
        dxyz = np.asarray(dxyz, dtype=np.float32).reshape(60, 3)
        x_norm = (dxyz - self.x_mean) / self.x_std
        x_t = torch.tensor(x_norm, device=self.device).unsqueeze(0)

        with torch.no_grad():
            y_t, _ = self.model(x_t)

        y_t = y_t * self._y_std_t + self._y_mean_t - self._bias_t
        return y_t.cpu().numpy()[0]

    def predict_batch(self, dxyz_batch):
        """批量预测
        Args:
            dxyz_batch: (T, 60, 3) 或 (T, 180)
        Returns:
            (T, output_dim)
        """
        dxyz_batch = np.asarray(dxyz_batch, dtype=np.float32)
        if dxyz_batch.ndim == 2:
            dxyz_batch = dxyz_batch.reshape(-1, 60, 3)

        x_norm = (dxyz_batch - self.x_mean) / self.x_std
        x_t = torch.tensor(x_norm, device=self.device)

        with torch.no_grad():
            y_t, _ = self.model(x_t)

        y_t = y_t * self._y_std_t + self._y_mean_t - self._bias_t
        return y_t.cpu().numpy()


# ─────────────────────── 测试 ───────────────────────

def main():
    import time
    import argparse
    from tkinter import Tk, filedialog

    parser = argparse.ArgumentParser(description="PointNet力预测推理测试")
    parser.add_argument('--model_dir', type=str, default=None, help="模型目录")
    args = parser.parse_args()

    if args.model_dir is None:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        args.model_dir = filedialog.askdirectory(
            title="选择模型目录 (model_output_pointnet)",
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        root.destroy()
        if not args.model_dir:
            print("未选择模型目录，退出")
            return

    predictor = ForcePredictor(args.model_dir)

    # 单帧测试
    dxyz = np.random.randn(60, 3).astype(np.float32) * 0.1
    force = predictor.predict(dxyz)
    print(f"\n单帧推理测试:")
    for col, val in zip(predictor.columns, force):
        print(f"  {col} = {val:.4f}")

    # 速度测试
    n_iter = 1000
    start = time.perf_counter()
    for _ in range(n_iter):
        predictor.predict(dxyz)
    elapsed = time.perf_counter() - start
    print(f"\n推理速度: {n_iter} 次 / {elapsed:.3f}s = {n_iter/elapsed:.0f} FPS "
          f"({elapsed/n_iter*1000:.2f} ms/帧)")

    # 批量测试
    batch = np.random.randn(100, 60, 3).astype(np.float32) * 0.1
    start = time.perf_counter()
    results = predictor.predict_batch(batch)
    elapsed = time.perf_counter() - start
    print(f"\n批量推理: 100 帧 / {elapsed*1000:.2f}ms, 输出 shape={results.shape}")


if __name__ == '__main__':
    main()


