# -*- coding: utf-8 -*-
"""
V6_force_predict_lightnet.py — LightNet 力标定模型实时推理接口
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightNet(nn.Module):
    """轻量级点云网络"""
    def __init__(self, output_dim=6):
        super().__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.dim() == 3 and x.size(2) == 3:
            x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        max_feat = torch.max(x, 2)[0]
        mean_feat = torch.mean(x, 2)
        global_feat = torch.cat([max_feat, mean_feat], dim=1)

        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ForcePredictor:
    """力预测器"""

    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(os.path.join(model_dir, "train_config.json"), 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        sc = np.load(os.path.join(model_dir, "scaler.npz"))
        self.x_mean = sc['x_mean'].astype(np.float32)
        self.x_std = sc['x_std'].astype(np.float32)
        self.y_mean = sc['y_mean'].astype(np.float32)
        self.y_std = sc['y_std'].astype(np.float32)
        self.bias = sc['bias'].astype(np.float32) if 'bias' in sc else np.zeros(6, dtype=np.float32)

        self.model = LightNet(output_dim=self.config['output_dim']).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pth"),
                       map_location=self.device, weights_only=True))
        self.model.eval()

        self._x_mean_t = torch.tensor(self.x_mean, device=self.device)
        self._x_std_t = torch.tensor(self.x_std, device=self.device)
        self._y_mean_t = torch.tensor(self.y_mean, device=self.device)
        self._y_std_t = torch.tensor(self.y_std, device=self.device)
        self._bias_t = torch.tensor(self.bias, device=self.device)

        self.columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:self.config['output_dim']]
        print(f"[ForcePredictor] LightNet已加载, 设备={self.device}, 输出={self.columns}")

    def predict(self, dxyz):
        """单帧预测"""
        x = np.asarray(dxyz, dtype=np.float32).reshape(60, 3)
        x_t = torch.tensor(x, device=self.device).unsqueeze(0)
        x_t = (x_t - self._x_mean_t) / self._x_std_t

        with torch.no_grad():
            y_t = self.model(x_t)

        y_t = y_t * self._y_std_t + self._y_mean_t - self._bias_t
        return y_t.cpu().numpy()[0]

    def predict_batch(self, dxyz_batch):
        """批量预测"""
        dxyz_batch = np.asarray(dxyz_batch, dtype=np.float32)
        if dxyz_batch.ndim == 3:
            x = dxyz_batch
        else:
            x = dxyz_batch.reshape(-1, 60, 3)

        x_t = torch.tensor(x, device=self.device)
        x_t = (x_t - self._x_mean_t) / self._x_std_t

        with torch.no_grad():
            y_t = self.model(x_t)

        y_t = y_t * self._y_std_t + self._y_mean_t - self._bias_t
        return y_t.cpu().numpy()


def main():
    """测试推理速度"""
    import time
    import argparse
    from tkinter import Tk, filedialog

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    args = parser.parse_args()

    if args.model_dir is None:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        args.model_dir = filedialog.askdirectory(title="选择模型目录")
        root.destroy()
        if not args.model_dir:
            return

    predictor = ForcePredictor(args.model_dir)

    dxyz = np.random.randn(60, 3).astype(np.float32) * 0.1
    force = predictor.predict(dxyz)
    print(f"\n单帧推理:")
    for col, val in zip(predictor.columns, force):
        print(f"  {col} = {val:.4f}")

    n_iter = 1000
    start = time.perf_counter()
    for _ in range(n_iter):
        predictor.predict(dxyz)
    elapsed = time.perf_counter() - start
    print(f"\n推理速度: {n_iter/elapsed:.0f} FPS ({elapsed/n_iter*1000:.2f} ms/帧)")


if __name__ == '__main__':
    main()
