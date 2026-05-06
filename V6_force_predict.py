# -*- coding: utf-8 -*-
"""
V6_force_predict.py — 力标定模型实时推理接口
加载训练好的模型，输入 dxyz 即可预测力。
可独立运行测试，也可作为模块被 V5_Vptac-force.py 导入使用。
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn


class ForceMLP(nn.Module):
    """与训练代码中完全一致的网络结构"""
    def __init__(self, input_dim=189, output_dim=6,
                 hidden_dims=(512, 256, 128), dropout=(0.2, 0.1, 0.0)):
        super().__init__()
        self.point_dim = 180
        self.contact_dim = input_dim - 180

        self.contact_net = nn.Sequential(
            nn.Linear(self.contact_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

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


class ForcePredictor:
    """
    力预测器 — 封装模型加载和推理逻辑。

    用法:
        predictor = ForcePredictor("force_calibration/model_output")
        force = predictor.predict(dxyz)  # dxyz: (60, 3) → force: (6,)
    """

    def __init__(self, model_dir, contact_threshold=0.02):
        """
        Args:
            model_dir: 包含 model.pth, scaler.npz, train_config.json 的目录
            contact_threshold: 接触判定阈值
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.contact_threshold = contact_threshold

        # 加载配置
        config_path = os.path.join(model_dir, "train_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载 scaler + bias
        sc = np.load(os.path.join(model_dir, "scaler.npz"))
        self.x_mean = sc['x_mean'].astype(np.float32)
        self.x_std = sc['x_std'].astype(np.float32)
        self.y_mean = sc['y_mean'].astype(np.float32)
        self.y_std = sc['y_std'].astype(np.float32)
        self.bias = sc['bias'].astype(np.float32) if 'bias' in sc else np.zeros(6, dtype=np.float32)

        # 加载模型
        self.model = ForceMLP(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            hidden_dims=self.config['hidden_dims'],
            dropout=self.config['dropout'],  # 保持与训练一致，eval() 自动关闭 dropout
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pth"),
                       map_location=self.device, weights_only=True))
        self.model.eval()

        # 预计算 tensor 版 scaler（加速推理）
        self._x_mean_t = torch.tensor(self.x_mean, device=self.device)
        self._x_std_t = torch.tensor(self.x_std, device=self.device)
        self._y_mean_t = torch.tensor(self.y_mean, device=self.device)
        self._y_std_t = torch.tensor(self.y_std, device=self.device)
        self._bias_t = torch.tensor(self.bias, device=self.device)

        self.columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:self.config['output_dim']]
        print(f"[ForcePredictor] 模型已加载, 设备={self.device}, "
              f"输入={self.config['input_dim']}维, 输出={self.columns}")

    def _extract_contact_features(self, dxyz):
        """提取接触特征 (单帧)

        Args:
            dxyz: (60, 3) 或 (180,)

        Returns:
            (9,) 接触特征
        """
        dxyz = np.asarray(dxyz).reshape(60, 3)
        disp_norm = np.linalg.norm(dxyz, axis=1)

        n_contact = (disp_norm > self.contact_threshold).sum()
        n_contact_ratio = n_contact / 60.0
        disp_sum = disp_norm.sum()
        concentration = disp_norm.max() / (disp_norm.mean() + 1e-8)
        avg_contact_disp = disp_sum / max(n_contact, 1)

        contact_area = max(n_contact, 1) * 0.01
        pressure_index = disp_sum / contact_area
        disp_prob = (disp_norm + 1e-8) / (disp_sum + 1e-8)
        entropy = -np.sum(disp_prob * np.log(disp_prob + 1e-8))

        return np.array([n_contact_ratio, disp_norm.std(), disp_norm.max(),
                         disp_norm.mean(), disp_sum, concentration, avg_contact_disp,
                         pressure_index, entropy], dtype=np.float32)

    def predict(self, dxyz, piezo_feat=None):
        """
        单帧预测。

        Args:
            dxyz: np.ndarray, shape (N, 3) 或 (N*3,)，N 个 marker 点的位移
            piezo_feat: np.ndarray, shape (5,)，压电统计特征（可选）
                        [mean, std, rms, max, energy]

        Returns:
            np.ndarray, shape (output_dim,) — 预测的力/力矩
        """
        x_flat = np.asarray(dxyz, dtype=np.float32).flatten()
        contact_feat = self._extract_contact_features(dxyz)
        x = np.concatenate([x_flat, contact_feat])

        # 如果模型需要压电特征
        if self.config.get('use_piezo', False):
            if piezo_feat is not None:
                x = np.concatenate([x, np.asarray(piezo_feat, dtype=np.float32)])
            else:
                # 无压电数据时填零（降级使用）
                x = np.concatenate([x, np.zeros(5, dtype=np.float32)])

        x_t = torch.tensor(x, device=self.device).unsqueeze(0)
        x_t = (x_t - self._x_mean_t) / self._x_std_t

        with torch.no_grad():
            y_t = self.model(x_t)

        y_t = y_t * self._y_std_t + self._y_mean_t - self._bias_t
        return y_t.cpu().numpy()[0]

    def predict_batch(self, dxyz_batch, piezo_feat_batch=None):
        """
        批量预测。

        Args:
            dxyz_batch: np.ndarray, shape (T, N, 3) 或 (T, N*3)
            piezo_feat_batch: np.ndarray, shape (T, 5)，压电统计特征（可选）

        Returns:
            np.ndarray, shape (T, output_dim)
        """
        dxyz_batch = np.asarray(dxyz_batch, dtype=np.float32)
        if dxyz_batch.ndim == 3:
            T = dxyz_batch.shape[0]
            x_flat = dxyz_batch.reshape(T, -1)
        else:
            T = dxyz_batch.shape[0]
            x_flat = dxyz_batch

        # 批量提取接触特征
        contact_feats = np.array([self._extract_contact_features(x_flat[i]) for i in range(T)])
        x = np.concatenate([x_flat, contact_feats], axis=1)

        # 如果模型需要压电特征
        if self.config.get('use_piezo', False):
            if piezo_feat_batch is not None:
                x = np.concatenate([x, np.asarray(piezo_feat_batch, dtype=np.float32)], axis=1)
            else:
                x = np.concatenate([x, np.zeros((T, 5), dtype=np.float32)], axis=1)

        x_t = torch.tensor(x, device=self.device)
        x_t = (x_t - self._x_mean_t) / self._x_std_t

        with torch.no_grad():
            y_t = self.model(x_t)

        y_t = y_t * self._y_std_t + self._y_mean_t - self._bias_t
        return y_t.cpu().numpy()


# ─────────────────────── 独立运行测试 ───────────────────────

def main():
    """加载模型并用随机数据测试推理速度"""
    import time
    import argparse
    from tkinter import Tk, filedialog

    parser = argparse.ArgumentParser(description="V6 力预测推理测试")
    parser.add_argument('--model_dir', type=str, default=None, help="模型目录")
    args = parser.parse_args()

    # 如果未指定模型目录，弹出选择对话框
    if args.model_dir is None:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        args.model_dir = filedialog.askdirectory(
            title="选择模型目录 (model_output)",
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        root.destroy()
        if not args.model_dir:
            print("未选择模型目录，退出")
            return

    predictor = ForcePredictor(args.model_dir)

    # 单帧推理测试
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

    # 批量推理测试
    batch = np.random.randn(100, 60, 3).astype(np.float32) * 0.1
    start = time.perf_counter()
    results = predictor.predict_batch(batch)
    elapsed = time.perf_counter() - start
    print(f"\n批量推理: 100 帧 / {elapsed*1000:.2f}ms, 输出 shape={results.shape}")


if __name__ == '__main__':
    main()
