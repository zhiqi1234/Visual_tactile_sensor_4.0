# -*- coding: utf-8 -*-
"""
V6_force_eval_ablation_pointnet.py — PointNet 消融实验对比评估
同时加载两个 PointNet 模型（Vision-only 和 Vision+Piezo），在同一图表中对比
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

from V6_force_train_pointnet import ForcePointNet, ForceDataset

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_model(model_dir, data_dir):
    """加载单个 PointNet 模型"""
    with open(os.path.join(model_dir, "train_config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)

    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    scaler = {k: sc[k] for k in sc.files}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ForcePointNet(
        output_dim=config['output_dim'],
        use_input_transform=config.get('use_input_transform', True),
        use_feature_transform=config.get('use_feature_transform', True),
        use_piezo=config.get('use_piezo', False),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"),
                                     map_location=device, weights_only=True))
    model.eval()

    use_piezo = config.get('use_piezo', False)
    h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))
    dataset = ForceDataset(h5_files, force_dims=config['output_dim'], use_piezo=use_piezo)
    dataset.set_scaler(scaler)

    sp = np.load(os.path.join(model_dir, "split_indices.npz"), allow_pickle=True)
    test_idx = sp['test'].tolist()

    return model, scaler, dataset, test_idx, config, device


def predict_all(model, dataset, indices, scaler, device):
    """预测"""
    model.eval()
    X_raw = dataset.X[indices]
    y_raw = dataset.y[indices]

    X_norm = (X_raw - scaler['x_mean']) / scaler['x_std']

    use_piezo = model.use_piezo
    P_norm = None
    if use_piezo:
        P_raw = dataset.P[indices]
        P_norm = (P_raw - scaler['p_mean']) / scaler['p_std']

    batch_size = 32
    y_pred_list = []

    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            X_batch = torch.tensor(X_norm[i:i+batch_size], dtype=torch.float32).to(device)
            P_batch = torch.tensor(P_norm[i:i+batch_size], dtype=torch.float32).to(device) if use_piezo else None
            y_batch, _ = model(X_batch, P_batch)
            y_pred_list.append(y_batch.cpu().numpy())

    y_pred_norm = np.vstack(y_pred_list)
    y_pred = y_pred_norm * scaler['y_std'] + scaler['y_mean']
    return y_raw, y_pred


def compute_metrics(y_true, y_pred, columns):
    metrics = {}
    for i, col in enumerate(columns):
        err = y_pred[:, i] - y_true[:, i]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        metrics[col] = {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}
    return metrics


def plot_ablation_comparison(y_true, y_pred_base, y_pred_multi, columns, save_dir):
    n = len(columns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred_base[:, i], s=10, alpha=0.4,
                  label='Vision-only', c='blue', edgecolors='none')
        ax.scatter(y_true[:, i], y_pred_multi[:, i], s=10, alpha=0.4,
                  label='Vision+Piezo', c='red', edgecolors='none')

        vmin = min(y_true[:, i].min(), y_pred_base[:, i].min(), y_pred_multi[:, i].min())
        vmax = max(y_true[:, i].max(), y_pred_base[:, i].max(), y_pred_multi[:, i].max())
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'k--', linewidth=1, label='理想线')
        ax.set_xlabel(f'真实 {col}')
        ax.set_ylabel(f'预测 {col}')
        ax.set_title(col)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('消融实验对比 (PointNet): Vision-only vs Vision+Piezo', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_scatter.png"), dpi=150)
    plt.close(fig)
    print("  ablation_scatter.png")


def plot_ablation_timeseries(y_true, y_pred_base, y_pred_multi, columns, save_dir, max_samples=1000):
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    if len(y_true) > max_samples:
        indices = np.linspace(0, len(y_true) - 1, max_samples, dtype=int)
        y_true = y_true[indices]
        y_pred_base = y_pred_base[indices]
        y_pred_multi = y_pred_multi[indices]

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.plot(y_true[:, i], label='真实', linewidth=1.2, alpha=0.8, c='black')
        ax.plot(y_pred_base[:, i], label='Vision-only', linewidth=0.8, alpha=0.7, c='blue')
        ax.plot(y_pred_multi[:, i], label='Vision+Piezo', linewidth=0.8, alpha=0.7, c='red')
        ax.set_ylabel(col)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('样本索引')
    fig.suptitle('时间序列对比 (PointNet)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_timeseries.png"), dpi=150)
    plt.close(fig)
    print("  ablation_timeseries.png")


def plot_ablation_error_comparison(y_true, y_pred_base, y_pred_multi, columns, save_dir):
    metrics_base = compute_metrics(y_true, y_pred_base, columns)
    metrics_multi = compute_metrics(y_true, y_pred_multi, columns)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_names = ['MAE', 'RMSE', 'R2']

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        base_vals = [metrics_base[col][metric] for col in columns]
        multi_vals = [metrics_multi[col][metric] for col in columns]
        x = np.arange(len(columns))
        width = 0.35
        ax.bar(x - width/2, base_vals, width, label='Vision-only', color='blue', alpha=0.7)
        ax.bar(x + width/2, multi_vals, width, label='Vision+Piezo', color='red', alpha=0.7)
        ax.set_xlabel('力分量')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 对比 (PointNet)')
        ax.set_xticks(x)
        ax.set_xticklabels(columns)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('消融实验：指标对比 (PointNet)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_metrics.png"), dpi=150)
    plt.close(fig)
    print("  ablation_metrics.png")


def main():
    parser = argparse.ArgumentParser(description="PointNet 消融实验对比评估")
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help="PointNet Vision-only 模型目录")
    parser.add_argument('--multimodal_dir', type=str, default=None,
                        help="PointNet Vision+Piezo 模型目录")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "force_calibration")

    default_baseline = os.path.join(args.data_dir, "model_output_pointnet")
    default_multimodal = os.path.join(args.data_dir, "model_output_pointnet_piezo")

    if args.baseline_dir is None:
        if os.path.isdir(default_baseline):
            args.baseline_dir = default_baseline
        else:
            from tkinter import Tk, filedialog
            root = Tk(); root.withdraw(); root.attributes("-topmost", True)
            args.baseline_dir = filedialog.askdirectory(
                title="选择 PointNet Vision-only 模型目录", initialdir=args.data_dir)
            root.destroy()

    if args.multimodal_dir is None:
        if os.path.isdir(default_multimodal):
            args.multimodal_dir = default_multimodal
        else:
            from tkinter import Tk, filedialog
            root = Tk(); root.withdraw(); root.attributes("-topmost", True)
            args.multimodal_dir = filedialog.askdirectory(
                title="选择 PointNet Vision+Piezo 模型目录", initialdir=args.data_dir)
            root.destroy()

    if not args.baseline_dir or not args.multimodal_dir:
        print("未找到模型目录，请先运行 V6_force_train_pointnet.py 训练两个模型")
        return

    print("=" * 60)
    print("PointNet 消融实验对比评估")
    print("=" * 60)

    print("\n加载 PointNet Vision-only 模型...")
    model_base, scaler_base, dataset_base, test_idx_base, config_base, device = \
        load_model(args.baseline_dir, args.data_dir)

    print("\n加载 PointNet Vision+Piezo 模型...")
    model_multi, scaler_multi, dataset_multi, test_idx_multi, config_multi, _ = \
        load_model(args.multimodal_dir, args.data_dir)

    test_idx = sorted(set(test_idx_base) & set(test_idx_multi))
    if len(test_idx) == 0:
        print("错误：两个模型没有共同的测试集")
        return

    print(f"\n共同测试集样本数: {len(test_idx)}")
    columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:config_base['output_dim']]

    print("\n预测中...")
    y_true_base, y_pred_base = predict_all(model_base, dataset_base, test_idx, scaler_base, device)
    y_true_multi, y_pred_multi = predict_all(model_multi, dataset_multi, test_idx, scaler_multi, device)

    metrics_base = compute_metrics(y_true_base, y_pred_base, columns)
    metrics_multi = compute_metrics(y_true_multi, y_pred_multi, columns)

    print("\n" + "=" * 80)
    print(f"{'分量':>6s} {'指标':>8s} {'Vision-only':>15s} {'Vision+Piezo':>15s} {'改进':>10s}")
    print("=" * 80)

    for col in columns:
        for metric in ['MAE', 'RMSE', 'R2']:
            base_val = metrics_base[col][metric]
            multi_val = metrics_multi[col][metric]
            if metric == 'R2':
                improve_str = f"+{multi_val - base_val:.4f}"
            else:
                improve_str = f"{(base_val - multi_val) / base_val * 100:+.2f}%"
            print(f"{col:>6s} {metric:>8s} {base_val:15.4f} {multi_val:15.4f} {improve_str:>10s}")

    print("=" * 80)

    eval_dir = os.path.join(args.data_dir, "ablation_comparison_pointnet")
    os.makedirs(eval_dir, exist_ok=True)

    comparison = {
        'baseline': metrics_base,
        'multimodal': metrics_multi,
        'baseline_dir': args.baseline_dir,
        'multimodal_dir': args.multimodal_dir,
        'model_type': 'PointNet',
    }
    with open(os.path.join(eval_dir, "ablation_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n生成对比图表:")
    plot_ablation_comparison(y_true_base, y_pred_base, y_pred_multi, columns, eval_dir)
    plot_ablation_timeseries(y_true_base, y_pred_base, y_pred_multi, columns, eval_dir)
    plot_ablation_error_comparison(y_true_base, y_pred_base, y_pred_multi, columns, eval_dir)

    print(f"\n所有对比结果已保存到: {eval_dir}")


if __name__ == '__main__':
    main()
