# -*- coding: utf-8 -*-
"""
V6_force_eval_lightnet.py — LightNet 力标定模型评估与可视化
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

from V6_force_train_lightnet import LightNet, ForceDataset

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_model_and_data(model_dir, data_dir):
    with open(os.path.join(model_dir, "train_config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)

    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    scaler = {k: sc[k] for k in sc.files}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightNet(output_dim=config['output_dim']).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"),
                                     map_location=device, weights_only=True))
    model.eval()

    h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))
    dataset = ForceDataset(h5_files, force_dims=config['output_dim'])
    dataset.set_scaler(scaler)

    sp = np.load(os.path.join(model_dir, "split_indices.npz"), allow_pickle=True)
    test_idx = sp['test'].tolist()

    return model, scaler, dataset, test_idx, config, device


def predict_all(model, dataset, indices, scaler, device):
    model.eval()
    X_raw = dataset.X[indices]  # (N, 60, 3)
    y_raw = dataset.y[indices]

    X_norm = (X_raw - scaler['x_mean'].reshape(60, 3)) / scaler['x_std'].reshape(60, 3)
    X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_norm = model(X_t).cpu().numpy()

    y_pred = y_pred_norm * scaler['y_std'] + scaler['y_mean']
    if 'bias' in scaler:
        y_pred -= scaler['bias']
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
        max_err = np.max(np.abs(err))

        mask = np.abs(y_true[:, i]) > 0.1
        mape = np.mean(np.abs(err[mask] / y_true[mask, i])) * 100 if mask.sum() > 0 else float('nan')

        metrics[col] = {
            'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2),
            'MaxError': float(max_err), 'MAPE(%)': float(mape),
        }
    return metrics


def plot_train_history(model_dir, save_dir):
    hist = np.load(os.path.join(model_dir, "train_history.npz"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist['train_loss'], label='Train Loss', linewidth=1)
    ax.plot(hist['val_loss'], label='Val Loss', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练曲线 (LightNet)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "train_history.png"), dpi=150)
    plt.close(fig)


def plot_scatter(y_true, y_pred, columns, save_dir):
    n = len(columns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.5, edgecolors='none')
        vmin = min(y_true[:, i].min(), y_pred[:, i].min())
        vmax = max(y_true[:, i].max(), y_pred[:, i].max())
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1, label='理想线')
        ax.set_xlabel(f'真实 {col}')
        ax.set_ylabel(f'预测 {col}')
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('预测 vs 真实 (LightNet)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "scatter.png"), dpi=150)
    plt.close(fig)


def plot_timeseries(y_true, y_pred, columns, save_dir):
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.plot(y_true[:, i], label='真实', linewidth=0.8, alpha=0.8)
        ax.plot(y_pred[:, i], label='预测', linewidth=0.8, alpha=0.8)
        ax.set_ylabel(col)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('样本索引')
    fig.suptitle('时间序列对比 (LightNet)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "timeseries.png"), dpi=150)
    plt.close(fig)


def plot_error_dist(y_true, y_pred, columns, save_dir):
    n = len(columns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        err = y_pred[:, i] - y_true[:, i]
        ax.hist(err, bins=40, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel(f'{col} 误差')
        ax.set_ylabel('频次')
        ax.set_title(f'{col}  μ={err.mean():.3f}  σ={err.std():.3f}')
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('误差分布 (LightNet)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "error_dist.png"), dpi=150)
    plt.close(fig)


def plot_contact_analysis(dataset, test_idx, y_true, y_pred, save_dir):
    """分析平面vs尖锐物体的预测差异"""
    contact_ratios = dataset.concentrations[test_idx]
    median_ratio = np.median(contact_ratios)
    is_flat = contact_ratios > median_ratio

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(y_true[~is_flat, 2], y_pred[~is_flat, 2], s=10, alpha=0.5, label='尖锐物体', c='blue')
    ax.scatter(y_true[is_flat, 2], y_pred[is_flat, 2], s=10, alpha=0.5, label='平面物体', c='red')
    lim = max(abs(y_true[:, 2]).max(), abs(y_pred[:, 2]).max())
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax.set_xlabel('真实 fz')
    ax.set_ylabel('预测 fz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1]
    err_sharp = y_pred[~is_flat, 2] - y_true[~is_flat, 2]
    err_flat = y_pred[is_flat, 2] - y_true[is_flat, 2]
    ax.hist(err_sharp, bins=30, alpha=0.6, label=f'尖锐 (μ={err_sharp.mean():.2f})', color='blue')
    ax.hist(err_flat, bins=30, alpha=0.6, label=f'平面 (μ={err_flat.mean():.2f})', color='red')
    ax.axvline(0, color='k', linestyle='--', lw=1)
    ax.set_xlabel('fz 误差')
    ax.set_ylabel('频次')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'接触类型分析 LightNet (平面:{is_flat.sum()}, 尖锐:{(~is_flat).sum()})', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "contact_analysis.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="V6 LightNet 评估")
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    args = parser.parse_args()

    if args.model_dir is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        args.model_dir = filedialog.askdirectory(
            title="选择 LightNet 模型目录",
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        root.destroy()
        if not args.model_dir:
            return

    if args.data_dir is None:
        args.data_dir = os.path.dirname(args.model_dir)

    model, scaler, dataset, test_idx, config, device = \
        load_model_and_data(args.model_dir, args.data_dir)

    columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:config['output_dim']]

    y_true, y_pred = predict_all(model, dataset, test_idx, scaler, device)
    print(f"\n测试集样本数: {len(test_idx)}")

    metrics = compute_metrics(y_true, y_pred, columns)
    print("\n评估指标:")
    print(f"{'分量':>6s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'MaxErr':>8s} {'MAPE%':>8s}")
    print("-" * 50)
    for col in columns:
        m = metrics[col]
        print(f"{col:>6s} {m['MAE']:8.4f} {m['RMSE']:8.4f} {m['R2']:8.4f} "
              f"{m['MaxError']:8.4f} {m['MAPE(%)']:8.2f}")

    eval_dir = os.path.join(args.model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_train_history(args.model_dir, eval_dir)
    plot_scatter(y_true, y_pred, columns, eval_dir)
    plot_timeseries(y_true, y_pred, columns, eval_dir)
    plot_error_dist(y_true, y_pred, columns, eval_dir)
    plot_contact_analysis(dataset, test_idx, y_true, y_pred, eval_dir)

    print(f"\n所有评估结果已保存到: {eval_dir}")


if __name__ == '__main__':
    main()
