# -*- coding: utf-8 -*-
"""
V6_force_eval.py — 力标定模型评估与可视化
从 model_output/ 加载训练好的模型，在测试集上评估并生成图表。
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 复用训练代码中的模型和数据集定义
from V6_force_train import ForceMLP, ForceDataset

# 中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_model_and_data(model_dir, data_dir):
    """加载模型、scaler、数据"""
    # 训练配置
    with open(os.path.join(model_dir, "train_config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)

    # scaler
    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    scaler = {k: sc[k] for k in sc.files}

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ForceMLP(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],  # 保持与训练一致，eval() 会自动关闭 dropout
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"),
                                     map_location=device, weights_only=True))
    model.eval()

    # 数据（使用与训练时相同的use_piezo设置）
    use_piezo = config.get('use_piezo', False)
    h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))
    dataset = ForceDataset(h5_files, force_dims=config['output_dim'], use_piezo=use_piezo)
    dataset.set_scaler(scaler)  # 使用训练时保存的 scaler

    # 划分索引
    sp = np.load(os.path.join(model_dir, "split_indices.npz"), allow_pickle=True)
    test_idx = sp['test'].tolist()

    return model, scaler, dataset, test_idx, config, device


def predict_all(model, dataset, indices, scaler, device):
    """对指定索引做预测，返回真实值和预测值（原始尺度）"""
    model.eval()
    X_raw = dataset.X[indices]
    y_raw = dataset.y[indices]

    # 使用训练集的 scaler 做标准化
    X_norm = (X_raw - scaler['x_mean']) / scaler['x_std']

    X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_norm = model(X_t).cpu().numpy()

    # 反标准化 + 偏置校正
    y_pred = y_pred_norm * scaler['y_std'] + scaler['y_mean']
    if 'bias' in scaler:
        y_pred -= scaler['bias']
    return y_raw, y_pred


def compute_metrics(y_true, y_pred, columns):
    """计算每个分量的评估指标"""
    metrics = {}
    for i, col in enumerate(columns):
        err = y_pred[:, i] - y_true[:, i]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        max_err = np.max(np.abs(err))

        # MAPE（排除接近零的值）
        mask = np.abs(y_true[:, i]) > 0.1
        mape = np.mean(np.abs(err[mask] / y_true[mask, i])) * 100 if mask.sum() > 0 else float('nan')

        metrics[col] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MaxError': float(max_err),
            'MAPE(%)': float(mape),
        }
    return metrics


def plot_train_history(model_dir, save_dir):
    """绘制训练 loss 曲线"""
    hist = np.load(os.path.join(model_dir, "train_history.npz"))
    train_loss = hist['train_loss']
    val_loss = hist['val_loss']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label='Train Loss', linewidth=1)
    ax.plot(val_loss, label='Val Loss', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('训练曲线')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "train_history.png"), dpi=150)
    plt.close(fig)
    print("  train_history.png")


def plot_scatter(y_true, y_pred, columns, save_dir):
    """预测 vs 真实 散点图"""
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

    fig.suptitle('预测 vs 真实', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "scatter.png"), dpi=150)
    plt.close(fig)
    print("  scatter.png")


def plot_timeseries(y_true, y_pred, columns, save_dir):
    """时间序列对比图"""
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
    fig.suptitle('时间序列对比', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "timeseries.png"), dpi=150)
    plt.close(fig)
    print("  timeseries.png")


def plot_error_dist(y_true, y_pred, columns, save_dir):
    """误差分布直方图"""
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

    fig.suptitle('误差分布', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "error_dist.png"), dpi=150)
    plt.close(fig)
    print("  error_dist.png")


def plot_error_vs_magnitude(y_true, y_pred, columns, save_dir):
    """残差 vs 力大小（检查异方差性）"""
    n = len(columns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        err = np.abs(y_pred[:, i] - y_true[:, i])
        ax.scatter(np.abs(y_true[:, i]), err, s=8, alpha=0.4, edgecolors='none')
        ax.set_xlabel(f'|{col}| 真实值')
        ax.set_ylabel(f'|误差|')
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('误差 vs 力大小', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "error_vs_magnitude.png"), dpi=150)
    plt.close(fig)
    print("  error_vs_magnitude.png")


def plot_contact_analysis(dataset, test_idx, y_true, y_pred, save_dir):
    """分析平面vs尖锐物体的预测差异"""
    X_raw = dataset.X[test_idx]
    contact_ratio = X_raw[:, 180]
    concentration = X_raw[:, 185]

    is_flat = (contact_ratio > contact_ratio.mean()) & (concentration < concentration.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # fz预测对比
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

    # 误差分布对比
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

    fig.suptitle(f'接触类型分析 (平面:{is_flat.sum()}, 尖锐:{(~is_flat).sum()})', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "contact_analysis.png"), dpi=150)
    plt.close(fig)
    print("  contact_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="V6 力标定模型评估")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="processed_*.h5 所在目录")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="模型输出目录")
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help="基线模型目录（用于消融实验对比，可选）")
    args = parser.parse_args()

    # 如果未指定 model_dir，弹出选择对话框
    if args.model_dir is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        args.model_dir = filedialog.askdirectory(
            title="选择模型目录 (包含 model.pth, scaler.npz 等)",
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        root.destroy()
        if not args.model_dir:
            print("未选择模型目录，退出")
            return

    # 从 model_dir 推断 data_dir
    if args.data_dir is None:
        args.data_dir = os.path.dirname(args.model_dir)

    print("=" * 50)
    print("V6 力标定模型评估")
    print("=" * 50)

    # 加载
    model, scaler, dataset, test_idx, config, device = \
        load_model_and_data(args.model_dir, args.data_dir)

    # 打印文件分组信息
    if 'test_files' in config:
        print(f"\n测试文件: {config['test_files']}")
    if 'train_val_files' in config:
        print(f"训练+验证文件: {config['train_val_files']}")

    columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:config['output_dim']]

    # 预测
    y_true, y_pred = predict_all(model, dataset, test_idx, scaler, device)
    print(f"\n测试集样本数: {len(test_idx)}")

    # 指标
    metrics = compute_metrics(y_true, y_pred, columns)
    print("\n评估指标:")
    print(f"{'分量':>6s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'MaxErr':>8s} {'MAPE%':>8s}")
    print("-" * 50)
    for col in columns:
        m = metrics[col]
        print(f"{col:>6s} {m['MAE']:8.4f} {m['RMSE']:8.4f} {m['R2']:8.4f} "
              f"{m['MaxError']:8.4f} {m['MAPE(%)']:8.2f}")

    # 保存指标
    eval_dir = os.path.join(args.model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存: metrics.json")

    # 生成图表
    print("\n生成图表:")
    plot_train_history(args.model_dir, eval_dir)
    plot_scatter(y_true, y_pred, columns, eval_dir)
    plot_timeseries(y_true, y_pred, columns, eval_dir)
    plot_error_dist(y_true, y_pred, columns, eval_dir)
    plot_error_vs_magnitude(y_true, y_pred, columns, eval_dir)
    plot_contact_analysis(dataset, test_idx, y_true, y_pred, eval_dir)

    print(f"\n所有评估结果已保存到: {eval_dir}")


if __name__ == '__main__':
    main()
