# -*- coding: utf-8 -*-
"""
V6_force_eval_cross_model.py — 跨架构综合对比评估
同时对比 MLP / PointNet / LightNet 三种架构的 Vision-only 和 Vision+Piezo 模型
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
import matplotlib.ticker as ticker
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ── 动态加载各架构的模型类 ──

def _import_model_class(model_type):
    if model_type == 'PointNet':
        from V6_force_train_pointnet import ForcePointNet, ForceDataset
        return ForcePointNet, ForceDataset
    elif model_type == 'LightNet':
        from V6_force_train_lightnet import LightNet, ForceDataset
        return LightNet, ForceDataset
    else:  # MLP
        from V6_force_train import ForceMLP, ForceDataset
        return ForceMLP, ForceDataset


def load_model(model_dir, data_dir):
    with open(os.path.join(model_dir, "train_config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)

    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    scaler = {k: sc[k] for k in sc.files}

    model_type = config.get('model_type', 'MLP')
    ModelClass, DatasetClass = _import_model_class(model_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 各架构不同的模型构造参数
    if model_type == 'PointNet':
        model = ModelClass(
            output_dim=config['output_dim'],
            use_input_transform=config.get('use_input_transform', True),
            use_feature_transform=config.get('use_feature_transform', True),
            use_piezo=config.get('use_piezo', False),
        ).to(device)
    elif model_type == 'LightNet':
        model = ModelClass(
            output_dim=config['output_dim'],
            use_piezo=config.get('use_piezo', False),
        ).to(device)
    else:  # MLP
        model = ModelClass(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
        ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"),
                                     map_location=device, weights_only=True))
    model.eval()

    use_piezo = config.get('use_piezo', False)
    h5_files = sorted(glob.glob(os.path.join(data_dir, "processed_*.h5")))
    dataset = DatasetClass(h5_files, force_dims=config['output_dim'], use_piezo=use_piezo)
    dataset.set_scaler(scaler)

    sp = np.load(os.path.join(model_dir, "split_indices.npz"), allow_pickle=True)
    test_idx = sp['test'].tolist()

    return model, scaler, dataset, test_idx, config, device, model_type


def predict_all(model, dataset, indices, scaler, device, model_type, use_piezo):
    model.eval()
    X_raw = dataset.X[indices]
    y_raw = dataset.y[indices]

    if model_type == 'MLP':
        X_norm = (X_raw - scaler['x_mean']) / scaler['x_std']
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
        P_t = None
        with torch.no_grad():
            y_pred_norm = model(X_t).cpu().numpy()
    elif model_type == 'PointNet':
        X_norm = (X_raw - scaler['x_mean']) / scaler['x_std']
        P_norm = None
        if use_piezo:
            P_norm = (dataset.P[indices] - scaler['p_mean']) / scaler['p_std']
        y_pred_list = []
        with torch.no_grad():
            for i in range(0, len(X_norm), 32):
                X_b = torch.tensor(X_norm[i:i+32], dtype=torch.float32).to(device)
                P_b = torch.tensor(P_norm[i:i+32], dtype=torch.float32).to(device) if use_piezo else None
                y_b, _ = model(X_b, P_b)
                y_pred_list.append(y_b.cpu().numpy())
        y_pred_norm = np.vstack(y_pred_list)
    else:  # LightNet
        X_norm = (X_raw - scaler['x_mean'].reshape(60, 3)) / scaler['x_std'].reshape(60, 3)
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
        P_t = None
        if use_piezo:
            P_norm = (dataset.P[indices] - scaler['p_mean']) / scaler['p_std']
            P_t = torch.tensor(P_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_norm = model(X_t, P_t).cpu().numpy()

    y_pred = y_pred_norm * scaler['y_std'] + scaler['y_mean']
    return y_raw, y_pred


def compute_metrics(y_true, y_pred, columns):
    metrics = {}
    for i, col in enumerate(columns):
        err = y_pred[:, i] - y_true[:, i]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-10))
        max_err = float(np.max(np.abs(err)))
        mask = np.abs(y_true[:, i]) > 0.1
        mape = float(np.mean(np.abs(err[mask] / y_true[mask, i])) * 100) if mask.sum() > 0 else float('nan')
        metrics[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MaxError': max_err,
            'MAPE(%)': mape,
        }
    return metrics


# ──────────────────── 控制台表格 ────────────────────

def print_comparison_table(all_results, columns):
    """打印完整的跨架构对比表"""
    arch_names = {'MLP': 'MLP', 'PointNet': 'PointNet', 'LightNet': 'LightNet'}
    variants = [('Vision-only', False), ('Vision+Piezo', True)]
    metric_fields = ['MAE', 'RMSE', 'MaxError', 'MAPE(%)', 'R2']

    for col in columns:
        print(f"\n{'='*110}")
        print(f"  {col}")
        print(f"{'='*110}")
        header = f"{'架构':>12s} {'变体':>14s}"
        for m in metric_fields:
            header += f" {m:>12s}"
        print(header)
        print("-" * 110)

        for arch_key in ['MLP', 'PointNet', 'LightNet']:
            if arch_key not in all_results:
                continue
            for var_label, use_piezo in variants:
                key = var_label
                if key not in all_results[arch_key]:
                    continue
                m = all_results[arch_key][key][col]
                vals = " ".join(f"{m[field]:12.4f}" for field in metric_fields[:-1])
                vals += f" {m['R2']:12.4f}"
                print(f"{arch_names[arch_key]:>12s} {var_label:>14s} {vals}")

    # 汇总排名：Fz MAE
    print(f"\n{'='*110}")
    print(f"  Fz MAE 排名 (越低越好)")
    print(f"{'='*110}")
    ranking = []
    for arch_key in ['MLP', 'PointNet', 'LightNet']:
        if arch_key not in all_results:
            continue
        for var_label, use_piezo in variants:
            key = var_label
            if key not in all_results[arch_key]:
                continue
            ranking.append((f"{arch_key} {var_label}", all_results[arch_key][key]['fz']['MAE']))
    ranking.sort(key=lambda x: x[1])
    for rank, (name, mae) in enumerate(ranking, 1):
        print(f"  [{rank}] {name:<25s}  MAE={mae:.4f}")


# ──────────────────── 图表 ────────────────────

def plot_radar_comparison(all_results, columns, save_dir):
    """雷达图：各模型在6个分量的MAE对比"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    variants = [('Vision-only', False), ('Vision+Piezo', True)]
    colors_vo = ['#3498DB', '#2ECC71', '#E67E22']
    colors_vp = ['#2980B9', '#27AE60', '#D35400']

    # 收集所有MAE计算全局范围
    all_mae = []
    for arch_key in arch_keys:
        if arch_key not in all_results:
            continue
        for var_label in ['Vision-only', 'Vision+Piezo']:
            if var_label in all_results[arch_key]:
                for col in columns:
                    all_mae.append(all_results[arch_key][var_label][col]['MAE'])
    mae_max = max(all_mae) * 1.15

    n_vars = len(columns)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

    for ax, use_piezo, title, colors in [
        (ax1, False, 'Vision-only', colors_vo),
        (ax2, True, 'Vision+Piezo', colors_vp),
    ]:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        for arch_key, color in zip(arch_keys, colors):
            if arch_key not in all_results:
                continue
            key = 'Vision+Piezo' if use_piezo else 'Vision-only'
            if key not in all_results[arch_key]:
                continue
            values = [all_results[arch_key][key][col]['MAE'] for col in columns]
            values += values[:1]
            ax.fill(angles, values, alpha=0.08, color=color)
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=arch_key, markersize=4)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(columns, fontsize=9)
        ax.set_ylim(0, mae_max)
        ax.set_title(f'{title} — MAE', fontsize=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle('MAE 雷达图', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_radar.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  cross_model_radar.png")


def plot_mae_bar_comparison(all_results, columns, save_dir):
    """分组柱状图：所有模型在所有分量上的 MAE"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    variants = [('Vision-only', False), ('Vision+Piezo', True)]
    n_cols = len(columns)
    n_bars = 0
    labels = []
    for ak in arch_keys:
        if ak in all_results:
            for vl, up in variants:
                if vl in all_results[ak]:
                    labels.append(f"{ak}\n{vl}")
                    n_bars += 1

    x = np.arange(n_cols)
    width = 0.8 / n_bars
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_idx = 0
    for arch_key in arch_keys:
        if arch_key not in all_results:
            continue
        for var_label, use_piezo in variants:
            key = var_label
            if key not in all_results[arch_key]:
                continue
            vals = [all_results[arch_key][key][col]['MAE'] for col in columns]
            offset = (bar_idx - (n_bars - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=labels[bar_idx],
                   color=colors[bar_idx], alpha=0.85, edgecolor='white', linewidth=0.5)
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(columns)
    ax.set_ylabel('MAE')
    ax.set_title(' MAE 对比', fontsize=13)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_mae.png"), dpi=150)
    plt.close(fig)
    print("  cross_model_mae.png")


def plot_rmse_bar_comparison(all_results, columns, save_dir):
    """分组柱状图：所有模型在所有分量上的 RMSE"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    variants = [('Vision-only', False), ('Vision+Piezo', True)]
    n_cols = len(columns)
    n_bars = 0
    labels = []
    for ak in arch_keys:
        if ak in all_results:
            for vl, up in variants:
                if vl in all_results[ak]:
                    labels.append(f"{ak}\n{vl}")
                    n_bars += 1

    x = np.arange(n_cols)
    width = 0.8 / n_bars
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_idx = 0
    for arch_key in arch_keys:
        if arch_key not in all_results:
            continue
        for var_label, use_piezo in variants:
            key = var_label
            if key not in all_results[arch_key]:
                continue
            vals = [all_results[arch_key][key][col]['RMSE'] for col in columns]
            offset = (bar_idx - (n_bars - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=labels[bar_idx],
                   color=colors[bar_idx], alpha=0.85, edgecolor='white', linewidth=0.5)
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(columns)
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE 对比', fontsize=13)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_rmse.png"), dpi=150)
    plt.close(fig)
    print("  cross_model_rmse.png")


def plot_scatter_grid(all_results, columns, save_dir):
    """3行×6列散点图矩阵：行=架构，列=分量"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    variants = [('Vision-only', False, 'blue'), ('Vision+Piezo', True, 'red')]
    active_archs = [ak for ak in arch_keys if ak in all_results]

    n_rows = len(active_archs)
    n_cols = len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, arch_key in enumerate(active_archs):
        for col_idx, col in enumerate(columns):
            ax = axes[row, col_idx]
            # 用 vision-only 的结果画图（两个变体共用同一组 y_true）
            key_vo = "Vision-only"
            key_vp = "Vision+Piezo"

            y_true = None
            if key_vo in all_results[arch_key]:
                y_true = all_results[arch_key][key_vo]['y_true']
                y_pred = all_results[arch_key][key_vo]['y_pred']
                ax.scatter(y_true[:, col_idx], y_pred[:, col_idx],
                          s=6, alpha=0.4, c='blue', edgecolors='none', label='Vision-only')

            if key_vp in all_results[arch_key] and y_true is not None:
                y_pred_vp = all_results[arch_key][key_vp]['y_pred']
                ax.scatter(y_true[:, col_idx], y_pred_vp[:, col_idx],
                          s=6, alpha=0.4, c='red', edgecolors='none', label='Vision+Piezo')

            vmin = y_true[:, col_idx].min() if y_true is not None else -1
            vmax = y_true[:, col_idx].max() if y_true is not None else 1
            margin = (vmax - vmin) * 0.05
            ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                    'k--', linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel(f'{arch_key}\n预测值', fontsize=8)
            if row == 0:
                ax.set_title(col, fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel('真实值', fontsize=7)
            if row == 0 and col_idx == n_cols - 1:
                ax.legend(fontsize=6, loc='upper left')
            ax.tick_params(labelsize=6)
            ax.set_aspect('equal', adjustable='box')

    fig.suptitle('跨架构散点图对比 (蓝=Vision-only, 红=Vision+Piezo)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_scatter.png"), dpi=150)
    plt.close(fig)
    print("  cross_model_scatter.png")


def plot_timeseries_comparison(all_results, columns, save_dir, max_samples=800):
    """所有模型的时间序列叠加（仅 fz 分量）"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    active_archs = [ak for ak in arch_keys if ak in all_results]

    # 取第一个可用模型的 y_true
    y_true = None
    for ak in active_archs:
        for key in ["Vision-only", "Vision+Piezo"]:
            if key in all_results[ak]:
                y_true = all_results[ak][key]['y_true']
                break
        if y_true is not None:
            break

    if y_true is None:
        return

    if len(y_true) > max_samples:
        idx = np.linspace(0, len(y_true) - 1, max_samples, dtype=int)
    else:
        idx = np.arange(len(y_true))

    fz_col = 2  # fz index

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(idx, y_true[idx, fz_col], 'k-', linewidth=1.5, alpha=0.9, label='真实值')

    colors = {'MLP': 'blue', 'PointNet': 'green', 'LightNet': 'orange'}
    for ak in active_archs:
        for var_label, ls in [('Vision-only', '--'), ('Vision+Piezo', '-')]:
            key = var_label
            if key in all_results[ak]:
                y_pred = all_results[ak][key]['y_pred']
                ax.plot(idx, y_pred[idx, fz_col], linestyle=ls, linewidth=1.0, alpha=0.7,
                       color=colors.get(ak, 'gray'), label=f'{ak} {var_label}')

    ax.set_xlabel('样本索引')
    ax.set_ylabel('fz (N)')
    ax.set_title('fz 时间序列对比 — 全部架构', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_timeseries.png"), dpi=150)
    plt.close(fig)
    print("  cross_model_timeseries.png")


def plot_r2_comparison(all_results, columns, save_dir):
    """R² 热力图对比：行=模型变体，列=力分量"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    variants = [('Vision-only', False), ('Vision+Piezo', True)]

    # 收集行标签和数据
    row_labels = []
    data = []
    for ak in arch_keys:
        if ak not in all_results:
            continue
        for vl, _ in variants:
            if vl not in all_results[ak]:
                continue
            row_labels.append(f"{ak} {vl}")
            data.append([all_results[ak][vl][col]['R2'] for col in columns])

    data = np.array(data)
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(11, 3.5))
    vmin = max(0, data.min() - 0.05)
    vmax = min(1, data.max() + 0.05)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

    # 单元格数字标注
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            # 参考中值决定文字颜色
            mid = (vmin + vmax) / 2
            color = 'white' if val > mid else 'black'
            ax.text(j, i, f"{val:.4f}", ha='center', va='center', fontsize=10,
                   color=color, fontweight='bold')

    # 画分隔线区分架构
    arch_boundaries = []
    idx = 0
    for ak in arch_keys:
        if ak in all_results:
            arch_boundaries.append(idx)
            idx += 2  # 每个架构占2行(Vision-only + Vision+Piezo)
    for b in arch_boundaries[1:]:
        ax.axhline(y=b - 0.5, color='gray', linewidth=1.2, linestyle='--', alpha=0.6)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(columns, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title('R$^2$ 热力图', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('R$^2$', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_r2.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  cross_model_r2.png")


def plot_comprehensive_improvement(all_results, columns, save_dir):
    """多指标改进热力图：MAE, RMSE, MAPE 的改进百分比 + R² 绝对提升"""
    arch_keys = ['MLP', 'PointNet', 'LightNet']
    active_archs = [ak for ak in arch_keys if ak in all_results]
    error_metrics = ['MAE', 'RMSE', 'MAPE(%)']
    n_archs = len(active_archs)
    n_cols = len(columns)

    # === 1. 误差指标改进百分比 (3 metrics × 6 components) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for m_idx, metric in enumerate(error_metrics):
        ax = axes[m_idx]
        data = np.zeros((n_archs, n_cols))
        annot = np.empty((n_archs, n_cols), dtype=object)
        for i, arch_key in enumerate(active_archs):
            for j, col in enumerate(columns):
                vo = all_results[arch_key].get("Vision-only", {}).get(col, {}).get(metric, None)
                vp = all_results[arch_key].get("Vision+Piezo", {}).get(col, {}).get(metric, None)
                if vo is not None and vp is not None and vo > 1e-10:
                    impr = (vo - vp) / vo * 100
                    data[i, j] = impr
                    annot[i, j] = f"{impr:+.1f}%"
                else:
                    data[i, j] = 0
                    annot[i, j] = "N/A"

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
        for i in range(n_archs):
            for j in range(n_cols):
                val = data[i, j]
                color = 'white' if abs(val) > 8 else 'black'
                ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=10,
                       color=color, fontweight='bold')
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(columns, fontsize=10)
        ax.set_yticks(range(n_archs))
        ax.set_yticklabels(active_archs, fontsize=10)
        ax.set_title(f'{metric} 改进 (正=更优)', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle('Vision+Piezo 相对 Vision-only 的改进', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_improvement.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  cross_model_improvement.png (已扩展为多指标)")

    # === 2. R² 绝对提升热力图 ===
    fig, ax = plt.subplots(figsize=(10, 2.5))
    r2_data = np.zeros((n_archs, n_cols))
    r2_annot = np.empty((n_archs, n_cols), dtype=object)
    for i, arch_key in enumerate(active_archs):
        for j, col in enumerate(columns):
            vo = all_results[arch_key].get("Vision-only", {}).get(col, {}).get('R2', None)
            vp = all_results[arch_key].get("Vision+Piezo", {}).get(col, {}).get('R2', None)
            if vo is not None and vp is not None:
                delta = vp - vo
                r2_data[i, j] = delta
                r2_annot[i, j] = f"{delta:+.4f}"
            else:
                r2_data[i, j] = 0
                r2_annot[i, j] = "N/A"

    abs_max = max(abs(r2_data.min()), abs(r2_data.max()), 0.01)
    im = ax.imshow(r2_data, cmap='RdYlGn', aspect='auto', vmin=-abs_max, vmax=abs_max)
    for i in range(n_archs):
        for j in range(n_cols):
            val = r2_data[i, j]
            color = 'white' if abs(val) > abs_max * 0.6 else 'black'
            ax.text(j, i, r2_annot[i, j], ha='center', va='center', fontsize=10,
                   color=color, fontweight='bold')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(columns, fontsize=10)
    ax.set_yticks(range(n_archs))
    ax.set_yticklabels(active_archs, fontsize=10)
    ax.set_title('R$^2$ 绝对提升 ($\\Delta$R$^2$, 正=更优)', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_r2_improvement.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  cross_model_r2_improvement.png")

    # === 3. MaxError 改进热力图 ===
    fig, ax = plt.subplots(figsize=(10, 2.5))
    me_data = np.zeros((n_archs, n_cols))
    me_annot = np.empty((n_archs, n_cols), dtype=object)
    for i, arch_key in enumerate(active_archs):
        for j, col in enumerate(columns):
            vo = all_results[arch_key].get("Vision-only", {}).get(col, {}).get('MaxError', None)
            vp = all_results[arch_key].get("Vision+Piezo", {}).get(col, {}).get('MaxError', None)
            if vo is not None and vp is not None and vo > 1e-10:
                impr = (vo - vp) / vo * 100
                me_data[i, j] = impr
                me_annot[i, j] = f"{impr:+.1f}%"
            else:
                me_data[i, j] = 0
                me_annot[i, j] = "N/A"

    im = ax.imshow(me_data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
    for i in range(n_archs):
        for j in range(n_cols):
            val = me_data[i, j]
            color = 'white' if abs(val) > 8 else 'black'
            ax.text(j, i, me_annot[i, j], ha='center', va='center', fontsize=10,
                   color=color, fontweight='bold')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(columns, fontsize=10)
    ax.set_yticks(range(n_archs))
    ax.set_yticklabels(active_archs, fontsize=10)
    ax.set_title('MaxError 改进 (正=更优)', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cross_model_maxerr_improvement.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  cross_model_maxerr_improvement.png")


# ──────────────────── 主入口 ────────────────────

def find_model_dirs(data_dir):
    """自动查找各架构模型目录"""
    dirs = {}
    specs = [
        ('MLP_Vision-only',     'model_output_vision_only'),
        ('MLP_Vision+Piezo',    'model_output_vision_piezo'),
        ('PointNet_Vision-only', 'model_output_pointnet'),
        ('PointNet_Vision+Piezo','model_output_pointnet_piezo'),
        ('LightNet_Vision-only', 'model_output_lightnet'),
        ('LightNet_Vision+Piezo','model_output_lightnet_piezo'),
    ]
    for label, subdir in specs:
        path = os.path.join(data_dir, subdir)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.pth")):
            dirs[label] = path
    return dirs


def main():
    parser = argparse.ArgumentParser(description="跨架构综合对比评估")
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--model_dirs', type=str, nargs='*', default=None,
                        help="手动指定6个模型目录（留空则自动检测）")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "force_calibration")

    if args.model_dirs:
        dirs = {}
        for path in args.model_dirs:
            name = os.path.basename(path)
            dirs[name] = path
    else:
        dirs = find_model_dirs(args.data_dir)

    if len(dirs) == 0:
        print(f"在 {args.data_dir} 中未找到任何模型目录")
        print("请先运行训练脚本生成模型")
        return

    print("=" * 60)
    print("V6 跨架构综合对比评估")
    print("=" * 60)
    print(f"\n找到 {len(dirs)} 个模型目录:")
    for label, path in dirs.items():
        print(f"  [{label}] {path}")

    # 逐个加载模型，收集结果
    all_results = {}  # {arch_key: {variant_key: {col: metrics, 'y_true':..., 'y_pred':...}}}
    common_test_idx = None
    columns = None

    for label, model_dir in dirs.items():
        # 解析 label: "MLP_Vision-only" → arch_key="MLP", variant="Vision-only"
        arch_key, variant = label.split('_', 1)
        print(f"\n--- 加载 [{label}] ---")
        try:
            model, scaler, dataset, test_idx, config, device, model_type = \
                load_model(model_dir, args.data_dir)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        if columns is None:
            columns = ['fx', 'fy', 'fz', 'mx', 'my', 'mz'][:config['output_dim']]

        # 统一测试集（取所有模型的交集）
        if common_test_idx is None:
            common_test_idx = test_idx
        else:
            common_test_idx = sorted(set(common_test_idx) & set(test_idx))

        if arch_key not in all_results:
            all_results[arch_key] = {}

        use_piezo = config.get('use_piezo', False)
        y_true, y_pred = predict_all(model, dataset, common_test_idx, scaler, device,
                                     model_type, use_piezo)
        metrics = compute_metrics(y_true, y_pred, columns)

        all_results[arch_key][variant] = {
            'y_true': y_true,
            'y_pred': y_pred,
            **metrics,
        }
        print(f"  测试集: {len(common_test_idx)} 样本, fz MAE={metrics['fz']['MAE']:.4f}")

    if columns is None:
        print("无可用模型")
        return

    print(f"\n最终共同测试集: {len(common_test_idx)} 样本")

    # ── 打印表格 ──
    print_comparison_table(all_results, columns)

    # ── 保存 JSON (扩展指标) ──
    eval_dir = os.path.join(args.data_dir, "cross_model_comparison")
    os.makedirs(eval_dir, exist_ok=True)

    json_out = {}
    for arch_key in all_results:
        json_out[arch_key] = {}
        for variant in all_results[arch_key]:
            cols_metrics = {col: {m: all_results[arch_key][variant][col][m]
                                  for m in ['MAE', 'RMSE', 'R2', 'MaxError', 'MAPE(%)']}
                           for col in columns}
            json_out[arch_key][variant] = cols_metrics

    with open(os.path.join(eval_dir, "cross_model_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存: cross_model_metrics.json")

    # ── 生成图表 ──
    print("\n生成图表:")
    plot_mae_bar_comparison(all_results, columns, eval_dir)
    plot_rmse_bar_comparison(all_results, columns, eval_dir)
    plot_radar_comparison(all_results, columns, eval_dir)
    plot_scatter_grid(all_results, columns, eval_dir)
    plot_comprehensive_improvement(all_results, columns, eval_dir)  # 替代旧版 improvement
    plot_r2_comparison(all_results, columns, eval_dir)              # 新增 R²
    plot_timeseries_comparison(all_results, columns, eval_dir)

    print(f"\n所有对比结果已保存到: {eval_dir}")


if __name__ == '__main__':
    main()
