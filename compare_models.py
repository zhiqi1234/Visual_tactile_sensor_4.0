import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 加载三个模型的metrics
models = {
    "BaseNet": "force_calibration/model_output/evaluation/metrics.json",
    "LightNet": "force_calibration/model_output_lightnet/evaluation/metrics.json",
    "PointNet": "force_calibration/model_output_pointnet/evaluation/metrics.json",
}

data = {}
for name, path in models.items():
    with open(path, "r") as f:
        data[name] = json.load(f)

channels = ["fx", "fy", "fz", "mx", "my", "mz"]
metrics = ["MAE", "RMSE", "R2", "MaxError"]
colors = ["#4C72B0", "#DD8452", "#55A868"]
model_names = list(models.keys())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Model Comparison: BaseNet vs LightNet vs PointNet", fontsize=16, fontweight="bold")

metric_labels = {"MAE": "MAE (↓ better)", "RMSE": "RMSE (↓ better)", "R2": "R² (↑ better)", "MaxError": "Max Error (↓ better)"}

x = np.arange(len(channels))
width = 0.25

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2][idx % 2]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        vals = [data[model_name][ch][metric] for ch in channels]
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=color, alpha=0.85, edgecolor="white")
        # 标注数值
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_title(metric_labels[metric], fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(channels, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Channel")

plt.tight_layout()
plt.savefig("force_calibration_2/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Radar chart for R2 ----
fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(channels), endpoint=False).tolist()
angles += angles[:1]

for model_name, color in zip(model_names, colors):
    vals = [data[model_name][ch]["R2"] for ch in channels]
    vals += vals[:1]
    ax2.plot(angles, vals, color=color, linewidth=2, label=model_name)
    ax2.fill(angles, vals, color=color, alpha=0.15)

ax2.set_thetagrids(np.degrees(angles[:-1]), channels, fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_title("R² Radar Chart (↑ better)", fontsize=13, pad=20)
ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
ax2.grid(True)

plt.tight_layout()
plt.savefig("force_calibration_2/model_comparison_radar.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Summary table ----
print("\n===== 各模型平均指标 =====")
print(f"{'Model':<12} {'Avg MAE':>10} {'Avg RMSE':>10} {'Avg R2':>10} {'Avg MaxErr':>12}")
print("-" * 56)
for model_name in model_names:
    avg_mae  = np.mean([data[model_name][ch]["MAE"]      for ch in channels])
    avg_rmse = np.mean([data[model_name][ch]["RMSE"]     for ch in channels])
    avg_r2   = np.mean([data[model_name][ch]["R2"]       for ch in channels])
    avg_max  = np.mean([data[model_name][ch]["MaxError"] for ch in channels])
    print(f"{model_name:<12} {avg_mae:>10.4f} {avg_rmse:>10.4f} {avg_r2:>10.4f} {avg_max:>12.4f}")
