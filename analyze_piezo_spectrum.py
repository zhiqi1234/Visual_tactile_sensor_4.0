# -*- coding: utf-8 -*-
"""压电信号频谱分析 — 查看噪声主频"""
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze(filepath, adc_group=2, channel=4, max_freq=500):
    """分析指定 ADC 组和通道的频谱"""
    print(f"文件: {filepath}")
    print(f"ADC{adc_group} CH{channel}")

    with h5py.File(filepath, 'r') as f:
        grp = f[f'piezo_stream/adc{adc_group}']
        ts = grp['timestamp'][:]
        raw = grp['values'][:]  # (N, 8)
        vals = raw[:, channel - 1].astype(np.float64)  # 1-indexed → 0-indexed

    # 估算采样率（Windows time.time() 精度约15ms，直接用总数/总时长更可靠）
    t_rel = ts - ts[0]
    total_dur = max(t_rel[-1], 1e-9)
    fs = len(vals) / total_dur  # 总数/时长，不受时间戳精度影响
    fs = min(fs, 20000.0)
    print(f"采样数: {len(vals)},  时长: {t_rel[-1]:.1f}s,  采样率: {fs:.0f} Hz")

    # ── 时域 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_time = axes[0, 0]
    ax_time.plot(t_rel, vals, linewidth=0.3)
    ax_time.set_xlabel('时间 (s)')
    ax_time.set_ylabel('电压 (V)')
    ax_time.set_title(f'ADC{adc_group} CH{channel} 原始波形')
    ax_time.grid(True, alpha=0.3)

    # ── 频谱（Welch 方法） ──
    nperseg = min(4096, len(vals) // 4)
    freqs, psd = signal.welch(vals, fs, nperseg=nperseg)

    ax_psd = axes[0, 1]
    ax_psd.semilogy(freqs, psd, linewidth=0.8)
    ax_psd.set_xlabel('频率 (Hz)')
    ax_psd.set_ylabel('功率谱密度 (V^2/Hz)')
    ax_psd.set_title('功率谱密度 (PSD)')
    ax_psd.grid(True, alpha=0.3)

    # 标记工频
    for hz, color in [(50, 'red'), (100, 'orange'), (150, 'gold')]:
        if hz <= max_freq:
            ax_psd.axvline(hz, color=color, linestyle='--', alpha=0.5, label=f'{hz}Hz' if hz == 50 else None)
    ax_psd.legend(fontsize=8)

    # ── 放大低频段 (0-200Hz) ──
    ax_low = axes[1, 0]
    mask_low = freqs <= 200
    ax_low.semilogy(freqs[mask_low], psd[mask_low], linewidth=0.8)
    for hz, color in [(50, 'red'), (100, 'orange'), (150, 'gold')]:
        ax_low.axvline(hz, color=color, linestyle='--', alpha=0.5)
    ax_low.set_xlabel('频率 (Hz)')
    ax_low.set_ylabel('功率谱密度 (V^2/Hz)')
    ax_low.set_title('低频段 0-200Hz')
    ax_low.grid(True, alpha=0.3)

    # ── 找主要峰值 ──
    ax_peaks = axes[1, 1]
    mask = freqs <= max_freq
    f_sub, p_sub = freqs[mask], psd[mask]

    # 找前10个最高峰
    from scipy.signal import find_peaks
    peak_idx, peak_props = find_peaks(p_sub, height=np.median(p_sub) * 2, distance=max(1, int(fs / nperseg * 5)))
    peak_freqs = f_sub[peak_idx]
    peak_heights = p_sub[peak_idx]
    top_k = min(10, len(peak_freqs))
    top_idx = np.argsort(peak_heights)[-top_k:]

    ax_peaks.bar(range(top_k), peak_heights[top_idx], color='steelblue')
    ax_peaks.set_xticks(range(top_k))
    ax_peaks.set_xticklabels([f'{peak_freqs[i]:.1f}' for i in top_idx], rotation=45)
    ax_peaks.set_xlabel('频率 (Hz)')
    ax_peaks.set_ylabel('功率谱密度')
    ax_peaks.set_title(f'前 {top_k} 个主频峰 (0-{max_freq}Hz)')
    ax_peaks.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    # 打印峰值表
    print(f"\n{'='*50}")
    print(f"  前 {top_k} 个主频峰 (0-{max_freq}Hz)")
    print(f"{'='*50}")
    print(f"{'频率 (Hz)':>12s}  {'PSD (V^2/Hz)':>15s}")
    print(f"{'-'*30}")
    for i in reversed(top_idx):
        is_50hz = abs(peak_freqs[i] - 50) < 2 or abs(peak_freqs[i] - 100) < 2 or abs(peak_freqs[i] - 150) < 2
        marker = ' ← 工频?' if is_50hz else ''
        print(f"{peak_freqs[i]:12.2f}  {peak_heights[i]:15.6e}{marker}")

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        fpath = sys.argv[1]
        adc = int(sys.argv[2]) if len(sys.argv) >= 3 else 2
        ch = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
    else:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        fpath = filedialog.askopenfilename(
            title="选择点云 HDF5 文件（含 piezo_stream）",
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_calibration"),
            filetypes=[("HDF5 文件", "*.h5 *.hdf5"), ("所有文件", "*.*")])
        root.destroy()
        if not fpath:
            print("未选择文件，退出")
            sys.exit(0)

        # 交互式选择 ADC 组和通道
        while True:
            try:
                adc = int(input("选择 ADC 组 (1-5，默认=2): ").strip() or "2")
                if 1 <= adc <= 5:
                    break
            except ValueError:
                pass
            print("  请输入 1-5")

        while True:
            try:
                ch = int(input("选择通道 (1-8，默认=4): ").strip() or "4")
                if 1 <= ch <= 8:
                    break
            except ValueError:
                pass
            print("  请输入 1-8")
    max_hz = int(sys.argv[4]) if len(sys.argv) >= 5 else 500

    analyze(fpath, adc_group=adc, channel=ch, max_freq=max_hz)
