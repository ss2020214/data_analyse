# -*- coding: utf-8 -*-
"""
Normal 组（020/075/030）标准运动 & 自由运动 mean±std 对比图。

将 optimized_v3 中三只兔子的 mean±std 子图合并到一张对比图中，
颜色与 3D 散点图一致。
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'KaiTi', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore", message=".*Font family.*not found")
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font")

from pathlib import Path
from .config import WEEKS, OUT_FIG

# ── 配色与偏移 ────────────────────────────────────────
# 正常组统一蓝色，组内通过透明度 alpha 分层区分不同兔子
NORMAL_RABBITS = ["020", "075", "030"]
RABBIT_COLORS = {
    "020": "#1f77b4",
    "075": "#1f77b4",
    "030": "#1f77b4",
}
# 组内透明度分层（三档：L1 最显著 → L3 最淡）
RABBIT_ALPHAS = {"020": 0.95, "075": 0.65, "030": 0.40}
RABBIT_OFFSETS = {"020": -0.15, "075": 0, "030": 0.15}

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _compute_weekly_stats(df, weeks):
    """计算每周 mean ± std。"""
    means, stds, ws = [], [], []
    for w in weeks:
        sub = df[df["week"] == w]["delta_r_r0"]
        if sub.size >= 2:
            means.append(sub.mean())
            stds.append(sub.std())
            ws.append(int(w[1:]))
    return ws, means, stds


def plot_comparison():
    from .delta_r_r0_plot import (
        _extract_standard_motion_peaks,
        _extract_free_motion_points,
    )
    # ── 加载数据 ──
    print("加载数据...")
    std_data, free_data = {}, {}
    for rabbit in NORMAL_RABBITS:
        print(f"  {rabbit}...")
        std_data[rabbit] = _extract_standard_motion_peaks(rabbit)
        free_data[rabbit] = _extract_free_motion_points(rabbit, n_per_week=30)

    # 字体重设（绘图前确保中文字体就位）
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'KaiTi', 'DejaVu Sans',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False

    # ══════════════════════════════════════════════════
    # 图1：标准运动 mean±std 对比
    # ══════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for rabbit in NORMAL_RABBITS:
        df = std_data[rabbit]
        if df.empty:
            continue
        ws, means, stds = _compute_weekly_stats(df, WEEKS)
        color = RABBIT_COLORS[rabbit]
        offset = RABBIT_OFFSETS[rabbit]
        ax1.errorbar(
            [w + offset for w in ws], means, yerr=stds,
            fmt='o-', color=color, capsize=4, lw=2, markersize=8,
            label=rabbit, alpha=RABBIT_ALPHAS[rabbit],
        )
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("ΔR/R0", fontsize=12)
    ax1.set_title("Normal 组 — 标准运动 ΔR/R0 (mean±std)", fontsize=14)
    ax1.set_xticks(range(1, 7))
    ax1.set_xticklabels([f"Week{i}" for i in range(1, 7)])
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    save1 = OUT_DIR / "normal_standard_comparison.png"
    fig1.savefig(save1, dpi=180)
    plt.close(fig1)
    print(f"  已保存: {save1}")

    # ══════════════════════════════════════════════════
    # 图2：自由运动 mean±std 对比
    # ══════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for rabbit in NORMAL_RABBITS:
        df = free_data[rabbit]
        if df.empty:
            continue
        ws, means, stds = _compute_weekly_stats(df, WEEKS)
        color = RABBIT_COLORS[rabbit]
        offset = RABBIT_OFFSETS[rabbit]
        ax2.errorbar(
            [w + offset for w in ws], means, yerr=stds,
            fmt='o-', color=color, capsize=4, lw=2, markersize=8,
            label=rabbit, alpha=RABBIT_ALPHAS[rabbit],
        )
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("ΔR/R0", fontsize=12)
    ax2.set_title("Normal 组 — 自由运动 ΔR/R0 (mean±std)", fontsize=14)
    ax2.set_xticks(range(1, 7))
    ax2.set_xticklabels([f"Week{i}" for i in range(1, 7)])
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    save2 = OUT_DIR / "normal_free_comparison.png"
    fig2.savefig(save2, dpi=180)
    plt.close(fig2)
    print(f"  已保存: {save2}")


def main():
    print("=" * 50)
    print("Normal 组 mean±std 对比图")
    print("=" * 50)
    plot_comparison()
    print("\n完成!")


if __name__ == "__main__":
    main()
