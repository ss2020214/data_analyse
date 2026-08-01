# -*- coding: utf-8 -*-
"""
086 / 068 / 030 三兔 mean±std 对比图。

标准运动和自由运动，颜色与 3D 图一致。
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

RABBITS = ["094", "068", "030"]
COLORS = {"094": "#8c564b", "068": "#cd853f", "030": "#1f77b4"}
OFFSETS = {"094": -0.15, "068": 0, "030": 0.15}

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _weekly_stats(df):
    means, stds, ws = [], [], []
    for w in WEEKS:
        sub = df[df["week"] == w]["delta_r_r0"]
        if sub.size >= 2:
            means.append(sub.mean()); stds.append(sub.std())
            ws.append(int(w[1:]))
    return ws, means, stds


def main():
    from .delta_r_r0_plot import (
        _extract_standard_motion_peaks,
        _extract_free_motion_points,
    )

    print("加载数据...")
    std_data, free_data = {}, {}
    for r in RABBITS:
        print(f"  {r}...")
        std_data[r] = _extract_standard_motion_peaks(r)
        free_data[r] = _extract_free_motion_points(r, n_per_week=30)

    # 字体重设
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'KaiTi', 'DejaVu Sans',
    ]

    for ct, dd, fn in [
        ("standard", std_data, "094_068_030_standard_comparison.png"),
        ("free", free_data, "094_068_030_free_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in RABBITS:
            df = dd.get(r)
            if df is None or df.empty:
                continue
            ws, means, stds = _weekly_stats(df)
            ax.errorbar(
                [w + OFFSETS[r] for w in ws], means, yerr=stds,
                fmt='o-', color=COLORS[r], capsize=4, lw=2, markersize=8,
                label=r, alpha=0.65,
            )
        title = "标准运动" if ct == "standard" else "自由运动"
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("ΔR/R0", fontsize=12)
        ax.set_title(f"094 / 068 / 030 — {title} ΔR/R0 (mean±std)", fontsize=14)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([f"Week{i}" for i in range(1, 7)])
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        sp = OUT_DIR / fn
        fig.savefig(sp, dpi=180)
        plt.close(fig)
        print(f"  已保存: {sp}")

    print("完成!")


if __name__ == "__main__":
    main()
