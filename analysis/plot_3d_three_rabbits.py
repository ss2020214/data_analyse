# -*- coding: utf-8 -*-
"""
三只代表兔子 3D 联合对比图：030 (normal) / 086 (pathological) / 068 (sham)。

从后到前排列，每只兔子按自身最大值归一化到 0-100%。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import warnings
warnings.filterwarnings("ignore", message=".*Font family.*not found")
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font")

def _setup_chinese_font():
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'KaiTi', 'DejaVu Sans',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from .config import WEEKS, OUT_FIG, RABBIT2GROUP

# ── 三只兔子（从后到前）─────────────────────────────────
RABBITS = ["086", "068", "030"]
RABBIT_COLORS = {
    "030": "#1f77b4",  # normal
    "086": "#9467bd",  # pathological
    "068": "#cd853f",  # sham
}

TYPE_LABELS = {
    "standard": "标准运动",
    "free":     "自由运动",
    "peaks":    "自由运动峰值",
}

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3" / "3d"


def _desaturate_hex(hex_color: str, blend: float = 0.70) -> tuple:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = r + (255 - r) * blend
    g2 = g + (255 - g) * blend
    b2 = b + (255 - b) * blend
    return (r2 / 255.0, g2 / 255.0, b2 / 255.0)


def _draw_envelope(ax, df_all, rabbit_to_x, rabbit_env_colors):
    fill_alpha = 0.55
    for rabbit, x_pos in rabbit_to_x.items():
        env_color = rabbit_env_colors.get(rabbit, (0.5, 0.5, 0.5))
        sub_r = df_all[df_all["rabbit"] == rabbit]
        week_z: dict[int, float] = {}
        for wn in range(1, 7):
            sub = sub_r[sub_r["week_num"] == wn]
            if sub.empty:
                continue
            vals = sub["delta_r_r0"].dropna().to_numpy()
            vals_pos = vals[vals > 0]
            if len(vals_pos) < 2:
                continue
            z = float(np.max(vals_pos))
            if z > 0:
                week_z[wn] = z
        if len(week_z) < 2:
            continue
        sw = sorted(week_z.keys())
        for i in range(len(sw) - 1):
            w1, w2 = sw[i], sw[i + 1]
            z1, z2 = week_z[w1], week_z[w2]
            face = Poly3DCollection(
                [[(x_pos, w1, 0), (x_pos, w2, 0),
                  (x_pos, w2, z2), (x_pos, w1, z1)]],
                alpha=fill_alpha, facecolor=env_color,
                edgecolor=env_color, linewidths=0.5,
            )
            ax.add_collection3d(face)


def _load_data():
    from .delta_r_r0_plot import (
        _extract_standard_motion_peaks,
        _extract_free_motion_points,
        _extract_free_motion_peaks_in_window,
    )
    std, free, peaks = {}, {}, {}
    for r in RABBITS:
        print(f"  {r}...")
        std[r] = _extract_standard_motion_peaks(r)
        free[r] = _extract_free_motion_points(r, n_per_week=30)
        peaks[r] = _extract_free_motion_peaks_in_window(r)
    return std, free, peaks


def _to_df(data_dict):
    rows = []
    for rabbit, df in data_dict.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                "rabbit": rabbit,
                "week_num": int(str(row["week"])[1:]),
                "delta_r_r0": float(row["delta_r_r0"]),
            })
    return pd.DataFrame(rows)


def plot_one(df_all, chart_type):
    _setup_chinese_font()
    if df_all.empty:
        return

    # X 轴
    rabbit_to_x = {}
    rabbit_list = []
    x_spacing = 1.8
    for i, r in enumerate(RABBITS):
        rabbit_to_x[r] = i * x_spacing
        rabbit_list.append(r)

    # 抖动
    rng = np.random.default_rng(42)
    df_plot = df_all[df_all["rabbit"].isin(rabbit_to_x)].copy()
    df_plot["x"] = df_plot["rabbit"].map(rabbit_to_x)
    df_plot["x_jit"] = df_plot["x"] + rng.uniform(-0.30, 0.30, size=len(df_plot))
    df_plot["y_jit"] = df_plot["week_num"] + rng.uniform(-0.25, 0.25, size=len(df_plot))

    # Z 轴：每只兔子归一化
    df_env = df_all[df_all["rabbit"].isin(rabbit_to_x)].copy()
    for rabbit in rabbit_to_x:
        mask = df_plot["rabbit"] == rabbit
        vals = df_plot.loc[mask, "delta_r_r0"]
        vmax = float(vals.max())
        if vmax > 0:
            df_plot.loc[mask, "delta_r_r0"] = vals / vmax * 100
        mask_env = df_env["rabbit"] == rabbit
        vals_env = df_env.loc[mask_env, "delta_r_r0"]
        vmax_env = float(vals_env.max())
        if vmax_env > 0:
            df_env.loc[mask_env, "delta_r_r0"] = vals_env / vmax_env * 100
    z_max = 105
    df_plot["delta_r_r0_clipped"] = df_plot["delta_r_r0"].clip(upper=z_max)

    # 图
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")
    ax.set_box_aspect([1, 4, 0.6])

    env_colors = {r: _desaturate_hex(RABBIT_COLORS[r]) for r in rabbit_to_x}
    _draw_envelope(ax, df_env, rabbit_to_x, env_colors)

    for rabbit in rabbit_list:
        sub = df_plot[df_plot["rabbit"] == rabbit]
        if sub.empty:
            continue
        ax.scatter(sub["x_jit"], sub["y_jit"], sub["delta_r_r0_clipped"],
                   c=[RABBIT_COLORS[rabbit]], s=28, alpha=0.10,
                   edgecolors="none", label=rabbit, zorder=10)

    ax.set_xlabel("Rabbit ID", fontsize=12, labelpad=10)
    ax.set_ylabel("Time", fontsize=12, labelpad=10)
    ax.set_zlabel("Normalized ΔR/R0 (%)", fontsize=12, labelpad=10)
    cn = TYPE_LABELS.get(chart_type, chart_type)
    ax.set_title(f"086 / 068 / 030 — {cn}", fontsize=14, pad=18)

    x_ticks = [rabbit_to_x[r] for r in rabbit_list]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(rabbit_list, fontsize=10, rotation=25)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Week{i}" for i in range(1, 7)], fontsize=10)
    ax.set_zlim(0, z_max)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, ncol=3,
              loc="upper left", bbox_to_anchor=(0.0, 1.0), framealpha=0.7)
    ax.view_init(elev=18, azim=-30)

    save_path = OUT_DIR / f"3d_030_086_068_{chart_type}.png"
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")


def main():
    _setup_chinese_font()
    print("=" * 60)
    print("030 / 086 / 068 联合 3D 可视化")
    print("=" * 60)

    print("\n加载数据...")
    std, free, peaks = _load_data()

    for ct, dd in [("standard", std), ("free", free), ("peaks", peaks)]:
        print(f"\n--- {TYPE_LABELS[ct]} ---")
        plot_one(_to_df(dd), ct)

    print("\n完成!")


if __name__ == "__main__":
    main()
