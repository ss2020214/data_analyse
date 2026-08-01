# -*- coding: utf-8 -*-
"""
Normal + Infection 组联合 3D 散点图包络线可视化。

将 normal 组（020/075/030）和 infection 组（016/065/088）
合并到同一坐标轴中，各生成三张 3D 图：
  1. 标准运动
  2. 自由运动
  3. 自由运动峰值

X 轴：normal 组在前，infection 组在后，组间留空隙。
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
from .config import GROUPS, WEEKS, OUT_FIG, RABBIT2GROUP, ALL_RABBITS

# ── 配置 ─────────────────────────────────────────────────
EXCLUDED_RABBITS = {"071"}

# 显示的两个组及顺序
DISPLAY_GROUPS = ["infection", "normal"]

# 兔子颜色：组间用色相区分（正常组=蓝，感染组=红），
# 组内同色，仅通过透明度 alpha 分层区分不同兔子
RABBIT_COLORS: dict[str, str] = {
    # 正常组：统一蓝色
    "030": "#1f77b4", "075": "#1f77b4", "020": "#1f77b4",
    # 感染组：统一红色
    "016": "#d62728", "065": "#d62728", "088": "#d62728",
}

# 组内透明度分层（三档：L1 最显著 → L3 最淡）
RABBIT_ALPHAS: dict[str, float] = {
    "020": 0.35, "075": 0.22, "030": 0.12,   # 正常组
    "088": 0.35, "065": 0.22, "016": 0.12,   # 感染组
}

# 包络线透明度分层（三档：L1 最显著 → L3 最淡）
RABBIT_ENV_ALPHA: dict[str, float] = {
    "020": 0.55, "075": 0.38, "030": 0.22,   # 正常组
    "088": 0.55, "065": 0.38, "016": 0.22,   # 感染组
}

# 图表类型
TYPE_LABELS = {
    "standard": "标准运动",
    "free":     "自由运动",
    "peaks":    "自由运动峰值",
}

OUT_3D_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3" / "3d"


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def _desaturate_hex(hex_color: str, blend: float = 0.70) -> tuple:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = r + (255 - r) * blend
    g2 = g + (255 - g) * blend
    b2 = b + (255 - b) * blend
    return (r2 / 255.0, g2 / 255.0, b2 / 255.0)


def _draw_envelope_for_rabbits(ax, df_all, rabbit_to_x, rabbit_env_colors, rabbit_env_alphas):
    """为所有兔子绘制 Y-Z 平面包络。"""
    x_half = 0.55

    for rabbit, x_pos in rabbit_to_x.items():
        env_color = rabbit_env_colors.get(rabbit, (0.5, 0.5, 0.5))
        fill_alpha = rabbit_env_alphas.get(rabbit, 0.55)
        sub_r = df_all[df_all["rabbit"] == rabbit]

        week_z: dict[int, float] = {}
        for week_num in range(1, 7):
            sub = sub_r[sub_r["week_num"] == week_num]
            if sub.empty:
                continue
            vals = sub["delta_r_r0"].dropna().to_numpy()
            vals_pos = vals[vals > 0]
            if len(vals_pos) < 2:
                continue
            z = float(np.max(vals_pos))
            if z > 0:
                week_z[week_num] = z

        if len(week_z) < 2:
            continue

        sorted_ws = sorted(week_z.keys())
        for i in range(len(sorted_ws) - 1):
            w1, w2 = sorted_ws[i], sorted_ws[i + 1]
            z1, z2 = week_z[w1], week_z[w2]
            y1, y2 = w1, w2
            face = Poly3DCollection(
                [[(x_pos, y1, 0), (x_pos, y2, 0),
                  (x_pos, y2, z2), (x_pos, y1, z1)]],
                alpha=fill_alpha, facecolor=env_color,
                edgecolor=env_color, linewidths=0.5,
            )
            ax.add_collection3d(face)


# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════

def _load_standard_data() -> dict[str, pd.DataFrame]:
    from .delta_r_r0_plot import _extract_standard_motion_peaks
    result = {}
    for g in DISPLAY_GROUPS:
        for r in GROUPS[g]:
            if r in EXCLUDED_RABBITS:
                continue
            try:
                df = _extract_standard_motion_peaks(r)
                if not df.empty:
                    result[r] = df
            except Exception as e:
                print(f"    [WARN] {r}: {e}")
    return result


def _load_free_data() -> dict[str, pd.DataFrame]:
    from .delta_r_r0_plot import _extract_free_motion_points
    result = {}
    for g in DISPLAY_GROUPS:
        for r in GROUPS[g]:
            if r in EXCLUDED_RABBITS:
                continue
            try:
                df = _extract_free_motion_points(r, n_per_week=30)
                if not df.empty:
                    result[r] = df
            except Exception as e:
                print(f"    [WARN] {r}: {e}")
    return result


def _load_peaks_data() -> dict[str, pd.DataFrame]:
    from .delta_r_r0_plot import _extract_free_motion_peaks_in_window
    result = {}
    for g in DISPLAY_GROUPS:
        for r in GROUPS[g]:
            if r in EXCLUDED_RABBITS:
                continue
            try:
                df = _extract_free_motion_peaks_in_window(r)
                if not df.empty:
                    result[r] = df
            except Exception as e:
                print(f"    [WARN] {r}: {e}")
    return result


def _build_df(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rabbit, df in data_dict.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            week_str = str(row["week"])
            week_num = int(week_str[1:])
            rows.append({
                "rabbit": rabbit,
                "group": RABBIT2GROUP.get(rabbit, ""),
                "week_num": week_num,
                "delta_r_r0": float(row["delta_r_r0"]),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 联合绘图
# ══════════════════════════════════════════════════════════

def plot_combined(df_all: pd.DataFrame, chart_type: str):
    _setup_chinese_font()

    if df_all.empty:
        print(f"  [SKIP] {chart_type}: 无数据")
        return

    # ── 构建 X 轴映射 ──
    rabbit_to_x: dict[str, float] = {}
    rabbit_groups: list[tuple[str, str]] = []  # (rabbit, group)
    x_current = 0.0
    x_spacing = 1.8
    group_gap = 1.2  # 组间间距

    for group_name in DISPLAY_GROUPS:
        rabbits = [
            r for r in ALL_RABBITS
            if RABBIT2GROUP.get(r) == group_name and r not in EXCLUDED_RABBITS
        ]
        # normal 组自定义顺序
        if group_name == "normal":
            rabbits = ["020"] + [r for r in rabbits if r not in ("020", "030")] + ["030"]
        # infection 组自定义顺序
        if group_name == "infection":
            rabbits = ["088", "065", "016"]
        for rabbit in rabbits:
            rabbit_to_x[rabbit] = x_current
            rabbit_groups.append((rabbit, group_name))
            x_current += x_spacing
        x_current += group_gap  # 组间空隙

    # ── 抖动 ──
    rng = np.random.default_rng(42)
    df_plot = df_all[df_all["rabbit"].isin(rabbit_to_x)].copy()
    df_plot["x"] = df_plot["rabbit"].map(rabbit_to_x)
    df_plot["x_jit"] = df_plot["x"] + rng.uniform(-0.30, 0.30, size=len(df_plot))
    df_plot["y_jit"] = df_plot["week_num"] + rng.uniform(-0.25, 0.25, size=len(df_plot))

    # ── Z 轴：每只兔子按自身最大值归一化到 0-100% ──
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
    z_label = "Normalized ΔR/R0 (%)"
    z_max = 105
    df_plot["delta_r_r0_clipped"] = df_plot["delta_r_r0"].clip(upper=z_max)

    # ── 创建 3D 图 ──
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")
    ax.set_box_aspect([1, 4, 0.6])

    # ── 包络：组内统一浅色，透明度分层区分 ──
    rabbit_env_colors = {
        r: _desaturate_hex(RABBIT_COLORS.get(r, "#808080"), 0.55)
        for r in rabbit_to_x
    }
    rabbit_env_alphas = {
        r: RABBIT_ENV_ALPHA.get(r, 0.55) for r in rabbit_to_x
    }
    _draw_envelope_for_rabbits(ax, df_env, rabbit_to_x, rabbit_env_colors, rabbit_env_alphas)

    # ── 散点 ──
    for rabbit, group_name in rabbit_groups:
        sub = df_plot[df_plot["rabbit"] == rabbit]
        if sub.empty:
            continue
        color = RABBIT_COLORS.get(rabbit, "gray")
        ax.scatter(
            sub["x_jit"], sub["y_jit"], sub["delta_r_r0_clipped"],
            c=[color], s=22, alpha=RABBIT_ALPHAS.get(rabbit, 0.10), edgecolors="black", linewidths=1.6,
            label=f"{rabbit}({group_name})", zorder=10,
        )

    # ── 轴 ──
    ax.set_xlabel("Rabbit ID", fontsize=12, labelpad=10)
    ax.set_ylabel("Time", fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    cn_type = TYPE_LABELS.get(chart_type, chart_type)
    ax.set_title(f"Infection + Normal — {cn_type}", fontsize=14, pad=18)

    x_ticks = [rabbit_to_x[r] for r, _ in rabbit_groups]
    x_labels = [r for r, _ in rabbit_groups]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10, rotation=25)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Week{i}" for i in range(1, 7)], fontsize=10)
    ax.set_zlim(0, z_max)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(1.6)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8,
              ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0), framealpha=0.7)

    ax.view_init(elev=18, azim=-30)

    # ── 保存 ──
    save_path = OUT_3D_DIR / f"3d_normal_infection_{chart_type}.png"
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def main():
    _setup_chinese_font()
    print("=" * 60)
    print("Normal + Infection 联合 3D 可视化")
    print("=" * 60)

    print("\n[1/3] 标准运动...")
    std = _load_standard_data()
    print(f"      共 {len(std)} 只兔子")

    print("\n[2/3] 自由运动...")
    free = _load_free_data()
    print(f"      共 {len(free)} 只兔子")

    print("\n[3/3] 自由运动峰值...")
    peaks = _load_peaks_data()
    print(f"      共 {len(peaks)} 只兔子")

    print("\n" + "=" * 60)
    print("生成联合 3D 图...")
    print("=" * 60)

    for chart_type, data_dict in [
        ("standard", std), ("free", free), ("peaks", peaks),
    ]:
        print(f"\n--- {TYPE_LABELS[chart_type]} ---")
        df = _build_df(data_dict)
        plot_combined(df, chart_type)

    print("\n完成!")


if __name__ == "__main__":
    main()
