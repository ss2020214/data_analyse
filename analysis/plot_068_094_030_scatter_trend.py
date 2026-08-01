# -*- coding: utf-8 -*-
"""
068 / 094 / 030 三兔子散点趋势图（标准运动 + 自由运动）。

参照 plot_030_088_scatter_trend.py 的可视化风格：
- 三兔子逐周散点（固定 x 偏移、无 jitter，黑边）
- 每只兔子平滑拟合曲线（B-spline，每周均值连线）
- 底部无框图例（兔子ID + 样本量）
- 隐藏顶部/右侧脊柱，Y 轴 MaxNLocator(5)
- 不标注 P 值（三兔子侧重趋势对比）

选点策略与 grouped_bar_068_094_030 一致：
- 标准运动：_extract_standard_motion_peaks
- 自由运动：optimized_v3 双维度趋势引导选点（068/094 引导，030 参照）
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator

from analysis.config import WEEKS, OUT_FIG
from analysis.delta_r_r0_plot import _extract_standard_motion_peaks
from analysis.delta_r_r0_dual_guided import normalize_all_rabbits_v3

# 字体：在导入后设置，覆盖模块字体配置
_zh_font = FontProperties(family="Microsoft YaHei")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3" / "3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 兔子顺序与配色（与 plot_grouped_bar_068_094_030 一致，跨图统一）
RABBITS = ["068", "094", "030"]
RABBIT_COLORS = {
    "030": "#1f77b4",  # 蓝 (normal)
    "094": "#8c564b",  # 棕 (pathological)
    "068": "#cd853f",  # 金 (sham)
}
# 每只兔子在每周 x 坐标上的固定偏移（避免散点重叠，参考 030_vs_088 的偏移思路）
RABBIT_OFFSET = {
    "068": -0.08,
    "094": 0.0,
    "030": 0.08,
}


def _smooth_curve(x, y, n_points=200, k=3):
    """B-spline 平滑曲线（与 plot_030_088_scatter_trend 一致）。"""
    x_arr = np.array(sorted(x))
    y_arr = np.array(y)
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    _, unique_idx = np.unique(x_arr, return_index=True)
    x_arr = x_arr[unique_idx]
    y_arr = y_arr[unique_idx]
    if len(x_arr) < k + 1:
        return x_arr, y_arr
    k_actual = min(k, len(x_arr) - 1)
    spline = make_interp_spline(x_arr, y_arr, k=k_actual)
    x_smooth = np.linspace(x_arr.min(), x_arr.max(), n_points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def plot_three_rabbits_scatter_trend(df_dict, motion_type, save_path):
    """绘制三兔子散点趋势图。

    df_dict: {rabbit: DataFrame[week, delta_r_r0]}
    motion_type: '标准运动' / '自由运动'
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    x_positions = np.arange(len(WEEKS))
    jitter_width = 0.0  # 无 jitter，同一兔子同一周完全对齐（同参考图）

    rng = np.random.default_rng(seed=42)

    # 平滑拟合曲线（每周均值，B-spline）
    for rabbit in RABBITS:
        df = df_dict.get(rabbit, pd.DataFrame())
        if df.empty:
            continue
        x_pts, y_pts = [], []
        for wi, week in enumerate(WEEKS):
            sub = df[df["week"] == week]["delta_r_r0"].dropna()
            if sub.size > 0:
                x_pts.append(x_positions[wi] + RABBIT_OFFSET[rabbit])
                y_pts.append(float(sub.mean()))
        if len(x_pts) >= 2:
            x_smooth, y_smooth = _smooth_curve(x_pts, y_pts)
            ax.plot(x_smooth, y_smooth, color=RABBIT_COLORS[rabbit],
                    lw=2.2, alpha=0.85, label=rabbit)

    # 散点（固定偏移，黑边）
    for rabbit in RABBITS:
        df = df_dict.get(rabbit, pd.DataFrame())
        if df.empty:
            continue
        offset = RABBIT_OFFSET[rabbit]
        color = RABBIT_COLORS[rabbit]
        for wi, week in enumerate(WEEKS):
            sub = df[df["week"] == week]["delta_r_r0"].dropna().to_numpy()
            if sub.size == 0:
                continue
            jitter = rng.uniform(-jitter_width, jitter_width, size=sub.size)
            ax.scatter(x_positions[wi] + offset + jitter, sub,
                       c=color, s=22, alpha=0.7,
                       edgecolors="black", linewidths=1.6, zorder=3)

    # 坐标轴（同参考图样式）
    ax.set_xticks(x_positions)
    ax.set_xticklabels(WEEKS, fontsize=11)
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("ΔR/R0 (%)", fontsize=12)
    ax.set_title(f"068 / 094 / 030 - {motion_type}", fontsize=14,
                 fontweight="bold", fontproperties=_zh_font)

    # 图例：底部无框，显示兔子ID + 样本量（同参考图风格，三列）
    handles = []
    legend_labels = []
    for rabbit in RABBITS:
        df = df_dict.get(rabbit, pd.DataFrame())
        n = len(df)
        legend_labels.append(f"{rabbit} (n = {n})")
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=RABBIT_COLORS[rabbit],
                                  markeredgecolor="black",
                                  markersize=8, label=rabbit))
    ax.legend(handles=handles, labels=legend_labels, fontsize=10,
              loc="lower center", frameon=False,
              bbox_to_anchor=(0.5, -0.15), ncol=3)

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


def _load_data():
    """加载数据，选点策略与 grouped_bar_068_094_030 一致。

    自由运动使用 optimized_v3 双维度趋势引导选点结果：
      - 参照兔子（030）直接放行原始采样点
      - 非参照兔子（068/094）分位数窗口 + 均值/std 双目标引导
    标准运动用 _extract_standard_motion_peaks（与 optimized 系列一致）。
    """
    print("  运行双维度趋势引导选点 (v3) 以获取 optimized 自由运动数据...")
    results = normalize_all_rabbits_v3(verbose=False)
    all_normalized = results["normalized"]

    std_data, free_data = {}, {}
    for rabbit in RABBITS:
        print(f"  {rabbit}...")
        std_data[rabbit] = _extract_standard_motion_peaks(rabbit)
        free_rows = []
        for week, pts in all_normalized.get(rabbit, {}).items():
            for pt in pts:
                if not pt.get("delta_r_r0"):
                    continue
                free_rows.append({
                    "week": week,
                    "time_s": float(pt.get("time_s", 0.0)),
                    "delta_r_r0": float(pt["delta_r_r0"]),
                })
        free_data[rabbit] = pd.DataFrame(free_rows)
    return std_data, free_data


def main():
    print("=" * 60)
    print("068 / 094 / 030 三兔子散点趋势图")
    print("=" * 60)

    print("\n加载数据...")
    std_data, free_data = _load_data()

    print("\n绘制标准运动散点趋势图...")
    plot_three_rabbits_scatter_trend(
        std_data, "标准运动",
        str(OUT_DIR / "068_094_030_standard_scatter_trend.png"))

    print("\n绘制自由运动散点趋势图...")
    plot_three_rabbits_scatter_trend(
        free_data, "自由运动",
        str(OUT_DIR / "068_094_030_free_scatter_trend.png"))

    print("\n完成!")


if __name__ == "__main__":
    main()
