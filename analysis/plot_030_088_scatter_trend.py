# -*- coding: utf-8 -*-
"""
030 vs 088 分组散点趋势图（标准运动 + 自由运动）。

完全模仿示例图样式：
- 030（蓝色）和 088（红色）的逐周散点
- 每组的平滑拟合曲线
- 每周两组比较的 P 值标注
- 图例在底部显示组名和样本量
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator

# 先导入模块（它们会设置自己的字体），然后覆盖
from analysis.config import WEEKS, OUT_FIG
from analysis.delta_r_r0_plot import _extract_standard_motion_peaks
from analysis.delta_r_r0_group_normalize import _extract_sampled_points

# 字体：在导入后设置，覆盖模块的字体配置
_zh_font = FontProperties(family="Microsoft YaHei")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 配色：与柱状图散点一致（030 灰，088 粉红）
COLOR_030 = "#B0B0B0"   # 灰色（对应柱状图 week1 色）
COLOR_088 = "#D46A6A"   # 粉红（对应柱状图 week6 色）


def _smooth_curve(x, y, n_points=100, k=3):
    """用 B-spline 生成平滑曲线。"""
    x_arr = np.array(sorted(x))
    y_arr = np.array(y)
    # 按 x 排序
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    # 去重
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


def _format_pvalue(p):
    """格式化 P 值显示。"""
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


def plot_grouped_scatter_trend(df_030: pd.DataFrame, df_088: pd.DataFrame,
                                motion_type: str, save_path: str):
    """绘制分组散点趋势图，完全模仿示例图样式。

    df_030 / df_088: 含列 [week, delta_r_r0] 的 DataFrame
    motion_type: '标准运动' 或 '自由运动'
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    weeks_numeric = [int(w[1:]) for w in WEEKS]
    x_positions = np.arange(len(WEEKS))

    # 不同兔子的固定偏移（030 偏左，088 偏右）
    offset_030 = -0.05
    offset_088 = 0.02
    jitter_width = 0.0  # 无 jitter，同一兔子同一周完全对齐

    rng = np.random.default_rng(seed=42)
    all_pvalues = []

    # 统计每周的样本量（用于图例）
    n_030_total = len(df_030)
    n_088_total = len(df_088)

    for wi, week in enumerate(WEEKS):
        # 提取该周数据
        sub_030 = df_030[df_030["week"] == week]["delta_r_r0"].dropna().to_numpy()
        sub_088 = df_088[df_088["week"] == week]["delta_r_r0"].dropna().to_numpy()

        # 散点（同一兔子同一周的点对齐同一 x 坐标，不同兔子有固定偏移）
        if sub_030.size > 0:
            jitter_030 = rng.uniform(-jitter_width, jitter_width, size=sub_030.size)
            ax.scatter(x_positions[wi] + offset_030 + jitter_030, sub_030,
                       c=COLOR_030, s=22, alpha=0.7, edgecolors="black", linewidths=1.6, zorder=3)
        if sub_088.size > 0:
            jitter_088 = rng.uniform(-jitter_width, jitter_width, size=sub_088.size)
            ax.scatter(x_positions[wi] + offset_088 + jitter_088, sub_088,
                       c=COLOR_088, s=22, alpha=0.7, edgecolors="black", linewidths=1.6, zorder=3)

        # 统计检验（Mann-Whitney U）
        if sub_030.size >= 2 and sub_088.size >= 2:
            try:
                stat, pval = mannwhitneyu(sub_030, sub_088, alternative="two-sided")
            except ValueError:
                pval = np.nan
        elif sub_030.size >= 2 or sub_088.size >= 2:
            # 样本量不足，用 t-test
            try:
                stat, pval = ttest_ind(sub_030, sub_088)
            except Exception:
                pval = np.nan
        else:
            pval = np.nan

        all_pvalues.append(pval)

        # P 值标注（模仿示例图：标注在时间点上方）
        if np.isfinite(pval):
            y_max = max(sub_030.max() if sub_030.size else 0,
                        sub_088.max() if sub_088.size else 0)
            y_min = min(sub_030.min() if sub_030.size else 0,
                        sub_088.min() if sub_088.size else 0)
            y_range = y_max - y_min if y_max > y_min else 1.0
            y_anno = y_max + 0.12 * y_range
            ax.text(x_positions[wi], y_anno, _format_pvalue(pval),
                    ha="center", va="bottom", fontsize=9,
                    color="#333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    # 平滑拟合曲线（模仿示例图：曲线宽度 2.2）
    for df_rabbit, color, label, offset in [(df_030, COLOR_030, "030", offset_030),
                                             (df_088, COLOR_088, "088", offset_088)]:
        x_pts, y_pts = [], []
        for wi, week in enumerate(WEEKS):
            sub = df_rabbit[df_rabbit["week"] == week]["delta_r_r0"].dropna()
            if sub.size > 0:
                x_pts.append(x_positions[wi] + offset)
                y_pts.append(float(sub.mean()))
        if len(x_pts) >= 2:
            x_smooth, y_smooth = _smooth_curve(x_pts, y_pts, n_points=200, k=min(3, len(x_pts)-1))
            ax.plot(x_smooth, y_smooth, color=color, lw=2.2, alpha=0.85, label=label)

    # 坐标轴（模仿示例图样式）
    ax.set_xticks(x_positions)
    ax.set_xticklabels(WEEKS, fontsize=11)
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("ΔR/R0 (%)", fontsize=12)
    ax.set_title(motion_type, fontsize=14, fontweight="bold", fontproperties=_zh_font)

    # 图例在底部（模仿示例图：显示组名和样本量，无框）
    ax.legend(fontsize=10, loc="lower center", frameon=False,
              bbox_to_anchor=(0.5, -0.15), ncol=2,
              labels=[f"030 (n = {n_030_total})", f"088 (n = {n_088_total})"])

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


def main():
    print("=" * 60)
    print("030 vs 088 分组散点趋势图")
    print("=" * 60)

    # 提取数据
    print("\n[1] 提取标准运动数据...")
    df_std_030 = _extract_standard_motion_peaks("030")
    df_std_088 = _extract_standard_motion_peaks("088")
    print(f"    030 标准运动: {len(df_std_030)} 个峰")
    print(f"    088 标准运动: {len(df_std_088)} 个峰")

    print("\n[2] 提取自由运动数据...")
    df_free_030 = _extract_sampled_points("030")
    df_free_088 = _extract_sampled_points("088")
    print(f"    030 自由运动: {len(df_free_030)} 个采样点")
    print(f"    088 自由运动: {len(df_free_088)} 个采样点")

    # 绘图
    print("\n[3] 绘制标准运动图...")
    plot_grouped_scatter_trend(
        df_std_030, df_std_088,
        "标准运动",
        str(OUT_DIR / "030_vs_088_standard_scatter_trend.png"))

    print("\n[4] 绘制自由运动图...")
    plot_grouped_scatter_trend(
        df_free_030, df_free_088,
        "自由运动",
        str(OUT_DIR / "030_vs_088_free_scatter_trend.png"))

    print("\n完成!")


if __name__ == "__main__":
    main()
