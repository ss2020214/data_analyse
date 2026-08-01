# -*- coding: utf-8 -*-
"""
030 vs 088 分组柱状图（标准运动 + 自由运动）。

样式：
- 窄柱子（bar_width ~ 0.18）
- 兔子 ID 标注在 x 轴下方
- 图例在绘图区上方
- 无显著性标注
- 空心圆散点叠加
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from matplotlib.ticker import MaxNLocator

# 先导入模块（它们会设置自己的字体），然后覆盖
from analysis.config import WEEKS, OUT_FIG
from analysis.delta_r_r0_plot import _extract_standard_motion_peaks
from analysis.delta_r_r0_group_normalize import _extract_sampled_points

# 字体
_zh_font = FontProperties(family="Microsoft YaHei")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 配色：与示例图一致（灰色 + 粉红/红）
COLOR_W1 = "#B0B0B0"   # 灰色
COLOR_W6 = "#D46A6A"   # 偏粉红


def _significance_stars(p):
    """P 值 -> 星号标注。"""
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _jitter_scatter(ax, x_center, values, bar_half_width, size=22, alpha=0.9,
                    edgecolor="black", linewidth=0.7, seed=42):
    """在 x_center 附近绘制带 jitter 的空心圆散点，jitter 限制在柱子宽度内。"""
    if len(values) == 0:
        return
    rng = np.random.default_rng(seed=seed)
    n = len(values)
    jitter = rng.uniform(-bar_half_width * 0.85, bar_half_width * 0.85, size=n)
    ax.scatter(x_center + jitter, values,
               facecolors="none", edgecolors=edgecolor,
               linewidths=linewidth, s=size, alpha=alpha, zorder=5)


def plot_grouped_bar(rabbit_labels: list[str],
                     data_w1: dict, data_w6: dict,
                     motion_type: str, save_path: str):
    """绘制分组柱状图。"""
    fig, ax = plt.subplots(figsize=(6, 9))

    n_rabbits = len(rabbit_labels)
    bar_width = 0.02
    intra_gap = bar_width / 4       # 组内柱子间隔 = 1/4 柱宽
    inter_gap = 5 * bar_width / 4   # 组间柱子间隔 = 5/4 柱宽
    # 兔子 030 中心在 0，088 中心 = 柱宽 + 组内间隔 + 组间间隔
    x_base = np.array([0, bar_width + intra_gap + inter_gap])
    x_w1 = x_base - bar_width / 2 - intra_gap / 2
    x_w6 = x_base + bar_width / 2 + intra_gap / 2
    bar_half_width = bar_width / 2  # 用于限制散点 jitter 范围

    means_w1, means_w6 = [], []
    stds_w1, stds_w6 = [], []
    pvalues = []

    for ri, rabbit in enumerate(rabbit_labels):
        v1 = np.array(data_w1.get(rabbit, []))
        v6 = np.array(data_w6.get(rabbit, []))

        m1 = float(v1.mean()) if v1.size else 0
        m6 = float(v6.mean()) if v6.size else 0
        s1 = float(v1.std(ddof=1)) if v1.size >= 2 else 0
        s6 = float(v6.std(ddof=1)) if v6.size >= 2 else 0

        means_w1.append(m1)
        means_w6.append(m6)
        stds_w1.append(s1)
        stds_w6.append(s6)

        # 统计检验
        if v1.size >= 2 and v6.size >= 2:
            try:
                _, p = mannwhitneyu(v1, v6, alternative="two-sided")
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        pvalues.append(p)

    # 绘制柱子（空白填充 + 黑色加粗边缘）
    ax.bar(x_w1, means_w1, bar_width,
           facecolor="none", edgecolor="black", linewidth=1.6,
           zorder=2)
    ax.bar(x_w6, means_w6, bar_width,
           facecolor="none", edgecolor="black", linewidth=1.6,
           zorder=2)

    # 误差棒（单独绘制，更精细）
    for ri in range(n_rabbits):
        ax.errorbar(x_w1[ri], means_w1[ri], yerr=stds_w1[ri],
                    fmt="none", color="black", capsize=3, lw=0.8, zorder=3)
        ax.errorbar(x_w6[ri], means_w6[ri], yerr=stds_w6[ri],
                    fmt="none", color="black", capsize=3, lw=0.8, zorder=3)

    # 叠加个体散点（空心圆，jitter 严格限制在柱子宽度内）
    # 030 散点用 week1 颜色（灰），088 散点用 week6 颜色（粉红）
    jitter_limit = bar_width * 0.4  # jitter 范围 = 柱子宽度的 40%
    for ri, rabbit in enumerate(rabbit_labels):
        scatter_color = COLOR_W1 if rabbit == "030" else COLOR_W6
        v1 = np.array(data_w1.get(rabbit, []))
        v6 = np.array(data_w6.get(rabbit, []))
        if v1.size > 0:
            rng1 = np.random.default_rng(seed=hash(rabbit + "w1") & 0xFFFF)
            j1 = rng1.uniform(-jitter_limit, jitter_limit, size=v1.size)
            ax.scatter(x_w1[ri] + j1, v1, facecolors=scatter_color, edgecolors="black",
                       linewidths=1.6, s=22, alpha=0.9, zorder=5)
        if v6.size > 0:
            rng6 = np.random.default_rng(seed=hash(rabbit + "w6") & 0xFFFF)
            j6 = rng6.uniform(-jitter_limit, jitter_limit, size=v6.size)
            ax.scatter(x_w6[ri] + j6, v6, facecolors=scatter_color, edgecolors="black",
                       linewidths=1.6, s=22, alpha=0.9, zorder=5)

    # 计算 y 轴范围
    y_max_all = max(
        max(means_w1[ri] + stds_w1[ri] for ri in range(n_rabbits)),
        max(means_w6[ri] + stds_w6[ri] for ri in range(n_rabbits)),
    )
    y_range = max(means_w1 + means_w6 + [1e-6])

    # 坐标轴
    ax.set_xticks(x_base)
    # 隐藏默认 x 轴标签，用手动 text 代替以控制位置
    ax.set_xticklabels([])
    ax.set_xlabel("")

    # 每个柱子下方标注 w1 / w6（黑色字体）
    for ri in range(n_rabbits):
        ax.text(x_w1[ri], -0.03, "w1", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="black", clip_on=False)
        ax.text(x_w6[ri], -0.03, "w6", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="black", clip_on=False)

    # 030 / 088 标签下移
    for ri in range(n_rabbits):
        ax.text(x_base[ri], -0.10, rabbit_labels[ri],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=13, clip_on=False)
    ax.set_ylabel("ΔR/R0 (%)", fontsize=11)
    ax.set_title(motion_type, fontsize=13, fontweight="bold",
                 fontproperties=_zh_font, pad=8)

    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    # 调整 y 轴上限 + 5 个刻度
    ax.set_ylim(bottom=0, top=y_max_all + 0.15 * y_range)
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.subplots_adjust(bottom=0.14, top=0.88)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("030 vs 088 分组柱状图（Week 1 vs Week 6）")
    print("=" * 60)

    rabbit_labels = ["030", "088"]

    # 提取数据
    print("\n[1] 提取标准运动数据...")
    df_std_030 = _extract_standard_motion_peaks("030")
    df_std_088 = _extract_standard_motion_peaks("088")

    print("\n[2] 提取自由运动数据...")
    df_free_030 = _extract_sampled_points("030")
    df_free_088 = _extract_sampled_points("088")

    # 整理数据
    def _get_week_vals(df, rabbit, week):
        sub = df[df["week"] == week]["delta_r_r0"].dropna().to_numpy()
        return sub.tolist()

    std_w1 = {r: _get_week_vals(df_std_030 if r == "030" else df_std_088, r, "w1")
              for r in rabbit_labels}
    std_w6 = {r: _get_week_vals(df_std_030 if r == "030" else df_std_088, r, "w6")
              for r in rabbit_labels}
    free_w1 = {r: _get_week_vals(df_free_030 if r == "030" else df_free_088, r, "w1")
               for r in rabbit_labels}
    free_w6 = {r: _get_week_vals(df_free_030 if r == "030" else df_free_088, r, "w6")
               for r in rabbit_labels}

    print(f"\n  标准运动: 030 w1={len(std_w1['030'])}, w6={len(std_w6['030'])} | "
          f"088 w1={len(std_w1['088'])}, w6={len(std_w6['088'])}")
    print(f"  自由运动: 030 w1={len(free_w1['030'])}, w6={len(free_w6['030'])} | "
          f"088 w1={len(free_w1['088'])}, w6={len(free_w6['088'])}")

    # 绘图
    print("\n[3] 绘制标准运动柱状图...")
    plot_grouped_bar(
        rabbit_labels, std_w1, std_w6,
        "标准运动",
        str(OUT_DIR / "030_vs_088_standard_bar.png"))

    print("\n[4] 绘制自由运动柱状图...")
    plot_grouped_bar(
        rabbit_labels, free_w1, free_w6,
        "自由运动",
        str(OUT_DIR / "030_vs_088_free_bar.png"))

    print("\n完成!")


if __name__ == "__main__":
    main()
