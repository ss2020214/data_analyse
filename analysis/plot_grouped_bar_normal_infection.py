# -*- coding: utf-8 -*-
"""
Normal + Infection 分组柱状图（标准运动 & 自由运动）。

X轴结构：
  大组：正常组（Normal） / 感染组（Infection）
  每组内：各只兔子
  每只兔子内：week1 / week3 / week6 三根小柱

柱上叠加每个样本的散点（按周数着色），带 mean+/-SEM 误差线，
组间添加 Mann-Whitney U 检验 P 值及显著性星号标注。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from scipy import stats
import warnings

warnings.filterwarnings("ignore", message=".*Font family.*not found")
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font")


def _setup_chinese_font():
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "KaiTi", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


from pathlib import Path
from .config import GROUPS, WEEKS, OUT_FIG, RABBIT2GROUP

# ── 兔子顺序 ─────────────────────────────────────────────
NORMAL_RABBITS = ["020", "075", "030"]
INFECTION_RABBITS = ["088", "065", "016"]
SELECTED_WEEKS = ["w1", "w2", "w3", "w4", "w5", "w6"]
WEEK_LABELS = {"w1": "w1", "w2": "w2", "w3": "w3", "w4": "w4", "w5": "w5", "w6": "w6"}

# ── 周数配色 ─────────────────────────────────────────────
WEEK_COLORS = {
    "w1": "#FF8C00",  # 橙
    "w3": "#228B22",  # 绿
    "w6": "#9932CC",  # 紫
}

# ── 组配色 ───────────────────────────────────────────────
GROUP_COLORS = {
    "normal": "#1f77b4",    # 蓝
    "infection": "#d62728",  # 红
}
GROUP_LABELS = {
    "normal": "Normal",
    "infection": "Infection",
}

# ── 兔子配色（与 3d_normal_infection 统一）──
RABBIT_COLORS: dict[str, str] = {
    # 正常组：统一蓝色
    "030": "#1f77b4", "075": "#1f77b4", "020": "#1f77b4",
    # 感染组：统一红色
    "016": "#d62728", "065": "#d62728", "088": "#d62728",
}

# ── 组内透明度分层（与 3d_normal_infection 一致）──
RABBIT_ALPHAS: dict[str, float] = {
    "020": 0.35, "075": 0.22, "030": 0.12,   # 正常组 L1→L3
    "088": 0.35, "065": 0.22, "016": 0.12,   # 感染组 L1→L3
}

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3" / "3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════

def _load_data():
    from .delta_r_r0_plot import (
        _extract_standard_motion_peaks,
        _extract_free_motion_points,
    )

    std_data, free_data = {}, {}
    all_rabbits = NORMAL_RABBITS + INFECTION_RABBITS
    for rabbit in all_rabbits:
        print(f"  {rabbit}...")
        std_data[rabbit] = _extract_standard_motion_peaks(rabbit)
        free_data[rabbit] = _extract_free_motion_points(rabbit, n_per_week=30)
    return std_data, free_data


def _build_df(data_dict, rabbits):
    """将 dict[rabbit] -> DataFrame 合并为统一 DataFrame。"""
    rows = []
    for rabbit in rabbits:
        df = data_dict.get(rabbit)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            week_str = str(row["week"])
            if week_str not in SELECTED_WEEKS:
                continue
            rows.append({
                "rabbit": rabbit,
                "group": RABBIT2GROUP.get(rabbit, ""),
                "week": week_str,
                "delta_r_r0": float(row["delta_r_r0"]),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════

def _mann_whitney_p(group_a, group_b):
    """Mann-Whitney U 检验，样本不足时降级为 t-test。"""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        _, p = stats.ttest_ind(a, b)
    return float(p)


def _format_p(p):
    """格式化 P 值。"""
    if np.isnan(p):
        return "N/A"
    if p < 0.0001:
        return "P < 0.0001"
    return f"P = {p:.4f}"


def _p_stars(p):
    """P 值 -> 星号。"""
    if np.isnan(p):
        return ""
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "NS"


# ══════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════

def _compute_bar_positions(rabbits_per_group, n_weeks, bar_width,
                           rabbit_gap, group_gap, intra_gap=0):
    """计算每根柱子的 X 轴位置。

    返回:
        bar_positions: dict[(group, rabbit, week)] -> x_center
        group_centers: dict[group] -> 大组中心 X
        rabbit_centers: dict[(group, rabbit)] -> 兔子中心 X
    """
    bar_positions = {}
    group_centers = {}
    rabbit_centers = {}

    x_cursor = 0.0
    for gi, (group_name, rabbits) in enumerate(rabbits_per_group.items()):
        group_start = x_cursor
        for ri, rabbit in enumerate(rabbits):
            rabbit_start = x_cursor
            for wi in range(n_weeks):
                x_center = x_cursor + bar_width / 2
                bar_positions[(group_name, rabbit, SELECTED_WEEKS[wi])] = x_center
                x_cursor += bar_width + intra_gap
            rabbit_centers[(group_name, rabbit)] = (rabbit_start + x_cursor - intra_gap) / 2
            x_cursor += rabbit_gap - intra_gap
        x_cursor += group_gap - intra_gap
        group_centers[group_name] = (group_start + x_cursor - group_gap) / 2

    return bar_positions, group_centers, rabbit_centers


def _plot_one(df_all, chart_type, chart_title_cn):
    _setup_chinese_font()

    if df_all.empty:
        print(f"  [SKIP] {chart_type}: 无数据")
        return

    n_weeks = len(SELECTED_WEEKS)
    bar_width = 0.275  # 原 0.55 减半以适应 6 周
    intra_gap = bar_width / 4          # 组内间隔 = 1/4 柱宽
    rabbit_gap = 5 * bar_width / 4     # 兔子间间隔 = 5/4 柱宽
    group_gap = 5 * bar_width / 4      # 组间间隔 = 5/4 柱宽

    rabbits_per_group = {
        "normal": NORMAL_RABBITS,
        "infection": INFECTION_RABBITS,
    }

    bar_pos, group_centers, rabbit_centers = _compute_bar_positions(
        rabbits_per_group, n_weeks, bar_width, rabbit_gap, group_gap, intra_gap
    )

    # ── 计算每根柱子的 mean / SEM ──
    bar_stats = {}
    for key, x in bar_pos.items():
        group, rabbit, week = key
        sub = df_all[
            (df_all["group"] == group)
            & (df_all["rabbit"] == rabbit)
            & (df_all["week"] == week)
        ]
        vals = sub["delta_r_r0"].dropna().to_numpy()
        if len(vals) >= 2:
            bar_stats[key] = {
                "mean": np.mean(vals),
                "sem": np.std(vals, ddof=1) / np.sqrt(len(vals)),
                "vals": vals,
                "n": len(vals),
            }
        elif len(vals) == 1:
            bar_stats[key] = {
                "mean": float(vals[0]),
                "sem": 0.0,
                "vals": vals,
                "n": 1,
            }

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制柱子
    for key, x in bar_pos.items():
        group, rabbit, week = key
        if key not in bar_stats:
            continue
        s = bar_stats[key]
        # 柱子：白色填充，黑色加粗描边
        ax.bar(
            x, s["mean"], bar_width,
            bottom=0,
            color="white",
            edgecolor="black",
            linewidth=1.6,
            zorder=2,
        )
        # 误差线
        if s["sem"] > 0:
            ax.errorbar(
                x, s["mean"], yerr=s["sem"],
                fmt="none", color="black",
                capsize=3, capthick=1.2,
                elinewidth=1.2,
                zorder=3,
            )

    # 叠加散点（按兔子着色，加 jitter）
    rng = np.random.default_rng(42)
    for _, row in df_all.iterrows():
        key = (row["group"], row["rabbit"], row["week"])
        if key not in bar_pos:
            continue
        x_center = bar_pos[key]
        jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35)
        ax.scatter(
            x_center + jitter, row["delta_r_r0"],
            c=[RABBIT_COLORS.get(row["rabbit"], "#888888")],
            s=22, alpha=RABBIT_ALPHAS.get(row["rabbit"], 0.7),
            edgecolors="black", linewidths=1.6,
            zorder=4,
        )

    # ─ X 轴标签 ──

    # Week 标签（每根柱子下方）
    for key, x in bar_pos.items():
        group, rabbit, week = key
        ax.text(x, -0.02, WEEK_LABELS.get(week, week),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6, color="black", clip_on=False)

    # 兔子级别标签（下移）
    rabbit_x_labels = []
    rabbit_x_ticks = []
    for group_name in ["normal", "infection"]:
        for rabbit in rabbits_per_group[group_name]:
            key_center = rabbit_centers.get((group_name, rabbit))
            if key_center is not None:
                rabbit_x_ticks.append(key_center)
                rabbit_x_labels.append(rabbit)

    ax.set_xticks(rabbit_x_ticks)
    ax.set_xticklabels([])  # 隐藏默认 tick，用手动 text 控制位置

    for xi, label in zip(rabbit_x_ticks, rabbit_x_labels):
        ax.text(xi, -0.07, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, clip_on=False)

    # 大组标签（放在更下方）
    label_y_map = {"standard": -0.13, "free": -0.13}
    label_y = label_y_map.get(chart_type, -0.13)

    for group_name in ["normal", "infection"]:
        group_xs = [
            bar_pos[(group_name, r, w)]
            for r in rabbits_per_group[group_name]
            for w in SELECTED_WEEKS
            if (group_name, r, w) in bar_pos
        ]
        if group_xs:
            x_l = min(group_xs) - bar_width * 0.7
            x_r = max(group_xs) + bar_width * 0.7
            ax.text(
                (x_l + x_r) / 2, label_y,
                GROUP_LABELS[group_name],
                ha="center", va="top", fontsize=13,
                fontweight="bold", color=GROUP_COLORS[group_name],
                clip_on=False,
            )

    # ── 图例（按兔子着色）──
    # 合并所有兔子列表，保持 Normal 在前、Infection 在后
    all_legend_rabbits = NORMAL_RABBITS + INFECTION_RABBITS
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=RABBIT_COLORS[r], markersize=8,
                   label=r)
        for r in all_legend_rabbits
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right",
              framealpha=0.8, title="Rabbit")

    # ── 轴 ─
    ax.set_ylabel("R/R0", fontsize=12)
    ax.set_title(chart_title_cn, fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="y", alpha=0.2, zorder=0)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.28)

    save_path = OUT_DIR / f"grouped_bar_normal_infection_{chart_type}.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def main():
    _setup_chinese_font()
    print("=" * 60)
    print("Normal + Infection 分组柱状图")
    print("=" * 60)

    print("\n加载数据...")
    std_data, free_data = _load_data()

    print("\n构建 DataFrame...")
    all_rabbits = NORMAL_RABBITS + INFECTION_RABBITS
    std_df = _build_df(std_data, all_rabbits)
    free_df = _build_df(free_data, all_rabbits)
    print(f"  标准运动: {len(std_df)} 条")
    print(f"  自由运动: {len(free_df)} 条")

    print("\n绘制标准运动柱状图...")
    _plot_one(std_df, "standard",
              "Normal vs Infection - Standard Motion (week1-6)")

    print("\n绘制自由运动柱状图...")
    _plot_one(free_df, "free",
              "Normal vs Infection - Free Motion (week1-6)")

    print("\n完成!")


if __name__ == "__main__":
    main()
