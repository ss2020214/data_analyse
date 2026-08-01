# -*- coding: utf-8 -*-
"""
068 / 094 / 030 三只兔子分组柱状图（标准运动 & 自由运动）。

X轴：按兔子区分（068、094、030）
每只兔子内：week1 / week3 / week6 三根小柱
柱上叠加每个样本的散点（按周数着色），带 mean+/-SEM 误差线。

另外生成3张单兔标准运动图（094和030去除Week3）。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
from .config import WEEKS, OUT_FIG

# ── 兔子顺序 ─────────────────────────────────────────────
RABBITS = ["068", "094", "030"]
SELECTED_WEEKS = ["w1", "w3", "w6"]
WEEK_LABELS = {"w1": "Week1", "w3": "Week3", "w6": "Week6"}

# ── 周数配色 ─────────────────────────────────────────────
WEEK_COLORS = {
    "w1": "#FF8C00",  # 橙
    "w3": "#228B22",  # 绿
    "w6": "#9932CC",  # 紫
}

# ── 兔子配色（与 3d_068_094_030 统一）──
RABBIT_COLORS = {
    "030": "#1f77b4",  # 蓝 (normal)
    "094": "#8c564b",  # 棕 (pathological)
    "068": "#cd853f",  # 金 (sham)
}

OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3" / "3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════

def _load_data():
    """加载数据。

    自由运动数据改用 optimized_v3 的双维度趋势引导选点结果，使 068/094 与
    delta_r_r0_*_optimized.png 的自由运动选点逻辑保持一致：
      - 参照兔子（030）直接放行原始采样点（= _extract_free_motion_points）
      - 非参照兔子（068/094）使用分位数窗口 + 均值/std 双目标引导选点
    标准运动仍用 _extract_standard_motion_peaks（与 optimized 系列一致）。
    """
    from .delta_r_r0_dual_guided import normalize_all_rabbits_v3
    from .delta_r_r0_plot import _extract_standard_motion_peaks

    print("  运行双维度趋势引导选点 (v3) 以获取 optimized 自由运动数据...")
    results = normalize_all_rabbits_v3(verbose=False)
    all_normalized = results["normalized"]  # {rabbit: {week: [{time_s, delta_r_r0}, ...]}}

    std_data, free_data = {}, {}
    for rabbit in RABBITS:
        print(f"  {rabbit}...")
        std_data[rabbit] = _extract_standard_motion_peaks(rabbit)
        # 取 optimized v3 归一化后的自由运动点（068/094 已做双维度引导）
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


def _build_df(data_dict, rabbits, weeks=None):
    if weeks is None:
        weeks = SELECTED_WEEKS
    rows = []
    for rabbit in rabbits:
        df = data_dict.get(rabbit)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            week_str = str(row["week"])
            if week_str not in weeks:
                continue
            rows.append({
                "rabbit": rabbit,
                "week": week_str,
                "delta_r_r0": float(row["delta_r_r0"]),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════

def _compute_bar_positions(rabbits, weeks, bar_width, rabbit_gap, intra_gap=0.0):
    bar_positions = {}
    rabbit_centers = {}

    x_cursor = 0.0
    for rabbit in rabbits:
        rabbit_start = x_cursor
        for wi, w in enumerate(weeks):
            x_center = x_cursor + bar_width / 2
            bar_positions[(rabbit, w)] = x_center
            x_cursor += bar_width + intra_gap
        x_cursor += rabbit_gap
        rabbit_centers[rabbit] = (rabbit_start + x_cursor - rabbit_gap) / 2

    return bar_positions, rabbit_centers


def _plot_one(df_all, chart_type, chart_title_cn, rabbits=None, weeks=None):
    _setup_chinese_font()

    if df_all.empty:
        print(f"  [SKIP] {chart_type}: 无数据")
        return

    if rabbits is None:
        rabbits = RABBITS
    if weeks is None:
        weeks = SELECTED_WEEKS

    n_weeks = len(weeks)
    bar_width = 0.275
    intra_gap = bar_width / 4         # 组内间隔 = 1/4 柱宽
    rabbit_gap = 5 * bar_width / 4    # 组间间隔 = 5/4 柱宽

    bar_pos, rabbit_centers = _compute_bar_positions(
        rabbits, weeks, bar_width, rabbit_gap, intra_gap
    )

    # ── 计算每根柱子的 mean / SEM ──
    bar_stats = {}
    for key, x in bar_pos.items():
        rabbit, week = key
        sub = df_all[
            (df_all["rabbit"] == rabbit)
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
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制柱子
    for key, x in bar_pos.items():
        rabbit, week = key
        if key not in bar_stats:
            continue
        s = bar_stats[key]
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
        key = (row["rabbit"], row["week"])
        if key not in bar_pos:
            continue
        x_center = bar_pos[key]
        jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35)
        ax.scatter(
            x_center + jitter, row["delta_r_r0"],
            c=[RABBIT_COLORS.get(row["rabbit"], "#888888")],
            s=22, alpha=0.7, edgecolors="black", linewidths=1.6,
            zorder=4,
        )

    # ─ X 轴双层标签 ──
    # 顶层：Week 标签（每根柱子正下方）
    week_ticks = [bar_pos[(r, w)] for r in rabbits for w in weeks if (r, w) in bar_pos]
    week_labels = [WEEK_LABELS[w] for r in rabbits for w in weeks if (r, w) in bar_pos]
    ax.set_xticks(week_ticks)
    ax.set_xticklabels(week_labels, fontsize=8, rotation=0)

    # 底层：兔子 ID 标签（用 text 放在更下方）
    rabbit_label_y = -0.08  # axes 坐标，负值置于轴下方
    for r in rabbits:
        if r in rabbit_centers:
            ax.text(
                rabbit_centers[r], rabbit_label_y, r,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11,
                fontweight="bold", color=RABBIT_COLORS.get(r, "#888888"),
                clip_on=False,
            )

    # ── 图例（按兔子着色）──
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=RABBIT_COLORS[r], markersize=8,
                   label=r)
        for r in rabbits
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right",
              framealpha=0.8, title="Rabbit")

    # ── 轴 ──
    ax.set_ylabel("R/R0", fontsize=12)
    ax.set_title(chart_title_cn, fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="y", alpha=0.2, zorder=0)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.22)

    save_path = OUT_DIR / f"grouped_bar_068_094_030_{chart_type}.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")


def _plot_single_rabbit(df_all, rabbit, chart_type, weeks, bar_spacing=None, ylim_top=None):
    """绘制单只兔子的柱状图。"""
    _setup_chinese_font()

    df_rabbit = df_all[df_all["rabbit"] == rabbit]
    if df_rabbit.empty:
        print(f"  [SKIP] {rabbit} {chart_type}: 无数据")
        return

    n_weeks = len(weeks)
    # 柱子宽度和间距（1:1:1:1 比例：柱宽=间隙=右留白）
    bar_width = 0.0056  # 柱子宽度
    if bar_spacing is None:
        bar_spacing = bar_width + bar_width / 4  # 组内间隔 = 1/4 柱宽

    # 柱子位置：Week1 距 Y 轴 0.5*bar_width，每个柱子间隔 bar_spacing
    bar_pos = {}
    for i, w in enumerate(weeks):
        bar_pos[(rabbit, w)] = bar_width + float(i) * bar_spacing

    # ── 计算每根柱子的 mean / SEM ──
    bar_stats = {}
    for key, x in bar_pos.items():
        _, week = key
        sub = df_rabbit[df_rabbit["week"] == week]
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
    fig, ax = plt.subplots(figsize=(3, 5))

    # 绘制柱子（统一兔子颜色，黑色描边，不区分周）
    rabbit_color = RABBIT_COLORS.get(rabbit, "#888888")
    for key, x in bar_pos.items():
        _, week = key
        if key not in bar_stats:
            continue
        s = bar_stats[key]
        ax.bar(
            x, s["mean"], bar_width,
            bottom=0,
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
            zorder=2,
        )
        if s["sem"] > 0:
            ax.errorbar(
                x, s["mean"], yerr=s["sem"],
                fmt="none", color="black",
                capsize=3, capthick=1.2,
                elinewidth=1.2,
                zorder=3,
            )

    # 叠加散点
    rng = np.random.default_rng(42)
    for _, row in df_rabbit.iterrows():
        key = (rabbit, row["week"])
        if key not in bar_pos:
            continue
        x_center = bar_pos[key]
        jitter = rng.uniform(-bar_width * 0.4, bar_width * 0.4)
        ax.scatter(
            x_center + jitter, row["delta_r_r0"],
            c=[rabbit_color],
            s=22, alpha=0.7, edgecolors="black", linewidths=1.6,
            zorder=4,
        )

    # X 轴标签（使用柱子的实际中心位置）
    week_ticks = [bar_pos[(rabbit, w)] for w in weeks]
    ax.set_xticks(week_ticks)
    ax.set_xticklabels([WEEK_LABELS[w] for w in weeks], fontsize=11)

    # X 轴范围（1:1:1:1：柱宽=间隙=右留白）
    half_bar = bar_width / 2
    last_bar_right = week_ticks[-1] + half_bar
    ax.set_xlim(0, last_bar_right + bar_width)

    # 图例（不区分周，移除周图例）
    ax.legend().remove() if ax.get_legend() else None

    # 轴
    ax.set_ylabel("R/R0", fontsize=12)
    ax.set_title(f"Rabbit {rabbit} - Standard Motion", fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="y", alpha=0.2, zorder=0)

    fig.subplots_adjust(left=0.15, right=0.92, top=0.88, bottom=0.12)

    save_path = OUT_DIR / f"grouped_bar_rabbit{rabbit}_standard.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def main():
    _setup_chinese_font()
    print("=" * 60)
    print("068 / 094 / 030 分组柱状图")
    print("=" * 60)

    print("\n加载数据...")
    std_data, free_data = _load_data()

    print("\n构建 DataFrame...")
    std_df = _build_df(std_data, RABBITS)
    free_df = _build_df(free_data, RABBITS)
    print(f"  标准运动: {len(std_df)} 条")
    print(f"  自由运动: {len(free_df)} 条")

    print("\n绘制标准运动柱状图（三只兔子合并）...")
    _plot_one(std_df, "standard",
              "068 / 094 / 030 - Standard Motion (week1/3/6)")

    print("\n绘制自由运动柱状图（三只兔子合并）...")
    _plot_one(free_df, "free",
              "068 / 094 / 030 - Free Motion (week1/3/6)")

    # ── 单兔标准运动图 ─
    print("\n绘制单兔标准运动图...")
    # 计算统一 y 轴上限
    single_configs = [
        ("068", ["w1", "w3"]),
        ("094", ["w1", "w6"]),
        ("030", ["w1", "w6"]),
    ]
    all_vals = []
    for rabbit, weeks in single_configs:
        for w in weeks:
            sub = std_df[(std_df["rabbit"] == rabbit) & (std_df["week"] == w)]
            vals = sub["delta_r_r0"].dropna().to_numpy()
            if vals.size > 0:
                all_vals.append(float(np.max(vals)))
    unified_ylim = max(all_vals) * 1.15 if all_vals else None
    for rabbit, weeks in single_configs:
        _plot_single_rabbit(std_df, rabbit, "standard", weeks, ylim_top=unified_ylim)

    print("\n完成!")


if __name__ == "__main__":
    main()
