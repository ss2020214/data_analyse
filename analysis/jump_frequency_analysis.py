# -*- coding: utf-8 -*-
"""
跳跃频率分析：直接从label文件计算每窗口的跳跃频率，
汇总每只兔子每周的统计特征，并绘制多面板对比图。

跳跃频率定义：每个窗口的 count / (end - start) * 60  → 次/分钟

输出：3张主图（最大值 / 平均值 / 中位数），每张包含6个子图(W1-W6)，
按4组（Normal / Pathological / Infection / Sham）分色展示。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 项目配置
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.config import (ALL_RABBITS, WEEKS, RABBIT2GROUP,
                              GROUPS, FOLDERS, OUT_FIG, OUT_TAB)
from analysis.preprocess import load_label

# ============================================================
# 组别配色与前缀
# ============================================================
GROUP_COLORS = {
    "normal":        "#2196F3",   # 蓝
    "pathological":  "#F44336",   # 红
    "infection":     "#FF9800",   # 橙
    "sham":          "#4CAF50",   # 绿
}
GROUP_PREFIX = {
    "normal":        "N",
    "pathological":  "P",
    "infection":     "I",
    "sham":          "S",
}
GROUP_ORDER = ["normal", "pathological", "infection", "sham"]
GROUP_LABELS_CN = {
    "normal":        "正常组(N)",
    "pathological":  "病理组(P)",
    "infection":     "感染组(I)",
    "sham":          "空白对照组(S)",
}


# ============================================================
# 1. 读取所有label数据，计算每窗口跳跃频率
# ============================================================
def collect_all_jump_frequencies():
    """遍历所有兔子和周，计算每窗口的跳跃频率（次/分钟）。"""
    records = []
    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        for week in WEEKS:
            label = load_label(rabbit, week)
            if label.empty:
                continue
            # 每窗口的跳跃频率 = count / duration_min
            durations = label["end"] - label["start"]  # 秒
            # 过滤掉持续时间为0或极短的窗口
            valid = durations > 0.5  # 至少0.5秒
            if valid.sum() == 0:
                continue
            freq = label.loc[valid, "count"] / (durations[valid] / 60.0)  # 次/分钟
            for i, (f, c, d) in enumerate(zip(freq, label.loc[valid, "count"], durations[valid])):
                records.append({
                    "rabbit": rabbit,
                    "group": group,
                    "week": week,
                    "window_idx": i,
                    "count": int(c),
                    "duration_s": float(d),
                    "freq_per_min": float(f),  # 次/分钟
                })
    return pd.DataFrame(records)


def compute_weekly_stats(df):
    """计算每只兔子每周的跳跃频率统计特征。"""
    stats = df.groupby(["rabbit", "group", "week"]).agg(
        freq_max=("freq_per_min", "max"),
        freq_median=("freq_per_min", "median"),
        freq_mean=("freq_per_min", "mean"),
        freq_std=("freq_per_min", "std"),
        freq_min=("freq_per_min", "min"),
        freq_q25=("freq_per_min", lambda x: x.quantile(0.25)),
        freq_q75=("freq_per_min", lambda x: x.quantile(0.75)),
        total_jumps=("count", "sum"),
        n_windows=("count", "count"),
        total_active_s=("duration_s", "sum"),
    ).reset_index()
    # 整体平均频率 = 总跳数 / 总活动时间（分钟）
    stats["freq_overall"] = stats["total_jumps"] / (stats["total_active_s"] / 60.0)
    return stats


# ============================================================
# 2. 可视化：3张主图（max / mean / median）
# ============================================================
FEATURES_TO_PLOT = [
    ("freq_max",    "窗口跳跃频率 — 最大值 (次/分钟)"),
    ("freq_mean",   "窗口跳跃频率 — 平均值 (次/分钟)"),
    ("freq_median", "窗口跳跃频率 — 中位数 (次/分钟)"),
]


def _rabbit_label(rabbit, group):
    """生成带组别前缀的兔子标签，如 N-030, P-043。"""
    return f"{GROUP_PREFIX[group]}-{rabbit}"


def plot_feature_6panels(stats, feature_col, title_cn, out_dir):
    """
    一张大图：2行×3列共6个子图，分别对应W1-W6。
    每个子图中按4组分色柱状图展示每只兔子的该特征值。
    统一纵坐标范围，便于跨周对比。
    """
    weeks_all = ["w1", "w2", "w3", "w4", "w5", "w6"]

    # 全局纵坐标范围
    all_vals = stats[feature_col].dropna()
    if all_vals.empty:
        return None
    y_max = all_vals.max() * 1.15

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle(title_cn, fontsize=16, fontweight='bold', y=0.98)

    for i, week in enumerate(weeks_all):
        ax = axes[i]
        week_data = stats[stats["week"] == week].copy()

        x_pos = 0
        tick_labels = []
        tick_positions = []
        bar_colors = []

        for group in GROUP_ORDER:
            gd = week_data[week_data["group"] == group].sort_values("rabbit")
            if gd.empty:
                continue
            for _, row in gd.iterrows():
                label_txt = _rabbit_label(row["rabbit"], group)
                ax.bar(x_pos, row[feature_col],
                       color=GROUP_COLORS[group], alpha=0.8, width=0.7,
                       edgecolor='black', linewidth=0.5)
                # 在柱顶标注数值
                val = row[feature_col]
                if not np.isnan(val):
                    ax.text(x_pos, val + y_max * 0.01, f"{val:.1f}",
                            ha='center', va='bottom', fontsize=6.5, rotation=0)
                tick_labels.append(label_txt)
                tick_positions.append(x_pos)
                x_pos += 1
            x_pos += 0.8  # 组间间距

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=55, ha='right', fontsize=8)
        ax.set_title(f"{week.upper()}", fontsize=13, fontweight='bold')
        ax.set_ylabel("次/分钟", fontsize=10)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 图例（底部居中）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GROUP_COLORS[g], alpha=0.8,
              label=GROUP_LABELS_CN[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.005),
               frameon=True, edgecolor='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = out_dir / f"jump_freq_{feature_col}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path.name}")
    return out_path


# ============================================================
# 3. 周间变化率图（每只兔子相对上一周的百分比变化）
# ============================================================
def compute_week_over_week_change(stats):
    """
    计算每只兔子每一周相对于上一周的变化率（百分比）。
    变化率 = (本周值 - 上周值) / 上周值 × 100%
    返回 DataFrame，包含 rabbit, group, week, 以及各指标的 _pct_change 列。
    """
    weeks_order = ["w1", "w2", "w3", "w4", "w5", "w6"]
    metrics = ["freq_max", "freq_mean", "freq_median"]
    records = []

    for rabbit in stats["rabbit"].unique():
        rd = stats[stats["rabbit"] == rabbit].copy()
        group = rd.iloc[0]["group"]
        rd = rd.set_index("week").reindex(weeks_order)

        for i in range(1, len(weeks_order)):
            cur_week = weeks_order[i]
            prev_week = weeks_order[i - 1]
            row = {"rabbit": rabbit, "group": group, "week": cur_week,
                   "prev_week": prev_week}
            for m in metrics:
                cur_val = rd.loc[cur_week, m] if cur_week in rd.index else np.nan
                prev_val = rd.loc[prev_week, m] if prev_week in rd.index else np.nan
                if pd.notna(cur_val) and pd.notna(prev_val) and prev_val != 0:
                    row[f"{m}_pct"] = (cur_val - prev_val) / prev_val * 100
                else:
                    row[f"{m}_pct"] = np.nan
            records.append(row)

    return pd.DataFrame(records)


def plot_week_change_6panels(change_df, feature_col, title_cn, out_dir):
    """
    绘制周间变化率图：5个子图（W1→W2, W2→W3, ..., W5→W6），
    柱状图展示每只兔子相对上一周的变化百分比，正值向上/负值向下。
    """
    change_col = f"{feature_col}_pct"
    weeks_change = ["w2", "w3", "w4", "w5", "w6"]
    week_labels = ["W1→W2", "W2→W3", "W3→W4", "W4→W5", "W5→W6"]

    # 全局纵坐标范围（对称）
    all_vals = change_df[change_col].dropna()
    if all_vals.empty:
        return None
    y_abs_max = max(abs(all_vals.min()), abs(all_vals.max())) * 1.15
    y_abs_max = max(y_abs_max, 20)  # 至少 ±20%

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle(f"{title_cn} — 周间变化率 (%)", fontsize=16, fontweight='bold', y=0.98)

    for i, (week, wlabel) in enumerate(zip(weeks_change, week_labels)):
        ax = axes[i]
        week_data = change_df[change_df["week"] == week].copy()

        x_pos = 0
        tick_labels = []
        tick_positions = []

        for group in GROUP_ORDER:
            gd = week_data[week_data["group"] == group].sort_values("rabbit")
            if gd.empty:
                continue
            for _, row in gd.iterrows():
                val = row[change_col]
                if np.isnan(val):
                    continue
                label_txt = _rabbit_label(row["rabbit"], group)
                color = GROUP_COLORS[group]
                # 正值向上，负值向下
                ax.bar(x_pos, val, color=color, alpha=0.8, width=0.7,
                       edgecolor='black', linewidth=0.5)
                # 标注数值
                offset = y_abs_max * 0.02 if val >= 0 else -y_abs_max * 0.02
                va = 'bottom' if val >= 0 else 'top'
                ax.text(x_pos, val + offset, f"{val:+.1f}%",
                        ha='center', va=va, fontsize=6.5)
                tick_labels.append(label_txt)
                tick_positions.append(x_pos)
                x_pos += 1
            x_pos += 0.8  # 组间间距

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=55, ha='right', fontsize=8)
        ax.set_title(wlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel("变化率 (%)", fontsize=10)
        ax.set_ylim(-y_abs_max, y_abs_max)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 隐藏第6个子图
    axes[5].set_visible(False)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GROUP_COLORS[g], alpha=0.8, label=GROUP_LABELS_CN[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.005),
               frameon=True, edgecolor='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = out_dir / f"jump_freq_{feature_col}_week_change.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path.name}")
    return out_path


# ============================================================
# 4. W1归一化个体轨迹 + 组均值折线图
# ============================================================
GROUP_MARKERS = {
    "normal":        "o",   # ●
    "pathological":  "s",   # ■
    "infection":     "D",   # ◆
    "sham":          "^",   # ▲
}


def plot_normalized_trajectory(stats, feature_col, title_cn, out_dir):
    """
    绘制相对于W1的归一化轨迹图：
    - 每只兔子的个体轨迹（细线 + 组别标记符）
    - 每组的平均值曲线（粗线）
    标准化方法：当前周数值 ÷ W1数值
    """
    weeks_order = ["w1", "w2", "w3", "w4", "w5", "w6"]
    week_nums = [1, 2, 3, 4, 5, 6]

    # 计算每只兔子的归一化值
    norm_records = []
    for rabbit in stats["rabbit"].unique():
        rd = stats[stats["rabbit"] == rabbit].copy()
        group = rd.iloc[0]["group"]
        rd_indexed = rd.set_index("week")

        # 获取W1基准值
        if "w1" not in rd_indexed.index:
            continue
        w1_val = rd_indexed.loc["w1", feature_col]
        if pd.isna(w1_val) or w1_val == 0:
            continue

        for week in weeks_order:
            if week in rd_indexed.index:
                val = rd_indexed.loc[week, feature_col]
                if pd.notna(val):
                    norm_records.append({
                        "rabbit": rabbit,
                        "group": group,
                        "week": week,
                        "week_num": int(week[1:]),
                        "norm_val": val / w1_val,
                    })

    norm_df = pd.DataFrame(norm_records)
    if norm_df.empty:
        return None

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"{title_cn}\n相对于W1的变化趋势（归一化 = 当前周 ÷ W1）",
                 fontsize=14, fontweight='bold')

    # 1) 绘制个体轨迹（细线）
    for rabbit in norm_df["rabbit"].unique():
        rd = norm_df[norm_df["rabbit"] == rabbit].sort_values("week_num")
        group = rd.iloc[0]["group"]
        ax.plot(rd["week_num"], rd["norm_val"],
                color=GROUP_COLORS[group], alpha=0.35, linewidth=1.0,
                marker=GROUP_MARKERS[group], markersize=5,
                markeredgecolor='none', zorder=2)
        # 在最后一个点标注兔子编号
        last = rd.iloc[-1]
        ax.annotate(f"{GROUP_PREFIX[group]}-{rabbit}",
                    (last["week_num"], last["norm_val"]),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 0), textcoords='offset points')

    # 2) 绘制组均值曲线（粗线）
    for group in GROUP_ORDER:
        gd = norm_df[norm_df["group"] == group]
        if gd.empty:
            continue
        group_mean = gd.groupby("week_num")["norm_val"].mean()
        ax.plot(group_mean.index, group_mean.values,
                color=GROUP_COLORS[group], linewidth=3.0, alpha=0.9,
                marker=GROUP_MARKERS[group], markersize=10,
                markeredgecolor='white', markeredgewidth=1.5,
                label=f"{GROUP_LABELS_CN[group]} 组均值", zorder=5)

    # 参考线：y=1
    ax.axhline(1.0, color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
               label='W1基准线 (=1.0)')

    ax.set_xlabel("周次", fontsize=12)
    ax.set_ylabel("相对值（当前周 / W1）", fontsize=12)
    ax.set_xticks(week_nums)
    ax.set_xticklabels([f"W{i}" for i in week_nums], fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out_path = out_dir / f"jump_freq_{feature_col}_vs_w1_normalized.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path.name}")
    return out_path


# ============================================================
# 5. W2归一化个体轨迹 + 组均值折线图
# ============================================================
def plot_normalized_trajectory_w2(stats, feature_col, title_cn, out_dir):
    """
    绘制相对于W2的归一化轨迹图：
    - 每只兔子的个体轨迹（细线 + 组别标记符）
    - 每组的平均值曲线（粗线）
    标准化方法：当前周数值 ÷ W2数值，不包含W1
    """
    weeks_order = ["w2", "w3", "w4", "w5", "w6"]
    week_nums = [2, 3, 4, 5, 6]

    # 计算每只兔子的归一化值
    norm_records = []
    for rabbit in stats["rabbit"].unique():
        rd = stats[stats["rabbit"] == rabbit].copy()
        group = rd.iloc[0]["group"]
        rd_indexed = rd.set_index("week")

        # 获取W2基准值
        if "w2" not in rd_indexed.index:
            continue
        w2_val = rd_indexed.loc["w2", feature_col]
        if pd.isna(w2_val) or w2_val == 0:
            continue

        for week in weeks_order:
            if week in rd_indexed.index:
                val = rd_indexed.loc[week, feature_col]
                if pd.notna(val):
                    norm_records.append({
                        "rabbit": rabbit,
                        "group": group,
                        "week": week,
                        "week_num": int(week[1:]),
                        "norm_val": val / w2_val,
                    })

    norm_df = pd.DataFrame(norm_records)
    if norm_df.empty:
        return None

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"{title_cn}\n相对于W2的变化趋势（归一化 = 当前周 ÷ W2）",
                 fontsize=14, fontweight='bold')

    # 1) 绘制个体轨迹（细线）
    for rabbit in norm_df["rabbit"].unique():
        rd = norm_df[norm_df["rabbit"] == rabbit].sort_values("week_num")
        group = rd.iloc[0]["group"]
        ax.plot(rd["week_num"], rd["norm_val"],
                color=GROUP_COLORS[group], alpha=0.35, linewidth=1.0,
                marker=GROUP_MARKERS[group], markersize=5,
                markeredgecolor='none', zorder=2)
        # 在最后一个点标注兔子编号
        last = rd.iloc[-1]
        ax.annotate(f"{GROUP_PREFIX[group]}-{rabbit}",
                    (last["week_num"], last["norm_val"]),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 0), textcoords='offset points')

    # 2) 绘制组均值曲线（粗线）
    for group in GROUP_ORDER:
        gd = norm_df[norm_df["group"] == group]
        if gd.empty:
            continue
        group_mean = gd.groupby("week_num")["norm_val"].mean()
        ax.plot(group_mean.index, group_mean.values,
                color=GROUP_COLORS[group], linewidth=3.0, alpha=0.9,
                marker=GROUP_MARKERS[group], markersize=10,
                markeredgecolor='white', markeredgewidth=1.5,
                label=f"{GROUP_LABELS_CN[group]} 组均值", zorder=5)

    # 参考线：y=1
    ax.axhline(1.0, color='gray', linewidth=1.0, linestyle='--', alpha=0.6,
               label='W2基准线 (=1.0)')

    ax.set_xlabel("周次", fontsize=12)
    ax.set_ylabel("相对值（当前周 / W2）", fontsize=12)
    ax.set_xticks(week_nums)
    ax.set_xticklabels([f"W{i}" for i in week_nums], fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out_path = out_dir / f"jump_freq_{feature_col}_vs_w2_normalized.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path.name}")
    return out_path


# ============================================================
# 6. 主入口
# ============================================================
def main():
    print("=" * 60)
    print("跳跃频率分析：从 label 文件直接计算")
    print("=" * 60)

    # 收集数据
    print("\n[1] 读取所有 label 文件...")
    df_raw = collect_all_jump_frequencies()
    print(f"    共读取 {len(df_raw)} 个窗口记录，"
          f"覆盖 {df_raw['rabbit'].nunique()} 只兔子")

    # 计算统计
    print("\n[2] 计算每只兔子每周的跳跃频率统计...")
    stats = compute_weekly_stats(df_raw)
    print(f"    生成 {len(stats)} 条统计记录")

    # 保存表格
    out_dir = OUT_FIG / "jump_frequency"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats.to_csv(OUT_TAB / "jump_frequency_stats.csv", index=False)
    print(f"\n    统计表已保存: jump_frequency_stats.csv")

    # 打印概览
    print("\n[3] 各组各周跳跃频率概览:")
    summary = stats.groupby(["group", "week"]).agg(
        mean_freq_overall=("freq_overall", "mean"),
        median_freq_overall=("freq_overall", "median"),
        n_rabbits=("rabbit", "count"),
    ).reset_index()
    print(summary.to_string(index=False))

    # 绘图：3张主图（最大值 / 平均值 / 中位数）
    print("\n[4] 绘制对比图（3张主图 × 6子图/周）...")
    for col, title_cn in FEATURES_TO_PLOT:
        plot_feature_6panels(stats, col, title_cn, out_dir)

    # 绘图：3张周间变化率图
    print("\n[5] 计算并绘制周间变化率...")
    change_df = compute_week_over_week_change(stats)
    change_df.to_csv(OUT_TAB / "jump_frequency_week_change.csv", index=False)
    print(f"    变化率表已保存: jump_frequency_week_change.csv")
    for col, title_cn in FEATURES_TO_PLOT:
        plot_week_change_6panels(change_df, col, title_cn, out_dir)

    # 绘图：3张W1归一化轨迹图
    print("\n[6] 绘制W1归一化个体轨迹 + 组均值图...")
    for col, title_cn in FEATURES_TO_PLOT:
        plot_normalized_trajectory(stats, col, title_cn, out_dir)

    # 绘图：3张W2归一化轨迹图
    print("\n[7] 绘制W2归一化个体轨迹 + 组均值图...")
    for col, title_cn in FEATURES_TO_PLOT:
        plot_normalized_trajectory_w2(stats, col, title_cn, out_dir)

    print("\n" + "=" * 60)
    print("分析完成！所有图表已保存到 output/figures/jump_frequency/")
    print("=" * 60)


if __name__ == "__main__":
    main()
