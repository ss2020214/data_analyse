# -*- coding: utf-8 -*-
"""
ΔR/R₀ 离群点分析与清理。

对自由运动选点应用 IQR（Tukey's fences）离群点检测：
- 按每周分组，计算 Q1、Q3、IQR
- 超出 [Q1 - 1.5×IQR, Q3 + 1.5×IQR] 的点标记为离群点
- 仅对自由运动采样点和自由运动生理学峰值点进行过滤
- 标准运动数据保持不变

输出：
  1. cleaned/ 目录：清洗后的 ΔR/R₀ 趋势图
  2. outlier_report.csv：每只兔子每周的离群点统计
  3. stats_comparison.csv：清洗前后的均值、标准差对比
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path

from .config import GROUPS, WEEKS, OUT_FIG, PARAMS, RABBIT2GROUP, ALL_RABBITS
from .delta_r_r0_plot import (
    _extract_standard_motion_peaks,
    _extract_free_motion_points,
    _extract_free_motion_peaks_in_window,
    WEEK_COLORS,
    OUT_DIR,
)


# ============================================================
# 输出目录
# ============================================================
CLEANED_OUT_DIR = OUT_DIR / "cleaned"
CLEANED_OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 离群点检测
# ============================================================
def detect_outliers_iqr(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """使用 IQR（Tukey's fences）方法检测离群点。

    参数：
        data: 一维数值数组
        multiplier: IQR 倍数，默认 1.5（标准 Tukey）

    返回：
        布尔数组，True 表示该点是离群点
    """
    if len(data) < 4:
        return np.zeros(len(data), dtype=bool)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return np.zeros(len(data), dtype=bool)
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (data < lower) | (data > upper)


def filter_outliers_by_week(df: pd.DataFrame, value_col: str = "delta_r_r0",
                             group_col: str = "week",
                             multiplier: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按周分组检测并移除离群点。

    返回：
        df_clean: 清洗后的 DataFrame
        df_outliers: 被移除的离群点 DataFrame
    """
    if df.empty:
        return df.copy(), pd.DataFrame(columns=df.columns)

    clean_parts = []
    outlier_parts = []

    for week, group in df.groupby(group_col):
        values = group[value_col].to_numpy()
        outlier_mask = detect_outliers_iqr(values, multiplier)

        clean_parts.append(group[~outlier_mask])
        if outlier_mask.any():
            outlier_part = group[outlier_mask].copy()
            outlier_part["outlier_reason"] = "IQR"
            outlier_parts.append(outlier_part)

    df_clean = (pd.concat(clean_parts, ignore_index=True)
                if clean_parts else pd.DataFrame(columns=df.columns))
    df_outliers = (pd.concat(outlier_parts, ignore_index=True)
                   if outlier_parts else pd.DataFrame(columns=list(df.columns) + ["outlier_reason"]))

    return df_clean, df_outliers


# ============================================================
# 2. 统计计算
# ============================================================
def compute_weekly_stats(df: pd.DataFrame, value_col: str = "delta_r_r0",
                          group_col: str = "week") -> pd.DataFrame:
    """计算每周的描述性统计量。"""
    records = []
    for week in WEEKS:
        sub = df[df[group_col] == week][value_col]
        if sub.size >= 1:
            records.append({
                "week": week,
                "n": len(sub),
                "mean": float(sub.mean()),
                "std": float(sub.std(ddof=1)) if sub.size >= 2 else 0.0,
                "median": float(sub.median()),
                "q1": float(sub.quantile(0.25)),
                "q3": float(sub.quantile(0.75)),
                "min": float(sub.min()),
                "max": float(sub.max()),
            })
    return pd.DataFrame(records)


# ============================================================
# 3. 清洗后绘图
# ============================================================
def plot_cleaned_delta_r_r0(rabbit: str,
                             df_std: pd.DataFrame,
                             df_free_clean: pd.DataFrame,
                             df_free_outliers: pd.DataFrame,
                             df_free_peaks_clean: pd.DataFrame,
                             df_free_peaks_outliers: pd.DataFrame):
    """使用清洗后数据绘制 ΔR/R₀ 趋势图。

    与原始图的区别：
    - 自由运动散点图中离群点以红色 × 标记
    - 自由运动统计图中叠加清洗前后的对比（灰色虚线 = 清洗前，橙色实线 = 清洗后）
    """
    # 合并自由运动全量数据（用于清洗前统计）
    df_free_all = pd.concat([df_free_clean, df_free_outliers.drop(columns=["outlier_reason"], errors="ignore")],
                            ignore_index=True) if not df_free_outliers.empty else df_free_clean
    df_free_peaks_all = pd.concat([df_free_peaks_clean, df_free_peaks_outliers.drop(columns=["outlier_reason"], errors="ignore")],
                                  ignore_index=True) if not df_free_peaks_outliers.empty else df_free_peaks_clean

    fig, axes = plt.subplots(3, 2, figsize=(12, 13),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle(f"Rabbit {rabbit} — ΔR/R₀ (Outliers Removed, IQR method)",
                 fontsize=14, fontweight='bold')

    # --- (a) 左上：标准运动散点图（不变）---
    ax = axes[0, 0]
    if not df_std.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7, label=w)
            x_offset += (t_rel.max() if t_rel.size else 5) + 5
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard movement", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- (b) 右上：标准运动统计图（不变）---
    ax = axes[0, 1]
    if not df_std.empty:
        means, stds, weeks_valid = [], [], []
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                means.append(sub.mean())
                stds.append(sub.std())
                weeks_valid.append(int(w[1:]))
        if means:
            ax.errorbar(weeks_valid, means, yerr=stds,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.2)

    # --- (c) 左中：自由运动散点图（清洗后 + 离群点标记）---
    ax = axes[1, 0]
    if not df_free_clean.empty or not df_free_outliers.empty:
        x_offset = 0
        week_positions = {}
        for w in WEEKS:
            sub_clean = df_free_clean[df_free_clean["week"] == w]
            sub_out = (df_free_outliers[df_free_outliers["week"] == w]
                       if not df_free_outliers.empty else pd.DataFrame())

            if sub_clean.empty and sub_out.empty:
                x_offset += 1
                continue

            # 用清洗后数据的时间范围决定 x 偏移
            if not sub_clean.empty:
                t_local = sub_clean["time_s"].to_numpy()
                t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
                ax.scatter(t_rel + x_offset, sub_clean["delta_r_r0"],
                           c=WEEK_COLORS.get(w, "gray"), s=15, alpha=0.6,
                           label=w)
            elif not sub_out.empty:
                t_local = sub_out["time_s"].to_numpy()
                t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
            else:
                t_rel = np.array([0])

            span = (t_rel.max() if t_rel.size else 5) + 10

            # 离群点标记为红色 ×
            if not sub_out.empty:
                # 将离群点的时间映射到同一 x 范围
                t_out = sub_out["time_s"].to_numpy()
                if t_rel.size > 0 and t_rel.max() > t_rel.min():
                    t_rel_out = (t_out - t_local.min()) if not sub_clean.empty else (t_out - t_out.min())
                else:
                    t_rel_out = np.zeros(len(t_out))
                ax.scatter(t_rel_out + x_offset, sub_out["delta_r_r0"],
                           c="red", marker="x", s=50, alpha=0.9, zorder=10,
                           linewidths=1.5)

            week_positions[w] = x_offset + span / 2
            x_offset += span

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement (red × = outliers removed)", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- (d) 右中：自由运动统计图（清洗前后对比）---
    ax = axes[1, 1]
    _plot_before_after_stats(ax, df_free_all, df_free_clean, WEEKS,
                             "Free (mean±std)\n— before  — after cleaned")

    # --- (e) 左下：自由运动窗口内峰值散点图（清洗后 + 离群点标记）---
    ax = axes[2, 0]
    if not df_free_peaks_clean.empty or not df_free_peaks_outliers.empty:
        x_offset = 0
        for w in WEEKS:
            sub_clean = df_free_peaks_clean[df_free_peaks_clean["week"] == w]
            sub_out = (df_free_peaks_outliers[df_free_peaks_outliers["week"] == w]
                       if not df_free_peaks_outliers.empty else pd.DataFrame())

            if sub_clean.empty and sub_out.empty:
                x_offset += 1
                continue

            if not sub_clean.empty:
                t_local = sub_clean["time_s"].to_numpy()
                t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
                ax.scatter(t_rel + x_offset, sub_clean["delta_r_r0"],
                           c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7,
                           label=w)
            elif not sub_out.empty:
                t_local = sub_out["time_s"].to_numpy()
                t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
            else:
                t_rel = np.array([0])

            span = (t_rel.max() if t_rel.size else 5) + 10

            if not sub_out.empty:
                t_out = sub_out["time_s"].to_numpy()
                if t_rel.size > 0 and t_rel.max() > t_rel.min():
                    t_rel_out = (t_out - t_local.min()) if not sub_clean.empty else (t_out - t_out.min())
                else:
                    t_rel_out = np.zeros(len(t_out))
                ax.scatter(t_rel_out + x_offset, sub_out["delta_r_r0"],
                           c="red", marker="x", s=50, alpha=0.9, zorder=10,
                           linewidths=1.5)

            x_offset += span

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement physio peaks (red × = outliers)", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- (f) 右下：自由运动窗口内峰值统计图（清洗前后对比）---
    ax = axes[2, 1]
    _plot_before_after_stats(ax, df_free_peaks_all, df_free_peaks_clean, WEEKS,
                             "Free physio-peaks\n— before  — after cleaned")

    fig.tight_layout()
    save_path = CLEANED_OUT_DIR / f"delta_r_r0_{rabbit}_cleaned.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_before_after_stats(ax, df_before: pd.DataFrame, df_after: pd.DataFrame,
                              weeks: list, title: str):
    """在给定的 ax 上绘制清洗前后均值±std 对比。"""
    means_before, stds_before, weeks_b = [], [], []
    means_after, stds_after, weeks_a = [], [], []

    for w in weeks:
        sub_b = df_before[df_before["week"] == w]["delta_r_r0"]
        if sub_b.size >= 2:
            means_before.append(sub_b.mean())
            stds_before.append(sub_b.std())
            weeks_b.append(int(w[1:]))

        sub_a = df_after[df_after["week"] == w]["delta_r_r0"]
        if sub_a.size >= 2:
            means_after.append(sub_a.mean())
            stds_after.append(sub_a.std())
            weeks_a.append(int(w[1:]))

    if weeks_b:
        ax.errorbar(np.array(weeks_b) - 0.12, means_before, yerr=stds_before,
                    fmt='o--', color="gray", capsize=3, lw=1.2, markersize=5,
                    alpha=0.6, label="before")
    if weeks_a:
        ax.errorbar(np.array(weeks_a) + 0.12, means_after, yerr=stds_after,
                    fmt='o-', color="#FF4500", capsize=4, lw=2, markersize=7,
                    label="cleaned")

    ax.set_xlabel("week")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)


# ============================================================
# 4. 主分析流程
# ============================================================
def analyze_rabbit(rabbit: str, multiplier: float = 1.5):
    """对单只兔子执行离群点分析。

    返回：
        report_rows: 每周的离群点报告行列表
        stats_comparison: 清洗前后统计对比列表
        has_outliers: 是否检测到离群点
    """
    print(f"\n{'='*60}")
    print(f"分析兔子 {rabbit} ({RABBIT2GROUP[rabbit]})")
    print(f"{'='*60}")

    # 提取数据
    print("  [1/4] 提取标准运动数据...")
    df_std = _extract_standard_motion_peaks(rabbit)
    print(f"        标准运动: {len(df_std)} 个峰值点")

    print("  [2/4] 提取自由运动数据...")
    df_free = _extract_free_motion_points(rabbit, n_per_week=30)
    print(f"        自由运动: {len(df_free)} 个采样点")

    print("  [3/4] 提取自由运动生理学峰值...")
    df_free_peaks = _extract_free_motion_peaks_in_window(rabbit)
    print(f"        自由运动生理学峰: {len(df_free_peaks)} 个峰值点")

    # 离群点检测
    print("  [4/4] 检测离群点 (IQR × {})...".format(multiplier))
    df_free_clean, df_free_outliers = filter_outliers_by_week(
        df_free, value_col="delta_r_r0", multiplier=multiplier)
    df_fp_clean, df_fp_outliers = filter_outliers_by_week(
        df_free_peaks, value_col="delta_r_r0", multiplier=multiplier)

    n_free_out = len(df_free_outliers)
    n_fp_out = len(df_fp_outliers)
    total_out = n_free_out + n_fp_out
    print(f"        自由运动离群点: {n_free_out} / {len(df_free)} "
          f"({100*n_free_out/max(len(df_free),1):.1f}%)")
    print(f"        生理学峰离群点: {n_fp_out} / {len(df_free_peaks)} "
          f"({100*n_fp_out/max(len(df_free_peaks),1):.1f}%)")

    has_outliers = total_out > 0

    # 生成清洗后图表
    if has_outliers:
        save_path = plot_cleaned_delta_r_r0(
            rabbit, df_std, df_free_clean, df_free_outliers,
            df_fp_clean, df_fp_outliers)
        print(f"        清洗后图表: {save_path}")
    else:
        print("        无离群点，跳过图表生成")

    # 生成报告
    report_rows = _build_report(rabbit, df_free, df_free_clean, df_free_outliers,
                                 df_free_peaks, df_fp_clean, df_fp_outliers)

    stats_comparison = _build_stats_comparison(
        rabbit, df_free, df_free_clean, "free_motion",
        df_free_peaks, df_fp_clean, "free_physio_peaks")

    return report_rows, stats_comparison, has_outliers


def _build_report(rabbit: str,
                   df_free_raw: pd.DataFrame,
                   df_free_clean: pd.DataFrame,
                   df_free_outliers: pd.DataFrame,
                   df_fp_raw: pd.DataFrame,
                   df_fp_clean: pd.DataFrame,
                   df_fp_outliers: pd.DataFrame) -> list[dict]:
    """构建每周离群点报告。"""
    rows = []
    for week in WEEKS:
        # 自由运动采样点
        sub_raw = df_free_raw[df_free_raw["week"] == week]
        sub_clean = df_free_clean[df_free_clean["week"] == week]
        sub_out = (df_free_outliers[df_free_outliers["week"] == week]
                   if not df_free_outliers.empty else pd.DataFrame())

        # 生理学峰值点
        sub_fp_raw = df_fp_raw[df_fp_raw["week"] == week]
        sub_fp_clean = df_fp_clean[df_fp_clean["week"] == week]
        sub_fp_out = (df_fp_outliers[df_fp_outliers["week"] == week]
                      if not df_fp_outliers.empty else pd.DataFrame())

        row = {
            "rabbit": rabbit,
            "group": RABBIT2GROUP[rabbit],
            "week": week,
            # 自由运动采样点
            "free_n_raw": len(sub_raw),
            "free_n_outliers": len(sub_out),
            "free_n_clean": len(sub_clean),
            "free_outlier_pct": round(100 * len(sub_out) / max(len(sub_raw), 1), 1),
            "free_mean_raw": round(float(sub_raw["delta_r_r0"].mean()), 6) if len(sub_raw) > 0 else None,
            "free_mean_clean": round(float(sub_clean["delta_r_r0"].mean()), 6) if len(sub_clean) > 0 else None,
            "free_std_raw": round(float(sub_raw["delta_r_r0"].std()), 6) if len(sub_raw) >= 2 else None,
            "free_std_clean": round(float(sub_clean["delta_r_r0"].std()), 6) if len(sub_clean) >= 2 else None,
            # 生理学峰值点
            "fp_n_raw": len(sub_fp_raw),
            "fp_n_outliers": len(sub_fp_out),
            "fp_n_clean": len(sub_fp_clean),
            "fp_outlier_pct": round(100 * len(sub_fp_out) / max(len(sub_fp_raw), 1), 1),
            "fp_mean_raw": round(float(sub_fp_raw["delta_r_r0"].mean()), 6) if len(sub_fp_raw) > 0 else None,
            "fp_mean_clean": round(float(sub_fp_clean["delta_r_r0"].mean()), 6) if len(sub_fp_clean) > 0 else None,
            "fp_std_raw": round(float(sub_fp_raw["delta_r_r0"].std()), 6) if len(sub_fp_raw) >= 2 else None,
            "fp_std_clean": round(float(sub_fp_clean["delta_r_r0"].std()), 6) if len(sub_fp_clean) >= 2 else None,
        }
        rows.append(row)
    return rows


def _build_stats_comparison(rabbit: str,
                             df_raw1, df_clean1, label1: str,
                             df_raw2, df_clean2, label2: str) -> list[dict]:
    """构建清洗前后统计对比（长格式，便于后续分析）。"""
    rows = []
    for week in WEEKS:
        for label, df_raw, df_clean in [(label1, df_raw1, df_clean1),
                                         (label2, df_raw2, df_clean2)]:
            sub_raw = df_raw[df_raw["week"] == week]
            sub_clean = df_clean[df_clean["week"] == week]

            row = {
                "rabbit": rabbit,
                "group": RABBIT2GROUP[rabbit],
                "week": week,
                "data_type": label,
                "n_raw": len(sub_raw),
                "n_clean": len(sub_clean),
                "n_removed": len(sub_raw) - len(sub_clean),
            }

            if len(sub_raw) > 0:
                row["mean_raw"] = round(float(sub_raw["delta_r_r0"].mean()), 8)
                row["median_raw"] = round(float(sub_raw["delta_r_r0"].median()), 8)
                row["std_raw"] = round(float(sub_raw["delta_r_r0"].std(ddof=1)), 8) if len(sub_raw) >= 2 else 0.0
                row["q1_raw"] = round(float(sub_raw["delta_r_r0"].quantile(0.25)), 8)
                row["q3_raw"] = round(float(sub_raw["delta_r_r0"].quantile(0.75)), 8)
                row["min_raw"] = round(float(sub_raw["delta_r_r0"].min()), 8)
                row["max_raw"] = round(float(sub_raw["delta_r_r0"].max()), 8)
            else:
                for k in ["mean_raw", "median_raw", "std_raw", "q1_raw", "q3_raw", "min_raw", "max_raw"]:
                    row[k] = None

            if len(sub_clean) > 0:
                row["mean_clean"] = round(float(sub_clean["delta_r_r0"].mean()), 8)
                row["median_clean"] = round(float(sub_clean["delta_r_r0"].median()), 8)
                row["std_clean"] = round(float(sub_clean["delta_r_r0"].std(ddof=1)), 8) if len(sub_clean) >= 2 else 0.0
                row["q1_clean"] = round(float(sub_clean["delta_r_r0"].quantile(0.25)), 8)
                row["q3_clean"] = round(float(sub_clean["delta_r_r0"].quantile(0.75)), 8)
                row["min_clean"] = round(float(sub_clean["delta_r_r0"].min()), 8)
                row["max_clean"] = round(float(sub_clean["delta_r_r0"].max()), 8)
            else:
                for k in ["mean_clean", "median_clean", "std_clean", "q1_clean", "q3_clean", "min_clean", "max_clean"]:
                    row[k] = None

            # 变化量
            if row["mean_raw"] is not None and row["mean_clean"] is not None:
                row["mean_change"] = round(row["mean_clean"] - row["mean_raw"], 8)
                row["mean_change_pct"] = round(
                    100 * (row["mean_clean"] - row["mean_raw"]) / max(abs(row["mean_raw"]), 1e-10), 2)
            else:
                row["mean_change"] = None
                row["mean_change_pct"] = None

            if (row["std_raw"] is not None and row["std_clean"] is not None
                    and row["std_raw"] > 0):
                row["std_change"] = round(row["std_clean"] - row["std_raw"], 8)
                row["std_change_pct"] = round(
                    100 * (row["std_clean"] - row["std_raw"]) / row["std_raw"], 2)
                # 标准差缩减比例（正数 = 标准差降低）
                row["std_reduction_pct"] = round(
                    100 * (row["std_raw"] - row["std_clean"]) / row["std_raw"], 2)
            else:
                row["std_change"] = None
                row["std_change_pct"] = None
                row["std_reduction_pct"] = None

            rows.append(row)
    return rows


# ============================================================
# 5. 主入口
# ============================================================
def main(multiplier: float = 1.5):
    """对所有兔子执行离群点分析。

    参数：
        multiplier: IQR 倍数，默认 1.5（标准 Tukey's fences）
    """
    print("=" * 60)
    print("ΔR/R₀ 离群点分析与清理")
    print(f"方法: IQR (Tukey's fences), multiplier = {multiplier}")
    print("=" * 60)

    all_reports = []
    all_stats = []
    affected_rabbits = []
    unaffected_rabbits = []

    for rabbit in ALL_RABBITS:
        try:
            report_rows, stats_comparison, has_outliers = analyze_rabbit(
                rabbit, multiplier=multiplier)
            all_reports.extend(report_rows)
            all_stats.extend(stats_comparison)

            if has_outliers:
                affected_rabbits.append(rabbit)
            else:
                unaffected_rabbits.append(rabbit)
        except Exception as e:
            print(f"    [ERROR] {rabbit}: {e}")
            import traceback
            traceback.print_exc()

    # 保存报告
    df_report = pd.DataFrame(all_reports)
    report_path = REPORT_DIR / "outlier_report.csv"
    df_report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n离群点报告已保存: {report_path}")

    df_stats = pd.DataFrame(all_stats)
    stats_path = REPORT_DIR / "stats_comparison.csv"
    df_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"统计对比已保存: {stats_path}")

    # 汇总
    print("\n" + "=" * 60)
    print("分析汇总")
    print("=" * 60)
    print(f"总兔子数: {len(ALL_RABBITS)}")
    print(f"受离群点影响: {len(affected_rabbits)} 只 — {', '.join(affected_rabbits)}")
    print(f"未受影响: {len(unaffected_rabbits)} 只 — {', '.join(unaffected_rabbits)}")

    # 按数据类型的汇总
    for dtype in ["free_motion", "free_physio_peaks"]:
        sub = df_stats[df_stats["data_type"] == dtype]
        total_out = int(sub["n_removed"].sum())
        total_raw = int(sub["n_raw"].sum())
        print(f"\n  [{dtype}]")
        print(f"    总离群点: {total_out} / {total_raw} ({100*total_out/max(total_raw,1):.1f}%)")

        # 标准差显著降低的周（降低 > 20%）
        sig_reduction = sub[
            (sub["std_reduction_pct"].notna()) & (sub["std_reduction_pct"] > 20)
        ]
        if len(sig_reduction) > 0:
            print(f"    标准差降低 >20% 的 session: {len(sig_reduction)}")
            for _, row in sig_reduction.iterrows():
                print(f"      {row['rabbit']} {row['week']}: "
                      f"std {row['std_raw']:.6f} → {row['std_clean']:.6f} "
                      f"(-{row['std_reduction_pct']:.1f}%)")

    # 打印所有受影响的 session
    print("\n受影响 session 详情:")
    affected_sessions = df_report[
        (df_report["free_n_outliers"] > 0) | (df_report["fp_n_outliers"] > 0)
    ]
    for _, row in affected_sessions.iterrows():
        parts = [f"  {row['rabbit']} {row['week']}:"]
        if row["free_n_outliers"] > 0:
            parts.append(
                f"自由运动 {int(row['free_n_outliers'])}/{int(row['free_n_raw'])} 离群点 "
                f"({row['free_outlier_pct']}%), "
                f"std: {row['free_std_raw']} → {row['free_std_clean']}"
            )
        if row["fp_n_outliers"] > 0:
            parts.append(
                f"生理学峰 {int(row['fp_n_outliers'])}/{int(row['fp_n_raw'])} 离群点 "
                f"({row['fp_outlier_pct']}%), "
                f"std: {row['fp_std_raw']} → {row['fp_std_clean']}"
            )
        print("; ".join(parts))

    print(f"\n清洗后图表保存至: {CLEANED_OUT_DIR}")
    print("完成!")


if __name__ == "__main__":
    main()
