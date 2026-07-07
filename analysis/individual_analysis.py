# -*- coding: utf-8 -*-
"""
个体级深入分析（v5）：
  1. 跨周池化对比：早期(W1-W2) vs 晚期(W5-W6) 各组特征均值差与 Mann-Whitney p
  2. 时间演变模式分类：对每只兔子的每个特征沿 W1→W6 拟合，判定为
     {monotone-up, monotone-down, U-shape (先降后升), inverted-U (先升后降),
      flat (无显著趋势)}
  3. 组层面汇总：每个特征在每组里出现各模式的占比，找出
     "正常组与病理组形状最不一样的特征"
  4. 写 report_individual_analysis.md

输入：output/tables/features_per_session.csv
输出：
  output/tables/pooled_early_vs_late.csv
  output/tables/temporal_pattern_per_rabbit.csv
  output/tables/temporal_pattern_group_summary.csv
  output/report_individual_analysis.md
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.config import OUT_TAB, ROOT

# --- 用于分析的特征清单（涵盖原有 + v5 新增） ---------------------------
ANALYSIS_FEATURES = [
    # 活跃度 / 节律
    "amp_norm_mean", "amp_mean", "amp_cv", "ipi_median", "ipi_cv",
    "burst_density_per_min", "jumps_per_min", "jumps_per_active_min",
    "active_duration_s", "burst_count",
    # 基线
    "baseline_mean", "baseline_drift_pct", "baseline_slope_pct_min",
    # 峰形态
    "peak_fwhm_median", "peak_rise_time_median", "peak_fall_time_median",
    "peak_asymmetry_median",
    # 频域
    "psd_peak_freq_hz", "psd_jump_band_ratio", "psd_total_power",
    # 时序模式
    "burst_duration_median", "burst_duration_max",
    "silence_median_s", "silence_max_s", "silence_cv",
    # 基线稳定性
    "baseline_short_drift_p95", "baseline_zero_cross_per_min", "baseline_rms",
    # 峰间关系
    "amp_autocorr_lag1", "amp_trend_slope_per_min", "ipi_trend_slope_s_per_min",
]

EARLY = ["w1", "w2"]
LATE = ["w5", "w6"]


# ---- 1) 跨周池化对比 ----
def pooled_early_vs_late(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in ANALYSIS_FEATURES:
        if f not in df.columns:
            continue
        for g in ["normal", "pathological", "infection", "sham"]:
            sub = df[df["group"] == g]
            early = sub[sub["week"].isin(EARLY)][f].dropna().to_numpy()
            late = sub[sub["week"].isin(LATE)][f].dropna().to_numpy()
            try:
                pv = (mannwhitneyu(early, late, alternative="two-sided").pvalue
                      if len(early) >= 2 and len(late) >= 2 else np.nan)
            except Exception:
                pv = np.nan
            rows.append({
                "feature": f, "group": g,
                "n_early": len(early), "n_late": len(late),
                "mean_early": float(np.mean(early)) if len(early) else np.nan,
                "mean_late": float(np.mean(late)) if len(late) else np.nan,
                "delta_late_minus_early":
                    (float(np.mean(late)) - float(np.mean(early)))
                    if len(early) and len(late) else np.nan,
                "rel_change_pct":
                    ((np.mean(late) - np.mean(early)) /
                     max(abs(np.mean(early)), 1e-9) * 100.0)
                    if len(early) and len(late) else np.nan,
                "mw_p_within_group_E_vs_L": pv,
            })
        # 同时计算 normal vs pathological 在 late 时的差异
        sub_n_late = df[(df["group"] == "normal") & df["week"].isin(LATE)][f].dropna()
        sub_p_late = df[(df["group"] == "pathological") & df["week"].isin(LATE)][f].dropna()
        sub_n_early = df[(df["group"] == "normal") & df["week"].isin(EARLY)][f].dropna()
        sub_p_early = df[(df["group"] == "pathological") & df["week"].isin(EARLY)][f].dropna()
        try:
            pv_late = (mannwhitneyu(sub_n_late, sub_p_late, alternative="two-sided").pvalue
                       if len(sub_n_late) >= 2 and len(sub_p_late) >= 2 else np.nan)
        except Exception:
            pv_late = np.nan
        try:
            pv_early = (mannwhitneyu(sub_n_early, sub_p_early, alternative="two-sided").pvalue
                        if len(sub_n_early) >= 2 and len(sub_p_early) >= 2 else np.nan)
        except Exception:
            pv_early = np.nan
        rows.append({
            "feature": f, "group": "N_vs_P (between)",
            "n_early": min(len(sub_n_early), len(sub_p_early)),
            "n_late": min(len(sub_n_late), len(sub_p_late)),
            "mean_early": np.nan, "mean_late": np.nan,
            "delta_late_minus_early": np.nan,
            "rel_change_pct": np.nan,
            "mw_p_within_group_E_vs_L": np.nan,
            "mw_p_normal_vs_path_early": pv_early,
            "mw_p_normal_vs_path_late": pv_late,
        })
    return pd.DataFrame(rows)


# ---- 2) 个体时间演变模式分类 ----
def _classify_pattern(values: np.ndarray) -> dict:
    """对 [w1, w2, ..., w6] 6 点序列分类。返回 dict 含 trend / shape / slope。"""
    out = {"shape": "missing", "slope": np.nan, "n_obs": 0,
           "delta_max_min": np.nan, "argmax_week_idx": -1, "argmin_week_idx": -1}
    if values is None or len(values) == 0:
        return out
    v = np.asarray(values, dtype=float)
    valid = ~np.isnan(v)
    if valid.sum() < 3:
        out["n_obs"] = int(valid.sum())
        return out
    x = np.arange(len(v))[valid]
    y = v[valid]
    out["n_obs"] = int(valid.sum())
    out["argmax_week_idx"] = int(x[np.argmax(y)])
    out["argmin_week_idx"] = int(x[np.argmin(y)])
    out["delta_max_min"] = float(np.max(y) - np.min(y))
    # 一次拟合斜率（按归一化 x ∈ [0,1]）
    try:
        slope = float(np.polyfit((x - x[0]) / max(x[-1] - x[0], 1), y, 1)[0])
    except Exception:
        slope = np.nan
    out["slope"] = slope

    # 形状判定：用动态阈值 = 序列幅度的 15%
    span = np.max(y) - np.min(y)
    eps = max(span * 0.15, 1e-9)
    if span < 1e-9 or (np.std(y) / max(abs(np.mean(y)), 1e-9) < 0.05):
        out["shape"] = "flat"
        return out

    # 找极值是否在中间
    arg_max = x[np.argmax(y)]
    arg_min = x[np.argmin(y)]
    is_mid_max = arg_max not in (x[0], x[-1])
    is_mid_min = arg_min not in (x[0], x[-1])

    end_minus_start = y[-1] - y[0]
    # 倒 U：极大值在中间，且端点都低于极大值至少 eps
    if is_mid_max and (np.max(y) - y[0] > eps) and (np.max(y) - y[-1] > eps):
        out["shape"] = "inverted-U"
    elif is_mid_min and (y[0] - np.min(y) > eps) and (y[-1] - np.min(y) > eps):
        out["shape"] = "U-shape"
    elif end_minus_start > eps:
        out["shape"] = "monotone-up"
    elif end_minus_start < -eps:
        out["shape"] = "monotone-down"
    else:
        out["shape"] = "flat"
    return out


def temporal_pattern_per_rabbit(df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (rabbit, feature) 沿 W1..W6 分类形状。"""
    weeks = sorted(df["week"].unique(), key=lambda x: int(x[1:]))
    rows = []
    for r in sorted(df["rabbit"].astype(str).unique()):
        sub = df[df["rabbit"].astype(str) == r].set_index("week")
        g = sub["group"].iloc[0] if len(sub) else "?"
        for f in ANALYSIS_FEATURES:
            if f not in df.columns:
                continue
            v = sub[f].reindex(weeks).to_numpy()
            cls = _classify_pattern(v)
            rows.append({"rabbit": r, "group": g, "feature": f, **cls})
    return pd.DataFrame(rows)


def temporal_pattern_group_summary(per_rabbit: pd.DataFrame) -> pd.DataFrame:
    """把每只兔子的形状汇总到组：每组每特征的形状占比 + 平均斜率。"""
    rows = []
    for f, g_df in per_rabbit.groupby("feature"):
        for g, gg in g_df.groupby("group"):
            n = len(gg)
            shape_counts = gg["shape"].value_counts()
            shape_str = "; ".join(f"{k}:{v}" for k, v in shape_counts.items())
            rows.append({
                "feature": f, "group": g, "n_rabbits": n,
                "mean_slope": float(np.nanmean(gg["slope"])),
                "median_slope": float(np.nanmedian(gg["slope"])),
                "dominant_shape": shape_counts.idxmax() if len(shape_counts) else "missing",
                "shape_distribution": shape_str,
            })
    return pd.DataFrame(rows).sort_values(["feature", "group"])


# ---- 3) 写 markdown 报告 ----
def _fmt(x, k=3):
    if isinstance(x, float):
        if np.isnan(x):
            return "—"
        return f"{x:.{k}f}"
    return str(x)


def _top_separating_features(pooled: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """挑出 normal 与 pathological 在晚期(W5-W6)差异最显著的 feature。"""
    sub = pooled[pooled["group"] == "N_vs_P (between)"].copy()
    sub = sub.dropna(subset=["mw_p_normal_vs_path_late"])
    sub = sub.sort_values("mw_p_normal_vs_path_late").head(k)
    return sub


def _shape_disagreement(group_summary: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """挑出 normal 与 pathological 主导形状不同的 feature。"""
    pivot = group_summary.pivot(index="feature", columns="group",
                                values="dominant_shape")
    if "normal" not in pivot.columns or "pathological" not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["normal", "pathological"])
    diff = pivot[pivot["normal"] != pivot["pathological"]].copy()
    # 同时附上斜率差作为强度排序依据
    slope_pivot = group_summary.pivot(index="feature", columns="group",
                                      values="mean_slope")
    diff["slope_normal"] = slope_pivot["normal"]
    diff["slope_pathological"] = slope_pivot["pathological"]
    diff["abs_slope_gap"] = (diff["slope_normal"] - diff["slope_pathological"]).abs()
    return diff.sort_values("abs_slope_gap", ascending=False).head(k)


def write_report(pooled: pd.DataFrame, per_rabbit: pd.DataFrame,
                 group_summary: pd.DataFrame, df: pd.DataFrame,
                 path):
    lines = []
    lines.append("# 个体级深入分析报告（v5）\n")
    lines.append("> 数据：`features_per_session.csv`，58 sessions，"
                 "Normal n=6 / Pathological n=3 / Sham n=1。\n")
    lines.append("> 视角：先做跨周池化（W1-W2 早期 vs W5-W6 晚期），"
                 "再对每只兔子做 W1→W6 时间演变形状分类，最后汇总各组的优势形状。\n")

    # 1. 关键区分特征
    lines.append("\n## 1. 在晚期最能区分 Normal / Pathological 的特征 (Top-8)\n")
    lines.append("Mann-Whitney U（双侧）on W5-W6 池化样本，按 p 值升序：\n")
    top = _top_separating_features(pooled, k=8)
    if len(top):
        lines.append("| feature | n_late_each | p (N vs P, late) | p (N vs P, early) |")
        lines.append("|---|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(
                f"| `{r['feature']}` | {int(r['n_late'])} | "
                f"{_fmt(r['mw_p_normal_vs_path_late'])} | "
                f"{_fmt(r['mw_p_normal_vs_path_early'])} |"
            )
    lines.append("")

    # 2. 形状差异最大的特征
    lines.append("\n## 2. 时间演变形状最不一致的特征（normal 与 pathological 主导形状不同）\n")
    sd = _shape_disagreement(group_summary, k=10)
    if len(sd):
        lines.append("| feature | normal 主导 | pathological 主导 | "
                     "slope_normal | slope_path | |Δslope| |")
        lines.append("|---|:---:|:---:|---:|---:|---:|")
        for f, r in sd.iterrows():
            lines.append(
                f"| `{f}` | {r['normal']} | {r['pathological']} | "
                f"{_fmt(r['slope_normal'])} | {_fmt(r['slope_pathological'])} | "
                f"{_fmt(r['abs_slope_gap'])} |"
            )
    else:
        lines.append("（暂无两组主导形状差异显著的特征）\n")
    lines.append("")

    # 3. 各组在每个特征上的形状分布（关键 12 个）
    key12 = [
        "amp_norm_mean", "amp_cv", "burst_density_per_min", "ipi_cv",
        "peak_fwhm_median", "peak_asymmetry_median", "psd_peak_freq_hz",
        "psd_jump_band_ratio", "silence_median_s", "burst_duration_median",
        "amp_autocorr_lag1", "ipi_trend_slope_s_per_min",
    ]
    lines.append("\n## 3. 各组在关键特征上的形状分布与平均斜率\n")
    lines.append("| feature | group | n | dominant shape | shape distribution | mean slope |")
    lines.append("|---|---|---:|:---:|---|---:|")
    g_idx = group_summary.set_index(["feature", "group"])
    for f in key12:
        for g in ["normal", "pathological", "infection", "sham"]:
            if (f, g) not in g_idx.index:
                continue
            r = g_idx.loc[(f, g)]
            lines.append(
                f"| `{f}` | {g} | {int(r['n_rabbits'])} | "
                f"{r['dominant_shape']} | {r['shape_distribution']} | "
                f"{_fmt(r['mean_slope'])} |"
            )
    lines.append("")

    # 4. 早期 vs 晚期 的组内变化（取关键 12 个）
    lines.append("\n## 4. 关键特征早期 vs 晚期的组内对比（W1-W2 → W5-W6）\n")
    lines.append("| feature | group | mean(early) | mean(late) | Δ | %变化 | p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    pidx = pooled[pooled["group"].isin(["normal", "pathological", "infection", "sham"])]
    pidx = pidx.set_index(["feature", "group"])
    for f in key12:
        for g in ["normal", "pathological", "infection", "sham"]:
            if (f, g) not in pidx.index:
                continue
            r = pidx.loc[(f, g)]
            lines.append(
                f"| `{f}` | {g} | {_fmt(r['mean_early'])} | "
                f"{_fmt(r['mean_late'])} | "
                f"{_fmt(r['delta_late_minus_early'])} | "
                f"{_fmt(r['rel_change_pct'], 1)}% | "
                f"{_fmt(r['mw_p_within_group_E_vs_L'])} |"
            )
    lines.append("")

    # 5. 总体描述（自动总结）
    lines.append("\n## 5. 自动总结（基于以上数据）\n")
    # normal 中以 inverted-U 主导的特征数 vs pathological
    n_iu = group_summary[(group_summary["group"] == "normal") &
                         (group_summary["dominant_shape"] == "inverted-U")]["feature"].tolist()
    p_iu = group_summary[(group_summary["group"] == "pathological") &
                         (group_summary["dominant_shape"] == "inverted-U")]["feature"].tolist()
    n_u = group_summary[(group_summary["group"] == "normal") &
                        (group_summary["dominant_shape"] == "U-shape")]["feature"].tolist()
    p_u = group_summary[(group_summary["group"] == "pathological") &
                        (group_summary["dominant_shape"] == "U-shape")]["feature"].tolist()
    n_up = group_summary[(group_summary["group"] == "normal") &
                         (group_summary["dominant_shape"] == "monotone-up")]["feature"].tolist()
    p_up = group_summary[(group_summary["group"] == "pathological") &
                         (group_summary["dominant_shape"] == "monotone-up")]["feature"].tolist()
    n_dn = group_summary[(group_summary["group"] == "normal") &
                         (group_summary["dominant_shape"] == "monotone-down")]["feature"].tolist()
    p_dn = group_summary[(group_summary["group"] == "pathological") &
                         (group_summary["dominant_shape"] == "monotone-down")]["feature"].tolist()

    def _fmt_list(xs, k=8):
        return "、".join(f"`{x}`" for x in xs[:k]) + (f" 等 {len(xs)} 项" if len(xs) > k else "")

    lines.append(f'- **normal 组以“倒U（先升后降）”为主导形状的特征**：'
                 f'{_fmt_list(n_iu) if n_iu else "无"}\n')
    lines.append(f'- **pathological 组以“倒U”为主导形状的特征**：'
                 f'{_fmt_list(p_iu) if p_iu else "无"}\n')
    lines.append(f'- **normal 组以“U 型（先降后升）”为主导**：'
                 f'{_fmt_list(n_u) if n_u else "无"}\n')
    lines.append(f'- **pathological 组以“U 型”为主导**：'
                 f'{_fmt_list(p_u) if p_u else "无"}\n')
    lines.append(f'- **normal 组单调上升的特征**：{_fmt_list(n_up) if n_up else "无"}\n')
    lines.append(f'- **pathological 组单调上升的特征**：{_fmt_list(p_up) if p_up else "无"}\n')
    lines.append(f'- **normal 组单调下降的特征**：{_fmt_list(n_dn) if n_dn else "无"}\n')
    lines.append(f'- **pathological 组单调下降的特征**：{_fmt_list(p_dn) if p_dn else "无"}\n')

    lines.append("\n## 6. 输出物索引\n")
    lines.append("- 详细个体特征表：`output/tables/features_per_session.csv`")
    lines.append("- 早晚池化对比：`output/tables/pooled_early_vs_late.csv`")
    lines.append("- 个体形状分类：`output/tables/temporal_pattern_per_rabbit.csv`")
    lines.append("- 组层形状汇总：`output/tables/temporal_pattern_group_summary.csv`")
    lines.append("- 个体轨迹图：`output/figures/trajectory_<feature>.png`（去组均值）")
    lines.append("- rabbit×week 热图：`output/figures/heatmap_<feature>.png`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def main():
    df = pd.read_csv(OUT_TAB / "features_per_session.csv")
    pooled = pooled_early_vs_late(df)
    pooled.to_csv(OUT_TAB / "pooled_early_vs_late.csv", index=False)
    print(f"Saved {OUT_TAB / 'pooled_early_vs_late.csv'}  rows={len(pooled)}")

    per_rabbit = temporal_pattern_per_rabbit(df)
    per_rabbit.to_csv(OUT_TAB / "temporal_pattern_per_rabbit.csv", index=False)
    print(f"Saved {OUT_TAB / 'temporal_pattern_per_rabbit.csv'}  rows={len(per_rabbit)}")

    group_summary = temporal_pattern_group_summary(per_rabbit)
    group_summary.to_csv(OUT_TAB / "temporal_pattern_group_summary.csv", index=False)
    print(f"Saved {OUT_TAB / 'temporal_pattern_group_summary.csv'}  rows={len(group_summary)}")

    out_report = ROOT / "output" / "report_individual_analysis.md"
    write_report(pooled, per_rabbit, group_summary, df, out_report)


if __name__ == "__main__":
    main()
