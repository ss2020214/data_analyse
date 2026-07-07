# -*- coding: utf-8 -*-
"""
主流水线：
  for rabbit:
    for week:
      load → preprocess → detect peaks → estimate lag → features → plot

输出：
  output/intermediate/peaks_<rabbit>_<week>.png        中间结果图（原始/基线/检出峰/标注窗）
  output/intermediate/align_<rabbit>_<week>.png        lag 搜索曲线
  output/tables/features_per_session.csv               每只每周特征表
  output/tables/group_stats_per_week.csv               组间逐周描述统计 + Mann-Whitney p
  output/figures/feature_<name>.png                    组间逐周对比图
  output/figures/heatmap_<name>.png                    rabbit × week 热图
  output/report.md                                     处理流程说明书
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# 允许以脚本方式直接运行：python -m analysis.main
from analysis.config import (ALL_RABBITS, WEEKS, RABBIT2GROUP,
                             OUT_INTER, OUT_FIG, OUT_TAB, PARAMS)
from analysis.preprocess import load_session, load_label, preprocess, preprocess_raw
from analysis.peaks import (detect_peaks_stage1, detect_peaks_stage1_adaptive,
                            detect_peaks_stage2, detect_peaks_stage2_adaptive,
                            estimate_lag)
from analysis.features import session_features, session_peak_table, session_peak_table_stage1
from analysis.standard_motion import get_reference_profile
from analysis.plot import (plot_session, plot_alignment_curve,
                           plot_dual_stage,
                           feature_trajectory, feature_trajectory_normalized,
                           feature_trajectory_zscore_split,
                           heatmap_feature, boxplot_groups_at_week,
                           per_rabbit_box_panels)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_session(rabbit: str, week: str, plot: bool = True):
    """
    双阶段处理流水线（对应 background.md 第三节第5条）：
    Stage 1: preprocess_raw(无清洗) → detect_peaks_stage1 → estimate_lag (时域特征)
    Stage 2: preprocess(完整清洗) → detect_peaks_stage2(ref) (幅值特征)
    """
    try:
        t, r = load_session(rabbit, week)
    except KeyError:
        return None
    if t.size < 50:
        return None
    label = load_label(rabbit, week)

    # 获取标准运动参考特征
    ref_profile = None
    try:
        ref_profile = get_reference_profile(rabbit, week)
    except Exception:
        pass

    # === Stage 1: 不做任何数据清洗先寻峰 ===
    # background.md: "第一次不做任何的数据清洗先寻峰"
    pp_s1 = preprocess_raw(t, r)
    pk_s1 = detect_peaks_stage1(pp_s1)

    # 用 Stage 1 的峰进行时间对齐（峰更多，对齐更稳定）
    al = estimate_lag(pp_s1, pk_s1, label)

    # 自适应二次寻峰：检查检出率，不足则降低阈值重新检测
    # 确保 IPI 和跳跃频率满足关系：60/ipi_median ≈ jump_frequency
    pk_s1 = detect_peaks_stage1_adaptive(pp_s1, label, al["lag"])

    # === Stage 2: 完整清洗 + 精确寻峰（幅值特征） ===
    pp = preprocess(t, r, ref_profile=ref_profile)
    pk = detect_peaks_stage2_adaptive(pp, label, al["lag"], ref_profile=ref_profile)

    if plot:
        plot_session(rabbit, week, pp, pk, label, al,
                     OUT_INTER / f"peaks_{rabbit}_{week}.png")
        plot_alignment_curve(rabbit, week, al,
                             OUT_INTER / f"align_{rabbit}_{week}.png")
        # 双阶段对比图
        plot_dual_stage(rabbit, week, pp_s1, pk_s1, pp, pk, label, al,
                        OUT_INTER / f"dual_stage_{rabbit}_{week}.png",
                        ref_profile=ref_profile)

    # 提取特征：双阶段峰分别用于时域/幅值
    feats = session_features(pp, pk, label, al,
                             peaks_stage1=pk_s1, pp_stage1=pp_s1)
    feats.update({
        "rabbit": rabbit,
        "week": week,
        "group": RABBIT2GROUP[rabbit],
        # 双阶段峰数统计
        "n_peaks_stage1": int(pk_s1["idx"].size),
        "n_peaks_stage2": int(pk["idx"].size),
    })

    # 记录标准运动参考信息（便于溯源）
    if ref_profile is not None:
        feats["ref_peak_amp"] = ref_profile.get("peak_amp_median", np.nan)
        feats["ref_snr"] = ref_profile.get("snr", np.nan)
        feats["ref_peak_fwhm"] = ref_profile.get("peak_fwhm_median", np.nan)
    else:
        feats["ref_peak_amp"] = np.nan
        feats["ref_snr"] = np.nan
        feats["ref_peak_fwhm"] = np.nan

    # per-peak 原始表（供后续箱线图使用）
    # 幅值特征用 Stage 2 峰，频率特征用 Stage 1 峰
    peak_df = session_peak_table(pp, pk, label, al,
                                 peaks_stage1=pk_s1, pp_stage1=pp_s1)
    if not peak_df.empty:
        peak_df = peak_df.assign(rabbit=rabbit, week=week,
                                 group=RABBIT2GROUP[rabbit])

    # Stage 1 per-peak 表（频率/节律箱线图专用，基于全部 Stage 1 窗口内峰）
    peak_s1_df = session_peak_table_stage1(pp_s1, pk_s1, label, al)
    if not peak_s1_df.empty:
        peak_s1_df = peak_s1_df.assign(rabbit=rabbit, week=week,
                                       group=RABBIT2GROUP[rabbit])

    return feats, peak_df, peak_s1_df


def run_all(plot: bool = True):
    feat_rows = []
    peak_dfs = []
    peak_s1_dfs = []
    for rabbit in ALL_RABBITS:
        for week in WEEKS:
            r = run_session(rabbit, week, plot=plot)
            if r is None:
                continue
            f, pdf, ps1df = r
            feat_rows.append(f)
            if pdf is not None and not pdf.empty:
                peak_dfs.append(pdf)
            if ps1df is not None and not ps1df.empty:
                peak_s1_dfs.append(ps1df)
            print(f"[done] {rabbit} {week}  "
                  f"lag={f['lag_estimated']:+.2f}s  "
                  f"hit={f['hit_rate']:.2f}  "
                  f"s1={f['n_peaks_stage1']}/"
                  f"s2={f['peaks_in_label']}/"
                  f"{f['jumps_labeled']}  "
                  f"ref={'Y' if not np.isnan(f.get('ref_snr', np.nan)) else 'N'}")
    df = pd.DataFrame(feat_rows)
    cols = ["rabbit", "group", "week"] + [c for c in df.columns
                                          if c not in ("rabbit", "group", "week")]
    df = df[cols]
    df.to_csv(OUT_TAB / "features_per_session.csv", index=False)

    peaks_df = (pd.concat(peak_dfs, ignore_index=True)
                if peak_dfs else pd.DataFrame())
    if not peaks_df.empty:
        peaks_df["rabbit"] = peaks_df["rabbit"].astype(str)
        peaks_df.to_csv(OUT_TAB / "peaks_per_session.csv", index=False)

    # Stage 1 per-peak 表（频率/节律箱线图专用）
    peaks_s1_df = (pd.concat(peak_s1_dfs, ignore_index=True)
                   if peak_s1_dfs else pd.DataFrame())
    if not peaks_s1_df.empty:
        peaks_s1_df["rabbit"] = peaks_s1_df["rabbit"].astype(str)
        peaks_s1_df.to_csv(OUT_TAB / "peaks_stage1_per_session.csv", index=False)

    return df, peaks_df, peaks_s1_df


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """每个特征 × 每周：4组之间的 mean/std + Mann-Whitney U p 值。
    重点对比：normal vs pathological, normal vs infection。"""
    feat_cols = [c for c in df.columns
                 if c not in ("rabbit", "group", "week")]
    rows = []
    groups_present = sorted(df["group"].unique())
    for w in sorted(df["week"].unique(), key=lambda x: int(x[1:])):
        for f in feat_cols:
            row = {"week": w, "feature": f}
            for g in groups_present:
                sub = df[(df["week"] == w) & (df["group"] == g)][f].dropna().to_numpy()
                row[f"n_{g}"] = len(sub)
                row[f"mean_{g}"] = float(np.mean(sub)) if len(sub) else np.nan
                row[f"std_{g}"] = float(np.std(sub, ddof=1)) if len(sub) >= 2 else np.nan
            # 两两比较（重点：normal vs pathological, normal vs infection）
            for g2 in ["pathological", "infection"]:
                xn = df[(df["week"] == w) & (df["group"] == "normal")][f].dropna().to_numpy()
                xg = df[(df["week"] == w) & (df["group"] == g2)][f].dropna().to_numpy()
                try:
                    if len(xn) >= 2 and len(xg) >= 2:
                        _, pv = mannwhitneyu(xn, xg, alternative="two-sided")
                    else:
                        pv = float("nan")
                except Exception:
                    pv = float("nan")
                row[f"mw_p_normal_vs_{g2}"] = pv
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "group_stats_per_week.csv", index=False)
    return out


def make_group_plots(df: pd.DataFrame, peaks_df: pd.DataFrame | None = None,
                     peaks_s1_df: pd.DataFrame | None = None):
    """挑出最能区分两组的关键特征作图（轨迹 + 热图）。"""
    key_features = [
        # 原有核心指标
        ("amp_norm_mean",    "Peak amplitude / baseline (norm)"),
        ("amp_mean",         "Peak amplitude above baseline (Ω)"),
        ("amp_cv",           "Amplitude CV (force homogeneity)"),
        ("ipi_median",       "Median inter-peak interval (s)"),
        ("ipi_cv",           "IPI CV (rhythm regularity)"),
        ("baseline_mean",    "Baseline resistance (Ω)"),
        ("baseline_drift_pct", "Baseline drift within session (%)"),
        ("burst_density_per_min", "Bursts per minute"),
        ("jumps_per_min",     "Jumps per minute (cumulative)"),
        ("jumps_per_active_min", "Jumps per active minute (motion intensity)"),
        ("detect_recall",    "Detected peaks / labeled jumps"),
        ("peaks_in_label",   "Detected peaks in labels"),
        # v5 峰形态/频域/时序/基线/峰间
        ("peak_fwhm_median",        "Peak FWHM median (s)"),
        ("peak_rise_time_median",   "Peak rise time median (s)"),
        ("peak_fall_time_median",   "Peak fall time median (s)"),
        ("peak_asymmetry_median",   "Peak asymmetry  (fall-rise)/(fall+rise)"),
        ("psd_peak_freq_hz",        "PSD peak frequency (Hz)"),
        ("psd_jump_band_ratio",     "PSD jump-band [0.5-4Hz] energy ratio"),
        ("burst_duration_median",   "Burst duration median (s)"),
        ("silence_median_s",        "Silence between bursts median (s)"),
        ("silence_max_s",           "Silence max (s)"),
        ("silence_cv",              "Silence CV"),
        ("baseline_short_drift_p95","Baseline 30s-window range p95 (Ω)"),
        ("baseline_zero_cross_per_min", "Baseline zero-crossings per min"),
        ("baseline_rms",            "Baseline RMS (Ω)"),
        ("amp_autocorr_lag1",       "Amplitude autocorr lag-1"),
        ("amp_trend_slope_per_min", "Amplitude trend slope (Ω/min)"),
        ("ipi_trend_slope_s_per_min","IPI trend slope (s/min)"),
        # v7 新增
        ("amp_skewness",            "Amplitude skewness"),
        ("amp_kurtosis",            "Amplitude kurtosis"),
        ("amp_large_ratio",         "Large peak (>P75) ratio"),
        ("peaks_per_burst_median",  "Peaks per burst (median)"),
        ("peaks_per_burst_cv",      "Peaks per burst CV"),
        ("intra_burst_ipi_median",  "Intra-burst IPI (s)"),
        ("intra_burst_amp_cv",      "Intra-burst amplitude CV"),
        ("burst_decay_slope",       "Burst decay slope (norm)"),
        ("amp_ratio_first_vs_second_half", "Amp 1st/2nd half ratio"),
        ("ipi_ratio_first_vs_second_half", "IPI 1st/2nd half ratio"),
        ("warmup_time_to_first_burst", "Warmup time to 1st burst (s)"),
        ("sample_entropy",          "Sample entropy"),
        ("ipi_entropy",             "IPI entropy (bits)"),
        ("fano_factor",             "Fano factor (30s bins)"),
        ("recovery_time_median",    "Recovery time median (s)"),
        ("post_burst_overshoot",    "Post-burst baseline overshoot (Ω)"),
    ]
    for fname, title in key_features:
        if fname not in df.columns:
            continue
        try:
            feature_trajectory(df, fname, title,
                               OUT_FIG / f"trajectory_{fname}.png",
                               ylabel=title)
        except Exception as e:
            print(f"[warn] trajectory {fname}: {e}")
        try:
            heatmap_feature(df, fname, title,
                            OUT_FIG / f"heatmap_{fname}.png")
        except Exception as e:
            print(f"[warn] heatmap {fname}: {e}")

    # 对“量级跨度大/个体差异掩盖趋势”的特征额外生成 per-rabbit/W1 归一化轨迹图
    norm_targets = [
        ("amp_mean",                "Peak amplitude (Ω)"),
        ("amp_norm_mean",           "Peak amplitude / baseline"),
        ("baseline_mean",           "Baseline resistance (Ω)"),
        ("baseline_rms",            "Baseline RMS (Ω)"),
        ("baseline_short_drift_p95","Baseline 30s drift p95 (Ω)"),
        ("peak_fwhm_median",        "Peak FWHM (s)"),
        ("psd_peak_freq_hz",        "PSD peak freq (Hz)"),
        ("psd_jump_band_ratio",     "PSD jump-band ratio"),
        ("psd_total_power",         "PSD total power"),
        ("burst_duration_median",   "Burst duration (s)"),
        ("silence_median_s",        "Silence between bursts (s)"),
        ("jumps_per_active_min",    "Jumps per active min"),
        ("burst_density_per_min",   "Bursts per min"),
    ]
    for fname, title in norm_targets:
        if fname not in df.columns:
            continue
        try:
            feature_trajectory_normalized(
                df, fname, title,
                OUT_FIG / f"trajectory_{fname}__norm_w1.png",
                ylabel=title)
        except Exception as e:
            print(f"[warn] trajectory_norm {fname}: {e}")

    # 分组子图 + z-score 归一化：看组内统一趋势
    zscore_targets = [
        ("amp_norm_mean",           "Peak amplitude / baseline"),
        ("amp_mean",                "Peak amplitude (Ω)"),
        ("amp_cv",                  "Amplitude CV"),
        ("ipi_median",             "IPI median (s)"),
        ("ipi_cv",                 "IPI CV"),
        ("burst_density_per_min",  "Bursts / min"),
        ("jumps_per_active_min",   "Jumps / active min"),
        ("peak_fwhm_median",       "Peak FWHM (s)"),
        ("peak_asymmetry_median",  "Peak asymmetry"),
        ("psd_peak_freq_hz",       "PSD peak freq (Hz)"),
        ("psd_jump_band_ratio",    "PSD jump-band ratio"),
        ("burst_duration_median",  "Burst duration (s)"),
        ("silence_median_s",       "Silence (s)"),
        ("baseline_short_drift_p95","Baseline drift p95"),
        ("amp_autocorr_lag1",      "Amplitude autocorr lag-1"),
        ("ipi_trend_slope_s_per_min","IPI trend slope"),
        # v7 新增
        ("amp_skewness",            "Amplitude skewness"),
        ("amp_kurtosis",            "Amplitude kurtosis"),
        ("peaks_per_burst_median",  "Peaks per burst"),
        ("intra_burst_ipi_median",  "Intra-burst IPI (s)"),
        ("intra_burst_amp_cv",      "Intra-burst amp CV"),
        ("burst_decay_slope",       "Burst decay slope"),
        ("amp_ratio_first_vs_second_half", "Amp 1st/2nd half"),
        ("warmup_time_to_first_burst", "Warmup to 1st burst (s)"),
        ("sample_entropy",          "Sample entropy"),
        ("ipi_entropy",             "IPI entropy"),
        ("fano_factor",             "Fano factor"),
        ("recovery_time_median",    "Recovery time (s)"),
        ("post_burst_overshoot",    "Post-burst overshoot (Ω)"),
    ]
    for fname, title in zscore_targets:
        if fname not in df.columns:
            continue
        try:
            feature_trajectory_zscore_split(
                df, fname, title,
                OUT_FIG / f"trajectory_{fname}__zscore.png")
        except Exception as e:
            print(f"[warn] trajectory_zscore {fname}: {e}")

    # ---- v9 重设计：per-rabbit per-peak 三子图箱线图 ----
    # 每个特征一张大图，含 3 个子图：W1 / W3 / each rabbit's last week。
    # 每个子图每只兔子一个箱（该周 in-label 所有峰的该特征分布）。
    # 不画「组均值」箱。
    box_dir = OUT_FIG / "boxplots"
    box_dir.mkdir(parents=True, exist_ok=True)
    if peaks_df is not None and not peaks_df.empty:
        # (feature, title, yscale)
        # log 适合「跨数量级 + 全为正」的特征：rel_baseline 系列、steepness、绝对幅值
        # linear 适合形态学/对称性等量级有界的特征
        per_peak_targets = [
            # ΔR/R₀ = (R − R₀) / R₀：相对本地基线的变化率，跨兔可比
            ("amp_rel_baseline",       "ΔR/R₀ = (R−R₀)/R₀",                  "log"),
            ("prominence_rel_baseline","Peak prominence / local baseline",     "log"),
            ("peak_area_rel_baseline", "Peak area (amp×fwhm) / baseline",       "log"),
            # 峰形态（与量纲无关，本身跨兔可比）
            ("peak_fwhm",              "Peak FWHM (s)",                        "linear"),
            ("peak_rise_time",         "Peak rise time (s)",                   "linear"),
            ("peak_fall_time",         "Peak fall time (s)",                   "linear"),
            ("peak_asymmetry",         "Peak asymmetry (fall-rise)/(fall+rise)", "linear"),
            ("peak_steepness",         "Peak steepness (Ω/s)",                 "log"),
        ]
        for fname, title, yscale in per_peak_targets:
            if fname not in peaks_df.columns:
                continue
            try:
                per_rabbit_box_panels(
                    peaks_df, fname, title,
                    box_dir / f"boxpanels_{fname}.png",
                    ylabel=title, weeks_to_show=("w1", "w3", "last"),
                    normalize="none", yscale=yscale)
                per_rabbit_box_panels(
                    peaks_df, fname, title,
                    box_dir / f"boxpanels_{fname}__norm_w1.png",
                    ylabel=title, weeks_to_show=("w1", "w3", "last"),
                    normalize="per_rabbit_w1", yscale=yscale)
            except Exception as e:
                print(f"[warn] boxpanels {fname}: {e}")

    # 频率/节律箱线图：使用 Stage 1 per-peak 表（基于全部 Stage 1 窗口内峰）
    if peaks_s1_df is not None and not peaks_s1_df.empty:
        s1_targets = [
            ("intra_burst_ipi", "Intra-burst IPI (s) [Stage 1]", "linear"),
        ]
        for fname, title, yscale in s1_targets:
            if fname not in peaks_s1_df.columns:
                continue
            try:
                per_rabbit_box_panels(
                    peaks_s1_df, fname, title,
                    box_dir / f"boxpanels_{fname}.png",
                    ylabel=title, weeks_to_show=("w1", "w3", "last"),
                    normalize="none", yscale=yscale)
                per_rabbit_box_panels(
                    peaks_s1_df, fname, title,
                    box_dir / f"boxpanels_{fname}__norm_w1.png",
                    ylabel=title, weeks_to_show=("w1", "w3", "last"),
                    normalize="per_rabbit_w1", yscale=yscale)
            except Exception as e:
                print(f"[warn] boxpanels_s1 {fname}: {e}")


def main():
    print("== Step 1: per-session pipeline (load → preprocess → detect → align → features)")
    df, peaks_df, peaks_s1_df = run_all(plot=True)
    print(f"== {len(df)} sessions processed.")
    print(f"== {len(peaks_df)} peaks (Stage 2 in-label) collected.")
    print(f"== {len(peaks_s1_df)} peaks (Stage 1 in-label) collected.")

    print("== Step 2: group stats per week (Mann-Whitney U)")
    gs = group_stats(df)
    print(f"== group_stats rows = {len(gs)}")

    print("== Step 3: group comparison plots")
    make_group_plots(df, peaks_df, peaks_s1_df)

    print("== Step 4: individual-level deep analysis (early-vs-late + temporal patterns)")
    from analysis.individual_analysis import main as ind_main
    ind_main()

    print("\n== Done.\n   Tables: ", OUT_TAB)
    print("   Figures:", OUT_FIG)
    print("   Intermediate:", OUT_INTER)


if __name__ == "__main__":
    main()
