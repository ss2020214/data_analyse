# -*- coding: utf-8 -*-
"""
特征提取（per session）：每只兔子每周一条记录。

特征定义（每项后续会写入 report.md 说明）：

[基线/漂移]
  baseline_mean         全局基础阻值（中位）
  baseline_drift_pct    会话首尾基线相对漂移 (last-first)/first × 100%
  baseline_slope_pct_min 基线一次拟合斜率（%/min，相对均值）

[峰幅与发力]
  amp_mean              检出峰相对基线的均值（Ω）
  amp_median            检出峰相对基线的中位数（Ω）
  amp_std               峰幅标准差
  amp_cv                变异系数（std/mean），反映"发力均一性"
  amp_norm_mean         峰幅 / 基线均值（无量纲，跨兔子可比）

[节律]
  ipi_median            inter-peak interval（s）中位数
  ipi_cv                IPI 变异系数（节律规整度）
  burst_count           label 窗口数量（含跳跃的“突发”次数）
  burst_density_per_min 每分钟突发数（只看“起动几次”，不看每次跳几下）
  jumps_per_min         每分钟累计跳跃次数 = jumps_labeled / (duration/60)
  jumps_per_active_min  每分钟运动密度 = jumps_labeled / (active_duration/60)
  active_duration_s     标注活动段累计时长（s）= Σ(end-start)

[活跃度]
  peaks_in_label        落入 label 窗口的检出峰数
  jumps_labeled         人工标注的总跳数
  detect_recall         检出/标注 比例（≤1 表示漏检；>1 表示器件捕获了标注以外的峰）
  hit_rate              检出峰落入窗口的比例（对齐质量）
  lag_estimated         本周估计的 sensor-label 延迟（s）

[宏观]
  duration_s            会话总时长
  outlier_ratio         预处理判定为污染并插值的比例
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.signal import peak_widths, welch


# --- 新增特征计算 ---------------------------------------------------

def _peak_shape_features(pp: dict, peaks: dict, in_label: np.ndarray) -> dict:
    """峰形态特征：FWHM、上升/下降时间、不对称性。只统计 label 窗内的峰。"""
    out = {"peak_fwhm_median": np.nan, "peak_rise_time_median": np.nan,
           "peak_fall_time_median": np.nan, "peak_asymmetry_median": np.nan}
    if peaks["idx"].size == 0 or not in_label.any():
        return out
    d = pp["detrended"]; fs = pp["fs"]
    idx = peaks["idx"][in_label]
    if idx.size == 0:
        return out
    try:
        # rel_height=0.5 → 半高宽（FWHM），也是最稳定的宽度测量位置
        # 上升/下降时间同样在 0.5 高度处计算，避免 0.9 处峰顶噪声放大
        widths_h, _, l50, r50 = peak_widths(d, idx, rel_height=0.5)
        fwhm = widths_h / fs
        rise = (idx - l50) / fs
        fall = (r50 - idx) / fs
        denom = rise + fall
        asym = np.where(denom > 0, (fall - rise) / denom, 0.0)
        out["peak_fwhm_median"] = float(np.nanmedian(fwhm))
        out["peak_rise_time_median"] = float(np.nanmedian(rise))
        out["peak_fall_time_median"] = float(np.nanmedian(fall))
        out["peak_asymmetry_median"] = float(np.nanmedian(asym))
    except Exception as e:
        warnings.warn(f"_peak_shape_features: peak_widths failed ({e}), "
                      f"n_peaks={idx.size}")
    return out


def _spectral_features(pp: dict) -> dict:
    """频域特征：Welch PSD 上的主频 + 跳跃频带 [0.5, 4] Hz 能量占比。"""
    out = {"psd_peak_freq_hz": np.nan, "psd_jump_band_ratio": np.nan,
           "psd_total_power": np.nan}
    d = pp["detrended"]; fs = pp["fs"]
    if d.size < 64:
        return out
    try:
        nperseg = min(d.size, max(64, int(8 * fs)))  # ♈8 s
        f, P = welch(d, fs=fs, nperseg=nperseg, detrend="constant")
        if P.sum() <= 0:
            return out
        # 去除 DC
        P[0] = 0.0
        # 主频：跳跃生理学上在 ~0.5–3 Hz，在该区间找峰谱
        band_lo, band_hi = 0.5, min(4.0, fs / 2 - 0.05)
        m = (f >= band_lo) & (f <= band_hi)
        if m.any():
            sub_P = P[m]; sub_f = f[m]
            j = int(np.argmax(sub_P))
            out["psd_peak_freq_hz"] = float(sub_f[j])
            out["psd_jump_band_ratio"] = float(sub_P.sum() / P.sum())
        out["psd_total_power"] = float(P.sum())
    except Exception as e:
        warnings.warn(f"_spectral_features: welch PSD failed ({e}), "
                      f"signal_len={d.size}")
    return out


def _temporal_pattern_features(pp: dict, label: pd.DataFrame, lag: float) -> dict:
    """时序模式：burst 持续时间、静默期、跨度间隔。仅需 label。"""
    out = {"burst_duration_median": np.nan, "burst_duration_max": np.nan,
           "silence_median_s": np.nan, "silence_max_s": np.nan,
           "silence_cv": np.nan}
    if label.empty:
        return out
    durs = (label["end"] - label["start"]).to_numpy()
    if durs.size:
        out["burst_duration_median"] = float(np.median(durs))
        out["burst_duration_max"] = float(np.max(durs))
    # 静默期 = 相邻两窗之间的间隔
    if len(label) >= 2:
        s = label["start"].to_numpy(); e = label["end"].to_numpy()
        order = np.argsort(s)
        s = s[order]; e = e[order]
        sil = s[1:] - e[:-1]
        sil = sil[sil > 0]
        if sil.size:
            out["silence_median_s"] = float(np.median(sil))
            out["silence_max_s"] = float(np.max(sil))
            mu = np.mean(sil)
            out["silence_cv"] = float(np.std(sil, ddof=1) / mu) \
                if mu > 0 and sil.size > 1 else 0.0
    return out


def _baseline_stability_features(pp: dict) -> dict:
    """基线稳定性：短期漂移幅度、穿越均值的频次、波动总能量。"""
    out = {"baseline_short_drift_p95": np.nan,
           "baseline_zero_cross_per_min": np.nan,
           "baseline_rms": np.nan}
    bl = pp["baseline"]; fs = pp["fs"]
    if bl.size < int(60 * fs):
        return out
    win = max(int(30 * fs), 30)            # 30 s 滑动
    s = pd.Series(bl)
    rng = s.rolling(win, center=True, min_periods=win // 2).max() - \
          s.rolling(win, center=True, min_periods=win // 2).min()
    out["baseline_short_drift_p95"] = float(np.nanpercentile(rng.to_numpy(), 95))
    # 穿越均值的频次（反映波动频率）
    centered = bl - np.nanmean(bl)
    sign = np.sign(centered)
    zc = int(np.sum(np.diff(sign) != 0))
    duration_min = bl.size / fs / 60.0
    out["baseline_zero_cross_per_min"] = zc / max(duration_min, 1e-6)
    out["baseline_rms"] = float(np.sqrt(np.nanmean(centered ** 2)))
    return out


def _peak_relation_features(pp: dict, peaks: dict, in_label: np.ndarray) -> dict:
    """峰间关系：相邻峰幅自相关 r1、峰幅随时间趋势、IPI 趋势。"""
    out = {"amp_autocorr_lag1": np.nan, "amp_trend_slope_per_min": np.nan,
           "ipi_trend_slope_s_per_min": np.nan}
    if not in_label.any():
        return out
    tp = peaks["t_peak"][in_label]
    amp = peaks["amp_above_baseline"][in_label]
    if tp.size < 4:
        return out
    order = np.argsort(tp)
    tp = tp[order]; amp = amp[order]
    # 相邻峰幅相关系数 r1
    a0, a1 = amp[:-1], amp[1:]
    if np.std(a0) > 0 and np.std(a1) > 0:
        out["amp_autocorr_lag1"] = float(np.corrcoef(a0, a1)[0, 1])
    # 峰幅随时间的线性趋势（单位：原始幅度 / min）
    t_min = (tp - tp[0]) / 60.0
    if t_min[-1] > 0:
        out["amp_trend_slope_per_min"] = float(np.polyfit(t_min, amp, 1)[0])
    # IPI 趋势：随时间 IPI 是变快还是变慢
    ipi = np.diff(tp)
    t_mid = (tp[:-1] + tp[1:]) / 2.0
    t_mid_min = (t_mid - t_mid[0]) / 60.0
    if t_mid_min[-1] > 0 and ipi.size >= 3:
        out["ipi_trend_slope_s_per_min"] = float(np.polyfit(t_mid_min, ipi, 1)[0])
    return out


# --- v7 新增特征组 -------------------------------------------------------

def _amp_distribution_features(peaks: dict, in_label: np.ndarray) -> dict:
    """峰幅分布形态：偏度、峰度、大峰占比。"""
    out = {"amp_skewness": np.nan, "amp_kurtosis": np.nan,
           "amp_large_ratio": np.nan}
    amp = peaks["amp_above_baseline"][in_label] if in_label.any() else np.array([])
    if amp.size < 4:
        return out
    from scipy.stats import skew, kurtosis
    out["amp_skewness"] = float(skew(amp, nan_policy="omit"))
    out["amp_kurtosis"] = float(kurtosis(amp, nan_policy="omit"))  # excess kurtosis
    p75 = np.percentile(amp, 75)
    out["amp_large_ratio"] = float(np.sum(amp > p75) / amp.size)
    return out


def _burst_structure_features(peaks: dict, in_label: np.ndarray,
                              win_id: np.ndarray) -> dict:
    """Burst 内部结构：每 burst 峰数/节奏/幅度变异/衰减斜率。"""
    out = {"peaks_per_burst_median": np.nan, "peaks_per_burst_cv": np.nan,
           "intra_burst_ipi_median": np.nan, "intra_burst_amp_cv": np.nan,
           "burst_decay_slope": np.nan}
    if not in_label.any():
        return out
    tp = peaks["t_peak"][in_label]
    amp = peaks["amp_above_baseline"][in_label]
    wid = win_id[in_label]
    unique_w = np.unique(wid[wid >= 0])
    if unique_w.size < 2:
        return out

    ppb_list, ipi_intra_list, amp_cv_list, decay_list = [], [], [], []
    for w in unique_w:
        mask = wid == w
        tp_w = tp[mask]
        amp_w = amp[mask]
        order = np.argsort(tp_w)
        tp_w = tp_w[order]
        amp_w = amp_w[order]
        ppb_list.append(tp_w.size)
        if tp_w.size >= 2:
            ipi_w = np.diff(tp_w)
            ipi_intra_list.extend(ipi_w.tolist())
        if amp_w.size >= 2:
            mu = np.mean(amp_w)
            if mu > 0:
                amp_cv_list.append(float(np.std(amp_w, ddof=1) / mu))
            # 衰减斜率：幅度随 peak index 的线性斜率（归一化到均值）
            if amp_w.size >= 3:
                idx_norm = np.arange(amp_w.size, dtype=float)
                slope = np.polyfit(idx_norm, amp_w, 1)[0]
                decay_list.append(slope / mu if mu > 0 else 0.0)

    ppb = np.array(ppb_list)
    out["peaks_per_burst_median"] = float(np.median(ppb))
    if ppb.size > 1 and np.mean(ppb) > 0:
        out["peaks_per_burst_cv"] = float(np.std(ppb, ddof=1) / np.mean(ppb))
    if ipi_intra_list:
        out["intra_burst_ipi_median"] = float(np.median(ipi_intra_list))
    if amp_cv_list:
        out["intra_burst_amp_cv"] = float(np.median(amp_cv_list))
    if decay_list:
        out["burst_decay_slope"] = float(np.median(decay_list))
    return out


def _fatigue_features(peaks: dict, in_label: np.ndarray,
                      win_id: np.ndarray, label: pd.DataFrame,
                      pp: dict) -> dict:
    """Session 内疑劳/适应：前后半幅值比、前后半 IPI 比、首次 burst 延迟。"""
    out = {"amp_ratio_first_vs_second_half": np.nan,
           "ipi_ratio_first_vs_second_half": np.nan,
           "warmup_time_to_first_burst": np.nan}
    t = pp["t"]
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    if duration <= 0:
        return out
    mid_time = t[0] + duration / 2.0

    # 幅值前后半比
    if in_label.any():
        tp = peaks["t_peak"][in_label]
        amp = peaks["amp_above_baseline"][in_label]
        first_half = amp[tp < mid_time]
        second_half = amp[tp >= mid_time]
        if first_half.size >= 2 and second_half.size >= 2:
            m1, m2 = np.mean(first_half), np.mean(second_half)
            if m2 > 0:
                out["amp_ratio_first_vs_second_half"] = float(m1 / m2)
        # IPI 前后半比
        order = np.argsort(tp)
        tp_s = tp[order]
        wid_s = win_id[in_label][order]
        ipi_all = np.diff(tp_s)
        same = wid_s[1:] == wid_s[:-1]
        ipi_v = ipi_all[same]
        tp_mid_v = (tp_s[:-1] + tp_s[1:])[same] / 2.0
        if ipi_v.size >= 4:
            i1 = ipi_v[tp_mid_v < mid_time]
            i2 = ipi_v[tp_mid_v >= mid_time]
            if i1.size >= 2 and i2.size >= 2:
                m1i, m2i = np.mean(i1), np.mean(i2)
                if m2i > 0:
                    out["ipi_ratio_first_vs_second_half"] = float(m1i / m2i)

    # warmup: session 开始到第一个 burst 的时间
    if not label.empty:
        first_start = label["start"].min()
        out["warmup_time_to_first_burst"] = float(first_start - t[0])
    return out


def _sample_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """简化版 Sample Entropy。m=嵌入维度，r_mult=容差倍数(* std)。"""
    n = x.size
    if n < 50:
        return np.nan
    r = r_mult * np.std(x, ddof=1)
    if r < 1e-12:
        return np.nan
    # 降采样加速（如果 n > 2000）
    if n > 2000:
        step = max(1, n // 2000)
        x = x[::step]
        n = x.size
    def _count_matches(dim):
        count = 0
        for i in range(n - dim):
            for j in range(i + 1, n - dim):
                if np.max(np.abs(x[i:i+dim] - x[j:j+dim])) <= r:
                    count += 1
        return count
    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else np.nan


def _complexity_features(pp: dict, peaks: dict, in_label: np.ndarray,
                         win_id: np.ndarray) -> dict:
    """信号复杂性：样本熵、IPI 信息熵、Fano 因子。"""
    out = {"sample_entropy": np.nan, "ipi_entropy": np.nan,
           "fano_factor": np.nan}
    d = pp["detrended"]
    # Sample Entropy（用 detrended 降采样）
    out["sample_entropy"] = _sample_entropy(d)

    # IPI 信息熵（将 IPI 分桶 → Shannon entropy）
    if in_label.any():
        tp = peaks["t_peak"][in_label]
        wid = win_id[in_label]
        order = np.argsort(tp)
        tp_s = tp[order]
        wid_s = wid[order]
        ipi = np.diff(tp_s)
        same = wid_s[1:] == wid_s[:-1]
        ipi_v = ipi[same]
        if ipi_v.size >= 8:
            # 10 等宽 bin
            counts, _ = np.histogram(ipi_v, bins=min(10, ipi_v.size // 2))
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            out["ipi_entropy"] = float(-np.sum(probs * np.log2(probs)))

    # Fano Factor（将 session 分成 30s 时间窗，统计每窗峰数的方差/均值）
    if peaks["t_peak"].size >= 4:
        t = pp["t"]
        t0, t1 = t[0], t[-1]
        bin_width = 30.0  # 30 s
        edges = np.arange(t0, t1 + bin_width, bin_width)
        if edges.size >= 3:
            counts, _ = np.histogram(peaks["t_peak"], bins=edges)
            mu = np.mean(counts)
            if mu > 0:
                out["fano_factor"] = float(np.var(counts, ddof=1) / mu)
    return out


def _recovery_features(pp: dict, label: pd.DataFrame) -> dict:
    """跨 Burst 恢复动力学：恢复时间、基线过冲。"""
    out = {"recovery_time_median": np.nan, "post_burst_overshoot": np.nan}
    if label.empty or len(label) < 2:
        return out
    bl = pp["baseline"]
    raw = pp["r_clean"]
    fs = pp["fs"]
    t = pp["t"]
    bl_mean = np.nanmean(bl)

    recovery_times, overshoots = [], []
    ends = label["end"].sort_values().to_numpy()
    starts = label["start"].sort_values().to_numpy()
    for i in range(len(ends) - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1] if i + 1 < len(starts) else t[-1]
        if gap_end <= gap_start:
            continue
        # 在序列中定位 gap 区间
        idx_s = int((gap_start - t[0]) * fs)
        idx_e = int((gap_end - t[0]) * fs)
        idx_s = max(0, min(idx_s, len(raw) - 1))
        idx_e = max(idx_s + 1, min(idx_e, len(raw)))
        seg = raw[idx_s:idx_e]
        if seg.size < int(fs):
            continue
        # 恢复时间：从 gap 开始到信号回到 bl_mean ± 1倍 baseline_rms 的时间
        bl_rms = np.sqrt(np.nanmean((bl[idx_s:idx_e] - bl_mean) ** 2)) \
            if idx_e - idx_s > 10 else np.nanstd(bl)
        thr = bl_mean + max(bl_rms, 1.0)
        below = seg < thr
        if below.any():
            first_below = int(np.argmax(below))
            recovery_times.append(first_below / fs)
        # 过冲：gap 内基线最小值相对 bl_mean 的偏差（负=下冲）
        seg_bl = bl[idx_s:idx_e] if idx_e <= len(bl) else bl[idx_s:]
        if seg_bl.size > 0:
            overshoots.append(float(np.min(seg_bl) - bl_mean))

    if recovery_times:
        out["recovery_time_median"] = float(np.median(recovery_times))
    if overshoots:
        out["post_burst_overshoot"] = float(np.median(overshoots))
    return out


# --- 主入口 -----------------------------------------------------------


def session_peak_table_stage1(pp_stage1: dict, peaks_stage1: dict,
                              label: pd.DataFrame, align: dict) -> pd.DataFrame:
    """基于 Stage 1 峰生成 per-peak 表，用于频率/节律类箱线图。

    每行一个 Stage 1 窗口内峰，列含：
      t_peak, win_id, intra_burst_ipi
    直接基于 Stage 1 的所有窗口内峰，不受 Stage 2 峰数限制。
    """
    from .peaks import assign_peaks_to_windows
    if peaks_stage1["idx"].size == 0:
        return pd.DataFrame()
    win_id = assign_peaks_to_windows(peaks_stage1, label, align["lag"])
    in_label = win_id >= 0
    if not in_label.any():
        return pd.DataFrame()

    t_peak = peaks_stage1["t_peak"][in_label]
    wid = win_id[in_label]
    n = t_peak.size

    # 同一窗口内相邻 Stage 1 峰的时间间隔
    intra_ipi = np.full(n, np.nan)
    order = np.argsort(t_peak)
    t_sorted = t_peak[order]
    w_sorted = wid[order]
    for i in range(1, n):
        if w_sorted[i] == w_sorted[i - 1]:
            intra_ipi[order[i]] = float(t_sorted[i] - t_sorted[i - 1])

    return pd.DataFrame({
        "t_peak": t_peak,
        "win_id": wid,
        "intra_burst_ipi": intra_ipi,
    })

def session_peak_table(pp: dict, peaks: dict, label: pd.DataFrame,
                       align: dict, peaks_stage1: dict = None,
                       pp_stage1: dict = None) -> pd.DataFrame:
    """返回一个 session 内所有 in-label 峰的 per-peak DataFrame。
    每行一个峰，列含：t_peak / win_id / amp_above_baseline / peak_fwhm /
    peak_rise_time / peak_fall_time / peak_asymmetry / intra_burst_ipi。
    供后续『per-rabbit per-week 箱线图」使用。

    intra_burst_ipi 使用 Stage 1 峰计算（频率特征）。
    """
    from .peaks import assign_peaks_to_windows
    if peaks["idx"].size == 0:
        return pd.DataFrame()
    win_id = assign_peaks_to_windows(peaks, label, align["lag"])
    in_label = win_id >= 0
    if not in_label.any():
        return pd.DataFrame()

    d = pp["detrended"]; fs = pp["fs"]
    bl = pp["baseline"]
    idx_global = peaks["idx"][in_label]
    t_peak = peaks["t_peak"][in_label]
    amp = peaks["amp_above_baseline"][in_label]
    wid = win_id[in_label]
    n = idx_global.size

    # 峰处的局部动态基线值（rolling P50/P30，取决于 baseline_quantile 参数）
    baseline_local = bl[idx_global].astype(float)
    bl_safe = np.where(np.abs(baseline_local) > 1e-6, baseline_local, np.nan)
    # 相对幅值：相对于本地基线的阶跃比例（无量纲，跨兔子可比）
    amp_rel = amp.astype(float) / bl_safe

    # prominence（find_peaks 给出，可能为空数组）
    prom_arr = peaks.get("prominence", np.array([]))
    if prom_arr.size == peaks["idx"].size:
        prominence = prom_arr[in_label].astype(float)
    else:
        prominence = np.full(n, np.nan)
    prominence_rel = prominence / bl_safe

    fwhm = np.full(n, np.nan)
    rise = np.full(n, np.nan)
    fall = np.full(n, np.nan)
    asym = np.full(n, np.nan)
    try:
        # rel_height=0.5 → 半高宽（FWHM）统一用于所有宽度测量，稳定可靠
        widths_h, _, l50, r50 = peak_widths(d, idx_global, rel_height=0.5)
        fwhm = widths_h / fs
        rise = (idx_global - l50) / fs
        fall = (r50 - idx_global) / fs
        denom = rise + fall
        with np.errstate(invalid="ignore", divide="ignore"):
            asym = np.where(denom > 0, (fall - rise) / denom, 0.0)
    except Exception as e:
        warnings.warn(f"session_peak_table: peak_widths failed ({e}), "
                      f"n_peaks={idx_global.size}")

    # 峰面积（近似）：amp × fwhm，代表单次跳跃事件总能量
    peak_area = amp.astype(float) * fwhm
    peak_area_rel = peak_area / bl_safe
    # 峰陡度（峰高/峰宽）：反映“锐峰”还是“钝峰”
    with np.errstate(invalid="ignore", divide="ignore"):
        peak_steepness = np.where(fwhm > 0, amp.astype(float) / fwhm, np.nan)

    # intra_burst_ipi: 使用 Stage 1 峰计算（频率特征应基于未清洗的寻峰结果）
    intra_ipi = np.full(n, np.nan)
    if peaks_stage1 is not None and peaks_stage1["idx"].size > 0:
        win_id_s1 = assign_peaks_to_windows(peaks_stage1, label, align["lag"])
        in_label_s1 = win_id_s1 >= 0
        if in_label_s1.any():
            tp_s1 = peaks_stage1["t_peak"][in_label_s1]
            wid_s1 = win_id_s1[in_label_s1]
            # 对每个 Stage 2 峰，找同一窗口内 Stage 1 的前一个峰计算间隔
            for i in range(n):
                w_cur = wid[i]
                t_cur = t_peak[i]
                mask_same = wid_s1 == w_cur
                if not mask_same.any():
                    continue
                tp_same = tp_s1[mask_same]
                tp_before = tp_same[tp_same < t_cur]
                if tp_before.size > 0:
                    intra_ipi[i] = float(t_cur - tp_before.max())
    else:
        # 回退：用 Stage 2 峰自身计算
        order = np.argsort(t_peak)
        t_sorted = t_peak[order]
        w_sorted = wid[order]
        for i in range(1, n):
            if w_sorted[i] == w_sorted[i - 1]:
                intra_ipi[order[i]] = float(t_sorted[i] - t_sorted[i - 1])

    return pd.DataFrame({
        "t_peak": t_peak,
        "win_id": wid,
        "baseline_local": baseline_local,
        "amp_above_baseline": amp,
        "amp_rel_baseline": amp_rel,
        "prominence": prominence,
        "prominence_rel_baseline": prominence_rel,
        "peak_fwhm": fwhm,
        "peak_rise_time": rise,
        "peak_fall_time": fall,
        "peak_asymmetry": asym,
        "peak_area": peak_area,
        "peak_area_rel_baseline": peak_area_rel,
        "peak_steepness": peak_steepness,
        "intra_burst_ipi": intra_ipi,
    })


def session_features(pp: dict, peaks: dict, label: pd.DataFrame,
                     align: dict, peaks_stage1: dict = None,
                     pp_stage1: dict = None) -> dict:
    """
    提取会话级特征。

    参数：
      pp:            Stage 2 完整预处理结果
      peaks:         Stage 2 峰（用于幅值特征）
      label:         标注数据
      align:         对齐结果
      peaks_stage1:  Stage 1 峰（用于时域特征），若 None 则回退使用 peaks
      pp_stage1:     Stage 1 预处理结果，若 None 则回退使用 pp

    双阶段特征分离：
      - 时域特征（IPI、节律、频率等）: 使用 peaks_stage1
      - 幅值特征（amp_mean、amp_cv、amp_norm_mean等）: 使用 peaks (stage2)
    """
    # 回退处理
    if peaks_stage1 is None:
        peaks_stage1 = peaks
    if pp_stage1 is None:
        pp_stage1 = pp

    t = pp["t"]
    bl = pp["baseline"]
    duration = float(t[-1] - t[0])

    # 基线
    bl_mean = float(np.nanmean(bl))
    bl_first, bl_last = float(np.nanmedian(bl[:max(20, len(bl) // 30)])), \
                        float(np.nanmedian(bl[-max(20, len(bl) // 30):]))
    bl_drift_pct = (bl_last - bl_first) / max(abs(bl_first), 1e-6) * 100.0
    # 一次拟合（按分钟）
    if len(t) > 5:
        coef = np.polyfit((t - t[0]) / 60.0, bl, 1)
        bl_slope_pct = float(coef[0] / max(abs(bl_mean), 1e-6) * 100.0)
    else:
        bl_slope_pct = float("nan")

    # label 窗内的峰：用对齐后的 lag
    from .peaks import assign_peaks_to_windows

    # === Stage 2 峰 → 幅值特征 ===
    win_id = assign_peaks_to_windows(peaks, label, align["lag"])
    in_label = win_id >= 0
    amp_in = peaks["amp_above_baseline"][in_label] \
        if peaks["amp_above_baseline"].size else np.array([])

    if amp_in.size:
        amp_mean = float(np.mean(amp_in))
        amp_median = float(np.median(amp_in))
        amp_std = float(np.std(amp_in, ddof=1)) if amp_in.size > 1 else 0.0
        amp_cv = amp_std / amp_mean if amp_mean > 0 else float("nan")
        amp_norm_mean = amp_mean / max(abs(bl_mean), 1e-6)
    else:
        amp_mean = amp_median = amp_std = float("nan")
        amp_cv = amp_norm_mean = float("nan")

    # === Stage 1 峰 → 时域特征（IPI、节律） ===
    win_id_s1 = assign_peaks_to_windows(peaks_stage1, label, align["lag"])
    in_label_s1 = win_id_s1 >= 0
    n_s1_in_label = int(in_label_s1.sum())

    # IPI（用 Stage 1 峰，因为时域信息更完整）
    # 添加生理学过滤：排除不可能的超短 IPI（< 0.2s，兑子无法这么快跳跃）
    # 这能有效排除噪声峰-噪声峰之间的伪间隔
    IPI_MIN_PHYSIOLOGICAL = 0.2  # 秒（最快 5次/秒 = 300次/分钟，生理上限）
    if n_s1_in_label > 2:
        tp_s1 = peaks_stage1["t_peak"][in_label_s1]
        ipi = np.diff(np.sort(tp_s1))
        wid_in_s1 = win_id_s1[in_label_s1]
        order = np.argsort(tp_s1)
        wid_sorted = wid_in_s1[order]
        same_win = wid_sorted[1:] == wid_sorted[:-1]
        ipi = ipi[same_win]
        # 生理学过滤：排除噪声峰导致的超短伪间隔
        ipi = ipi[ipi >= IPI_MIN_PHYSIOLOGICAL]
        if ipi.size > 0:
            ipi_median = float(np.median(ipi))
            ipi_cv = float(np.std(ipi, ddof=1) / np.mean(ipi)) \
                if np.mean(ipi) > 0 and ipi.size > 1 else 0.0
        else:
            ipi_median = ipi_cv = float("nan")
    else:
        ipi_median = ipi_cv = float("nan")

    burst_count = int(len(label))
    burst_density = burst_count / (duration / 60.0) if duration > 0 else float("nan")
    jumps_labeled = int(label["count"].sum()) if not label.empty else 0
    # 「按跳数累计」指标：每分钟累计跳跃次数，
    # 与 burst_density_per_min 互补：后者数「起动次数」，本指标数「总跳数」。
    jumps_per_min = jumps_labeled / (duration / 60.0) if duration > 0 else float("nan")
    # 「运动密度」指标：以标注活动段累计时长为分母，不含静息时间稀释。
    active_duration_s = float((label["end"] - label["start"]).sum()) if not label.empty else 0.0
    jumps_per_active_min = (jumps_labeled / (active_duration_s / 60.0)
                            if active_duration_s > 0 else float("nan"))
    peaks_in_label = int(in_label.sum())
    detect_recall = peaks_in_label / jumps_labeled if jumps_labeled > 0 else float("nan")

    # 新增特征集
    shape_f   = _peak_shape_features(pp, peaks, in_label)
    spec_f    = _spectral_features(pp_stage1)  # 频域特征用 Stage 1 信号
    temp_f    = _temporal_pattern_features(pp, label, align.get("lag", 0.0))
    blstab_f  = _baseline_stability_features(pp)
    # 峰间关系：IPI趋势等时域特征用 Stage 1
    relat_f   = _peak_relation_features(pp_stage1, peaks_stage1, in_label_s1)

    # v7 新增
    ampdist_f  = _amp_distribution_features(peaks, in_label)  # 幅值分布用 Stage 2
    # 频率/节律相关特征用 Stage 1 峰（background.md 第三节第5条）
    burst_sf   = _burst_structure_features(peaks_stage1, in_label_s1, win_id_s1)
    fatigue_f  = _fatigue_features(peaks_stage1, in_label_s1, win_id_s1, label, pp_stage1)
    complex_f  = _complexity_features(pp_stage1, peaks_stage1, in_label_s1, win_id_s1)
    recov_f    = _recovery_features(pp, label)

    out = {
        "baseline_mean": bl_mean,
        "baseline_drift_pct": bl_drift_pct,
        "baseline_slope_pct_min": bl_slope_pct,
        # 幅值特征（Stage 2）
        "amp_mean": amp_mean,
        "amp_median": amp_median,
        "amp_std": amp_std,
        "amp_cv": amp_cv,
        "amp_norm_mean": amp_norm_mean,
        # 时域特征（Stage 1）
        "ipi_median": ipi_median,
        "ipi_cv": ipi_cv,
        "burst_count": burst_count,
        "burst_density_per_min": burst_density,
        "jumps_per_min": jumps_per_min,
        "jumps_per_active_min": jumps_per_active_min,
        "active_duration_s": active_duration_s,
        # 检测统计
        "peaks_in_label": peaks_in_label,
        "peaks_in_label_stage1": n_s1_in_label,
        "jumps_labeled": jumps_labeled,
        "detect_recall": detect_recall,
        "detect_recall_stage1": n_s1_in_label / jumps_labeled
                                if jumps_labeled > 0 else float("nan"),
        "hit_rate": float(align.get("hit_rate", 0.0)),
        "lag_estimated": float(align.get("lag", 0.0)),
        "duration_s": duration,
        "outlier_ratio": float(pp.get("outlier_ratio", 0.0)),
        "unstable_ratio": float(pp.get("unstable_ratio", 0.0)),
        "fast_rise_ratio": float(pp.get("fast_rise_ratio", 0.0)),
    }
    out.update(shape_f)
    out.update(spec_f)
    out.update(temp_f)
    out.update(blstab_f)
    out.update(relat_f)
    out.update(ampdist_f)
    out.update(burst_sf)
    out.update(fatigue_f)
    out.update(complex_f)
    out.update(recov_f)
    return out
