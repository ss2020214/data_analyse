# -*- coding: utf-8 -*-
"""
文档法 ΔR/R₀（峰值−谷值 / 全局 R₀）核心实现。

口径（忠实背景文档）：\n
    ΔR/R₀ = (R_peak − R_valley) / R₀

其中：
- R_peak  : 峰处真实阻值（清洗后）
- R_valley: 该峰局部结构谷 —— find_peaks 返回的 left_base 位置处的阻值
- R₀      : 全局常量 —— 该 session 静息段（非活跃窗口）稳定分位数（默认 P20）

设计原则：
- 不动现有逻辑；本模块作为新口径的独立实现。
- 谷定位改为「find_peaks(-detrended) 检出真正局部极小」，并在峰左侧受限窗口
  （doc_valley_seek_s 内）取最近谷；保证谷在峰左、是极小值、且不会距峰过远。
- use_doc_method 默认关闭，现有图不因本模块而改变输出。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from .config import PARAMS, WEEKS

# 峰脚谷回退时，在峰±该窗口（秒）内寻找局部最小值作谷
_LOCAL_VALLEY_WIN_S = 0.5
# 无 active_mask 时，粗峰检测屏蔽其周围该窗口（秒）
_ROUGH_PEAK_SHIELD_S = 0.5


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _est_fs(t: np.ndarray) -> float:
    """从时间向量估算采样频率。"""
    if t.size < 2:
        return 5.0
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.isfinite(dt):
        return 5.0
    return 1.0 / dt


def _rough_detrend(r: np.ndarray, fs: float,
                   win_s: float = 5.0) -> np.ndarray:
    """粗去趋势：P30 滚动基线（与标准运动 P30 基线口径一致），返回 detrended。"""
    win_b = max(11, int(win_s * fs))
    if win_b % 2 == 0:
        win_b += 1
    if len(r) < win_b:
        win_b = max(5, (2 * len(r)) // 3)
        if win_b % 2 == 0:
            win_b += 1
    baseline = (pd.Series(r)
                .rolling(win_b, center=True, min_periods=max(3, win_b // 2))
                .quantile(0.30)
                .bfill().ffill().to_numpy())
    return r - baseline


def _find_left_bases(detrended: np.ndarray):
    """宽松重跑 find_peaks，返回 (峰位置数组, left_base 数组)。

    宽松参数（distance=1, height=0）保证尽量检出所有局部极大，
    从而为上游每个位置都能映射到一个结构谷（取最近峰）。"""
    idx, _ = find_peaks(detrended, distance=1, height=0.0)
    if idx.size == 0:
        return idx, np.array([], dtype=int)
    # find_peaks 仅在请求 prominence 时才计算 left_bases，这里手动用
    # peak_prominences 取左边基线（结构谷），并保护退化情况
    _lb, _, _ = peak_prominences(detrended, idx)
    lb = np.asarray(_lb, dtype=int)
    lb = np.clip(lb, 0, detrended.size - 1)
    return idx, lb


# ---------------------------------------------------------------------------
# 1) R₀：全局静息段分位数
# ---------------------------------------------------------------------------

def extract_rest_resistance_r0(t: np.ndarray, r: np.ndarray,
                               active_mask: np.ndarray = None,
                               r_quantile: float = 0.20,
                               params: dict = PARAMS):
    """从 session 静息段提取全局常量 R₀。

    参数
    ----
    t, r : 时间 / 阻值（清洗后），长度一致
    active_mask : bool 数组，True=活跃窗口（自由运动 label 含 lag+pad）。
        None 时（标准运动/无 label），自动做粗峰检测并屏蔽峰周围 ±0.5s，
        其余视为静息段；若静息段占比过低则回退整段 P5。
    r_quantile : 静息段分位数（P20=0.20）

    返回
    ----
    (R0, rest_ratio)
        R0        全局常量阻值 (Ω)
        rest_ratio 静息段占比（< doc_rest_min_ratio 时已回退并在 stderr 告警）
    """
    fs = _est_fs(t)
    rest_ratio = 0.0

    if active_mask is not None:
        rest = ~np.asarray(active_mask, dtype=bool)
        rest_ratio = float(rest.mean()) if rest.size else 0.0
    else:
        detrended = _rough_detrend(r, fs)
        idx, _ = find_peaks(detrended, height=0.0)
        rest = np.ones(len(r), dtype=bool)
        shield = max(1, int(_ROUGH_PEAK_SHIELD_S * fs))
        for p in idx:
            lo = max(0, int(p) - shield)
            hi = min(len(r), int(p) + shield)
            rest[lo:hi] = False
        rest_ratio = float(rest.mean()) if rest.size else 0.0

    min_ratio = params.get("doc_rest_min_ratio", 0.1)
    if rest.sum() < 5 or rest_ratio < min_ratio:
        # 静息过少 → 回退整段极低分位数，避免 R₀ 被峰体抬高
        print(f"[peak_trough_delta] R0 回退 P5: rest_ratio={rest_ratio:.3f} "
              f"< {min_ratio} 或静息点不足", file=None)
        R0 = float(np.nanpercentile(r, 5.0))
    else:
        R0 = float(np.nanpercentile(r[rest], r_quantile * 100.0))
    if not np.isfinite(R0) or R0 <= 0:
        R0 = float(np.nanpercentile(r, 5.0))
    return R0, rest_ratio


# ---------------------------------------------------------------------------
# 2) 文档法单峰幅度：峰值−结构谷 / R₀
# ---------------------------------------------------------------------------

def peak_trough_delta(r_clean: np.ndarray, detrended: np.ndarray,
                      positions: np.ndarray, R0: float,
                      fs: float, params: dict = PARAMS):
    """对给定位置序列逐个计算文档法 ΔR/R₀。

    参数
    ----
    r_clean     : 清洗后阻值数组
    detrended   : 去趋势信号（仅用于定位结构谷）
    positions   : 参与计算的峰/采样点位置（数组）
    R0          : 全局常量（extract_rest_resistance_r0 返回值）
    fs          : 采样频率

    返回
    ----
    (delta_r_r0, valley_r, left_base_idx)
        delta_r_r0   每个位置的文档法 ΔR/R₀ = (R_peak − R_valley) / R₀
        valley_r     每个位置对应的谷值阻值
        left_base_idx 每个位置对应的结构谷位置
    """
    positions = np.asarray(positions, dtype=int)
    n = len(r_clean)
    if positions.size == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=int))

    # 找出 detrended 的全部局部极小（真正的谷）候选
    valley_seek_s = params.get("doc_valley_seek_s", 1.5)   # 峰左侧搜索窗口（秒）
    valley_min_gap = params.get("doc_valley_min_gap_s", 0.2)
    min_gap_b = max(1, int(valley_min_gap * fs))
    seek_b = max(2, int(valley_seek_s * fs))

    # 对 -detrended 做 find_peaks 即得局部极小（谷）位置
    valleys, _ = find_peaks(-detrended, distance=min_gap_b)

    left_base_idx = np.zeros(positions.size, dtype=int)
    if valleys.size > 0:
        for i, pos in enumerate(positions):
            lo = max(0, pos - seek_b)
            hi = max(0, pos - min_gap_b)  # 严格在峰左侧，且与峰保持间隔
            # 取峰左侧搜索窗口内、距峰最近的谷
            cand = valleys[(valleys >= lo) & (valleys < hi)]
            if cand.size > 0:
                left_base_idx[i] = int(cand[-1])  # 离 pos 最近的左侧谷
            elif hi > lo:
                # 窗口内无真实极小：取窗口内 detrended 最高点（原始阻值最低）作谷
                left_base_idx[i] = int(lo + int(np.argmax(detrended[lo:hi])))
            else:
                left_base_idx[i] = max(0, pos - 1)
    else:
        # 无任何局部极小（异常）：逐个取峰左侧窗口内 detrended 最高点作谷
        for i, pos in enumerate(positions):
            lo = max(0, pos - seek_b)
            hi = max(0, pos - min_gap_b)
            if hi > lo:
                left_base_idx[i] = int(lo + int(np.argmax(detrended[lo:hi])))
            else:
                left_base_idx[i] = max(0, pos - 1)

    valley_r = r_clean[left_base_idx]
    peak_r = r_clean[positions]
    R0s = np.where(np.abs(R0) > 1e-9, R0, np.nan)
    delta_r_r0 = (peak_r - valley_r) / R0s
    return delta_r_r0, valley_r, left_base_idx


# ---------------------------------------------------------------------------
# 3) 标准运动文档法提取
# ---------------------------------------------------------------------------

def _select_standard_peaks(detrended, mad, count, fs, params=PARAMS):
    """复现上游标准运动寻峰：2×MAD → 1.5×MAD → 1.0×MAD + 数量上限分段取最大。"""
    min_dist_s = 1.0
    distance = max(1, int(min_dist_s * fs))
    peaks_idx = np.array([], dtype=int)
    for height in (2.0 * mad, 1.5 * mad, 1.0 * mad):
        idx, _ = find_peaks(detrended, distance=distance, height=height)
        if count > 0 and idx.size < count * 0.5:
            peaks_idx = idx
            continue
        peaks_idx = idx
        break

    if count > 0 and peaks_idx.size > int(count):
        n_sel = int(count)
        peak_vals = detrended[peaks_idx]
        seg_edges = np.linspace(0, len(peaks_idx), n_sel + 1)
        selected = []
        used = set()
        for si in range(n_sel):
            cand = [x for x in range(int(seg_edges[si]), int(seg_edges[si + 1]))
                    if x not in used]
            if cand:
                best = max(cand, key=lambda x: peak_vals[x])
                selected.append(best)
                used.add(best)
        if len(selected) < n_sel:
            remaining = [i for i in range(len(peaks_idx)) if i not in used]
            remaining.sort(key=lambda x: peak_vals[x], reverse=True)
            for ri in remaining:
                if len(selected) >= n_sel:
                    break
                selected.append(ri)
                used.add(ri)
        peaks_idx = peaks_idx[np.sort(selected)]
    return peaks_idx


def extract_standard_motion_delta(rabbit: str, max_per_week: int = 10,
                                  params: dict = PARAMS) -> pd.DataFrame:
    """标准运动文档法 ΔR/R₀ 提取。

    复用 load_standard_session；每组独立求 R₀（该组静息 P20）并为每峰计算
    文档法幅度；合并每周，周级双峰过滤 + 每周最多 max_per_week 均匀采样。

    返回 DataFrame: week, time_s, delta_r_r0, R0_val, valley_r, peak_r
    """
    from .standard_motion import load_standard_session, _estimate_fs

    records = []
    for week in WEEKS:
        try:
            groups = load_standard_session(rabbit, week)
        except Exception:
            continue

        week_records = []
        for grp in groups:
            t = grp["t"]
            r = grp["r"]
            count = grp["count"]
            if len(t) < 20:
                continue
            if not np.isfinite(count):
                count = 0

            # 全局 R₀（该组静息段 P20）
            R0, _ = extract_rest_resistance_r0(t, r, active_mask=None,
                                               r_quantile=params.get(
                                                   "doc_r0_quantile", 0.20),
                                               params=params)

            fs = _estimate_fs(t)
            detrended = _rough_detrend(r, fs, win_s=params.get(
                "doc_detrend_window_s", 5.0))
            mad = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
            if mad <= 0:
                mad = float(np.nanstd(detrended)) or 1.0

            peaks_idx = _select_standard_peaks(detrended, mad, count, fs, params)
            if peaks_idx.size == 0:
                continue

            delta, valley_r, _ = peak_trough_delta(r, detrended, peaks_idx,
                                                   R0, fs, params)
            for i in range(len(peaks_idx)):
                if np.isfinite(delta[i]) and delta[i] > 0:
                    week_records.append({
                        "week": week,
                        "time_s": float(t[peaks_idx[i]]),
                        "delta_r_r0": float(delta[i]),
                        "R0_val": float(R0),
                        "valley_r": float(valley_r[i]),
                        "peak_r": float(r[peaks_idx[i]]),
                    })

        # 周级双峰过滤（与上游一致：若整周呈明显双峰分布，剔除干扰群）
        if len(week_records) >= 6:
            dr_vals = np.array([p["delta_r_r0"] for p in week_records])
            dr_sorted = np.sort(dr_vals)
            ratios = dr_sorted[1:] / np.maximum(dr_sorted[:-1], 1e-10)
            max_gap_idx = int(np.argmax(ratios))
            gap_ratio = ratios[max_gap_idx]
            n_low = max_gap_idx + 1
            n_high = len(dr_sorted) - n_low
            if gap_ratio > 2.0 and n_low >= 3 and n_high >= 3:
                cutoff = (dr_sorted[max_gap_idx] + dr_sorted[max_gap_idx + 1]) / 2
                low_med = float(np.median(dr_sorted[:n_low]))
                high_med = float(np.median(dr_sorted[n_low:]))
                if high_med > 0 and low_med < 0.1 * high_med:
                    # 低群为近零噪声/谷值塌缩簇（如 035 w2），剔除低群保留高群
                    week_records = [p for p in week_records
                                    if p["delta_r_r0"] > cutoff]
                else:
                    # 原行为：高群为高值干扰群，只保留低群
                    week_records = [p for p in week_records
                                    if p["delta_r_r0"] <= cutoff]

        # 每周最多 max_per_week 个，按时间均匀选取（段内取最大）
        if week_records:
            df_week = pd.DataFrame(week_records).sort_values("time_s").reset_index(drop=True)
            if len(df_week) > max_per_week:
                t_arr = df_week["time_s"].to_numpy()
                seg_edges = np.linspace(t_arr.min() - 0.01, t_arr.max() + 0.01,
                                        max_per_week + 1)
                sel_idx = []
                for si in range(max_per_week):
                    mask = (t_arr >= seg_edges[si]) & (t_arr < seg_edges[si + 1])
                    cand = [c for c in np.where(mask)[0] if c not in sel_idx]
                    if cand:
                        best = max(cand, key=lambda x: df_week.iloc[x]["delta_r_r0"])
                        sel_idx.append(best)
                if len(sel_idx) < max_per_week:
                    remaining = [i for i in range(len(df_week)) if i not in sel_idx]
                    remaining.sort(key=lambda x: df_week.iloc[x]["delta_r_r0"],
                                   reverse=True)
                    for ri in remaining:
                        if len(sel_idx) >= max_per_week:
                            break
                        sel_idx.append(ri)
                df_week = df_week.iloc[sorted(sel_idx)]
            records.extend(df_week.to_dict("records"))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4) 自由运动文档法提取
# ---------------------------------------------------------------------------

def _build_in_window_mask(t_uni, label, lag, params=PARAMS):
    """构造自由运动活跃窗口遮罩（含 lag + 自适应 pad）。"""
    from .peaks import _adaptive_pad
    mask = np.zeros(len(t_uni), dtype=bool)
    if label.empty:
        return mask
    pads = _adaptive_pad(label, params)
    s_arr = label["start"].to_numpy() + lag - pads
    e_arr = label["end"].to_numpy() + lag + pads
    for s_i, e_i in zip(s_arr, e_arr):
        mask |= (t_uni >= s_i) & (t_uni <= e_i)
    return mask


def _upper_limit(ref_profile, params=PARAMS):
    """自由运动上限过滤：标准运动峰值 × 1.0（与旧法口径一致，保证可比）。"""
    if ref_profile is not None:
        ref_rel_amp = ref_profile.get("peak_rel_amp_p90", 0)
        if ref_rel_amp > 0:
            return ref_rel_amp * 1.0
    return None


def extract_free_motion_delta(rabbit: str, mode: str = "physio_peak",
                              n_per_week: int = 30,
                              params: dict = PARAMS) -> pd.DataFrame:
    """自由运动文档法 ΔR/R₀ 提取。

    mode ∈ {"sample", "physio_peak"}
        sample      活跃窗口内均匀采样点（文档法计算：取采样点最近结构谷作谷）
        physio_peak 活跃窗口内 Stage 2 生理学峰（幅值走文档法）

    R₀ 用当前周 active_mask 外静息段（跨组/跨周各自独立）。

    返回 DataFrame: week, time_s, delta_r_r0, mode
    """
    from .preprocess import load_session, load_label, preprocess, preprocess_raw
    from .peaks import (detect_peaks_stage1, estimate_lag,
                        detect_peaks_stage2_adaptive, assign_peaks_to_windows)
    from .standard_motion import get_reference_profile

    records = []
    for week in WEEKS:
        try:
            t, r = load_session(rabbit, week)
            if t.size < 50:
                continue
            label = load_label(rabbit, week)
            if label.empty:
                continue
        except (KeyError, FileNotFoundError):
            continue

        ref_profile = None
        try:
            ref_profile = get_reference_profile(rabbit, week)
        except Exception:
            pass

        pp = preprocess(t, r, ref_profile=ref_profile)
        t_uni = pp["t"]
        r_clean = pp["r_clean"]
        detrended = pp["detrended"]
        fs = pp["fs"]

        # 对齐
        pp_s1 = preprocess_raw(t, r)
        pk_s1 = detect_peaks_stage1(pp_s1)
        al = estimate_lag(pp_s1, pk_s1, label)
        lag = al["lag"]

        in_window = _build_in_window_mask(t_uni, label, lag, params)
        if not in_window.any():
            continue
        # 当前周静息段 R₀
        R0, _ = extract_rest_resistance_r0(
            t_uni, r_clean, active_mask=in_window,
            r_quantile=params.get("doc_r0_quantile", 0.20), params=params)

        upper = _upper_limit(ref_profile)

        if mode == "physio_peak":
            pk = detect_peaks_stage2_adaptive(pp, label, lag,
                                              ref_profile=ref_profile)
            if pk["idx"].size == 0:
                continue
            win_id = assign_peaks_to_windows(pk, label, lag)
            in_win = win_id >= 0
            if not in_win.any():
                continue
            pos = pk["idx"][in_win]
            delta, _, _ = peak_trough_delta(r_clean, detrended, pos, R0, fs, params)
            for i in range(len(pos)):
                val = delta[i]
                if not np.isfinite(val) or val <= 0:
                    continue
                if upper is not None and val > upper:
                    continue
                records.append({
                    "week": week,
                    "time_s": float(t_uni[pos[i]]),
                    "delta_r_r0": float(val),
                    "mode": mode,
                })
        else:  # sample
            idx_in = np.where(in_window)[0]
            if idx_in.size < 5:
                continue
            # 为所有窗口内点计算文档法 delta（取最近结构谷作谷）
            delta_all, _, _ = peak_trough_delta(r_clean, detrended, idx_in,
                                                R0, fs, params)
            valid = np.isfinite(delta_all) & (delta_all > 0)
            if upper is not None:
                valid = valid & (delta_all <= upper)
            chosen = np.where(valid)[0]
            if chosen.size > n_per_week:
                # 时间均匀采样（段内随机）
                pool_times = t_uni[idx_in[chosen]]
                # 注意：不能用 hash()（字符串 hash 每进程随机化，PYTHONHASHSEED），
                # 否则每次运行采样结果不同。用确定性编码作种子。
                rng = np.random.default_rng(seed=int(rabbit) * 100 + int(week[1:]))
                seg_edges = np.linspace(pool_times.min() - 0.001,
                                        pool_times.max() + 0.001, n_per_week + 1)
                sel = []
                used = set()
                for si in range(n_per_week):
                    mask = (pool_times >= seg_edges[si]) & (pool_times < seg_edges[si + 1])
                    cand = [c for c in np.where(mask)[0] if c not in used]
                    if cand:
                        pick = int(rng.choice(cand))
                        sel.append(chosen[pick])
                        used.add(pick)
                if len(sel) < n_per_week:
                    remaining = [c for c in range(len(chosen)) if c not in used]
                    if remaining:
                        need = n_per_week - len(sel)
                        extra = rng.choice(remaining, size=min(need, len(remaining)),
                                           replace=False)
                        # 注意：remaining/extra 是 pool 索引，必须经 chosen 映射回 delta_all 索引
                        sel.extend(int(chosen[x]) for x in extra)
                chosen = np.array(sel)
            for ci in chosen:
                records.append({
                    "week": week,
                    "time_s": float(t_uni[idx_in[ci]]),
                    "delta_r_r0": float(delta_all[ci]),
                    "mode": mode,
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    print("peak_trough_delta 模块自检（可配合单元测试使用）")