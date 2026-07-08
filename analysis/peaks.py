# -*- coding: utf-8 -*-
"""
峰值检测 + label/signal 时间对齐。

双阶段策略：
- Stage 1：在最小化预处理后的信号上寻峰，参数宽松，用于时域特征（IPI、节律等）。
- Stage 2：在完整清洗后的信号上寻峰，参考标准运动特征调整参数，
  用于幅值特征（delta_r/r0 等）。

对齐：以候选峰为信号事件，用 label 给定的窗口（向外扩 pad 秒）构造布尔遮罩，
  在合理延迟范围内搜索使「峰落入窗口」比例最大的 lag。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .config import PARAMS


# --- Stage 1 寻峰参数（极宽松，用于时域特征/IPI） ---
# 目标：检出率接近100%以保证 IPI ≈ 60/jump_freq
# 设计原则：宁多勿少，因为 IPI 只计算同一窗口内相邻峰的间隔，
# 窗外噪声峰不影响 IPI 计算。
STAGE1_PARAMS = {
    "peak_min_distance_s": 0.15,     # 允许快速连击识别（0.18→0.15）
    "peak_prominence_mad_k": 0.5,   # 大幅降低 prominence（1.2→0.5）
    "peak_height_mad_k": 0.3,       # 大幅降低 height（0.8→0.3）
}


def detect_peaks_stage1(pp: dict, params=PARAMS) -> dict:
    """
    Stage 1 寻峰：在最小化预处理后的信号上进行初步寻峰。

    特点：
    - 参数极宽松（低 prominence / height 阈值）
    - 不屏蔽 baseline 不稳定区和快速跃变区
    - 目的：尽可能多地检出峰，用于时域特征计算（IPI、跳跃频率等）
    - 检出率目标 ≥ 90%，确保 IPI ≈ 60/jump_frequency

    返回 dict: idx, t_peak, amp_detrended, amp_above_baseline, prominence
    """
    fs = pp["fs"]
    d = pp["detrended"]
    mad = pp["mad"]

    # 使用 Stage 1 专用参数（极宽松）
    distance = max(1, int(STAGE1_PARAMS["peak_min_distance_s"] * fs))
    height = STAGE1_PARAMS["peak_height_mad_k"] * mad
    prom = STAGE1_PARAMS["peak_prominence_mad_k"] * mad

    idx, props = find_peaks(d, distance=distance, height=height, prominence=prom)

    # Stage 1 不屏蔽任何区域（尽可能保留峰）
    return {
        "idx": idx,
        "t_peak": pp["t"][idx],
        "amp_detrended": d[idx],
        "amp_above_baseline": d[idx],
        "prominence": props.get("prominences", np.array([])),
    }


def detect_peaks_stage1_adaptive(pp: dict, label: pd.DataFrame,
                                  lag: float, params=PARAMS) -> dict:
    """
    双向自适应 Stage 1 寻峰：欠检时降低阈值，过检时提高阈值。

    流程：
    1. 用 STAGE1_PARAMS 初始检测
    2. 计算每个窗口的检出数 vs label count
    3. 若 recall < 0.85 → 降低阈值重检
    4. 若 recall > 1.3  → 提高阈值重检（过检保护）
    5. 最多迭代 4 轮

    确保 IPI 和跳跃频率的一致性：60/ipi_median ≈ jump_frequency
    """
    fs = pp["fs"]
    d = pp["detrended"]
    mad = pp["mad"]
    t = pp["t"]
    # 自适应 pad
    pads = _adaptive_pad(label, params)

    # 初始参数
    height_k = STAGE1_PARAMS["peak_height_mad_k"]
    prom_k = STAGE1_PARAMS["peak_prominence_mad_k"]
    distance = max(1, int(STAGE1_PARAMS["peak_min_distance_s"] * fs))

    target_lo = 0.85   # 检出率下限
    target_hi = 1.3    # 检出率上限
    max_iter = 4
    decay_down = 0.5   # 欠检时参数衰减比例
    grow_up = 1.8      # 过检时参数增长比例

    best_result = None
    best_recall = 0.0

    for iteration in range(max_iter):
        height = height_k * mad
        prom = prom_k * mad

        idx, props = find_peaks(d, distance=distance, height=height, prominence=prom)

        result = {
            "idx": idx,
            "t_peak": t[idx],
            "amp_detrended": d[idx],
            "amp_above_baseline": d[idx],
            "prominence": props.get("prominences", np.array([])),
        }

        # 如果没有 label 或 label 为空，直接返回
        if label.empty:
            return result

        # 计算窗口内检出 vs 标注
        total_labeled = int(label["count"].sum())
        if total_labeled == 0:
            return result

        t_peak = result["t_peak"]
        s_arr = label["start"].to_numpy() + lag - pads
        e_arr = label["end"].to_numpy() + lag + pads
        total_in_window = 0
        for j in range(len(s_arr)):
            total_in_window += int(np.sum((t_peak >= s_arr[j]) & (t_peak <= e_arr[j])))

        recall = total_in_window / total_labeled

        # 记录最接近目标的结果
        if best_result is None or abs(recall - 1.0) < abs(best_recall - 1.0):
            best_result = result
            best_recall = recall

        # 已在目标范围内，停止
        if target_lo <= recall <= target_hi:
            break

        # 欠检：降低阈值
        if recall < target_lo:
            height_k *= decay_down
            prom_k *= decay_down
        # 过检：提高阈值
        elif recall > target_hi:
            height_k *= grow_up
            prom_k *= grow_up

    return best_result


# --- Stage 2 寻峰（精确，参考标准运动，用于幅值特征） ---

def detect_peaks_stage2(pp: dict, params=PARAMS, ref_profile: dict = None,
                        height_scale: float = 1.0) -> dict:
    """
    Stage 2 寻峰：在完整清洗后的信号上进行精确寻峰。

    特点：
    - 屏蔽 baseline 不稳定区和快速跃变区
    - 峰必须满足：局部最高 + 高于当前基线（background.md 第三节第3条）
    - 标准运动峰值为自由运动的上限（background.md 第二节第4条）
      超过标准运动峰值的视为伪迹并剔除
    - 内置生理学峰过滤：仅保留上升时间 < 下降时间的峰

    ref_profile 影响（核心逻辑）：
    - 标准运动下用力更大，其峰值相对基线的变化是自由运动的上限
    - max_allowed = local_baseline * ref_rel_amp * tolerance
    - 超过此上限的峰视为传感器伪迹

    height_scale: 阈值缩放因子（默认 1.0），用于自适应衰减时降低阈值
    """
    fs = pp["fs"]
    d = pp["detrended"]
    mad = pp["mad"]
    bl = pp["baseline"]

    # 默认参数
    prom_k = params["peak_prominence_mad_k"]
    height_k = params["peak_height_mad_k"]
    min_dist = params["peak_min_distance_s"]

    # 默认阈值（基于 MAD），应用 height_scale 缩放
    height_thr = height_k * mad * height_scale
    prom_thr = (prom_k * mad * height_scale) if prom_k > 0 else None

    # 根据标准运动参考调整参数
    if ref_profile is not None:
        ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
        ref_fwhm = ref_profile.get("peak_fwhm_median", 0.4)

        if ref_rel_amp > 0:
            # 标准运动峰值为上限，自由运动峰值应小于此值
            # 用标准运动幅值的一半作为峰高下限（太小的可能是噪声）
            local_baseline = float(np.nanmedian(bl))
            expected_amp = local_baseline * ref_rel_amp

            # 如果 MAD 阈值高于期望幅值，适当降低以不漏检
            if height_thr > expected_amp * 0.5:
                height_thr = expected_amp * 0.3
                if prom_thr is not None:
                    prom_thr = expected_amp * 0.25

        # 标准峰宽可用于调整最小峰间距
        if ref_fwhm > 0:
            min_dist = max(min_dist, ref_fwhm * 0.5)

    distance = max(1, int(min_dist * fs))

    # 构建 find_peaks 参数（prominence 为 None 时不传）
    fp_kwargs = {"distance": distance, "height": height_thr}
    if prom_thr is not None and prom_thr > 0:
        fp_kwargs["prominence"] = prom_thr
    idx, props = find_peaks(d, **fp_kwargs)

    # 屏蔽伪迹区域的峰
    block = None
    for key in ("unstable_mask", "fast_rise_mask"):
        m = pp.get(key)
        if m is None:
            continue
        block = m.copy() if block is None else (block | m)
    if block is not None and idx.size > 0 and block.any():
        keep = ~block[idx]
        idx = idx[keep]
        for k in ("prominences", "peak_heights"):
            if k in props:
                props[k] = props[k][keep]

    # 峰高上限过滤：标准运动峰值为自由运动的上限
    # background.md: "标准运动下峰值相对于基线的变化是自由运动的上限"
    # 允许 1.2x 容差（考虑测量误差）
    UPPER_LIMIT_TOLERANCE = 1.2
    if ref_profile is not None and idx.size > 0:
        ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
        if ref_rel_amp > 0:
            local_bl_at_peak = bl[idx]
            max_allowed = local_bl_at_peak * ref_rel_amp * UPPER_LIMIT_TOLERANCE
            peak_amp = d[idx]
            keep_upper = peak_amp <= max_allowed
            if not keep_upper.all():
                idx = idx[keep_upper]
                for k in ("prominences", "peak_heights"):
                    if k in props:
                        props[k] = props[k][keep_upper]

    # 生理学峰过滤：仅保留上升时间 < 下降时间的峰
    if idx.size > 0:
        pk_tmp = {"idx": idx, "t_peak": pp["t"][idx],
                  "amp_detrended": d[idx], "amp_above_baseline": d[idx],
                  "prominence": props.get("prominences", np.array([]))}
        weights = _peak_physiological_weights(pp, pk_tmp)
        physio_mask = weights >= 1.0
        if physio_mask.any():
            idx = idx[physio_mask]
            for k in ("prominences", "peak_heights"):
                if k in props:
                    props[k] = props[k][physio_mask]
        else:
            idx = np.array([], dtype=int)

    return {
        "idx": idx,
        "t_peak": pp["t"][idx],
        "amp_detrended": d[idx],
        "amp_above_baseline": d[idx],
        "prominence": props.get("prominences", np.array([])),
    }


def detect_peaks_stage2_adaptive(pp: dict, label, lag: float,
                                 ref_profile=None, params=PARAMS,
                                 min_recall: float = 0.5) -> dict:
    """自适应 Stage 2 寻峰：检出率不足时逐级降低阈值。

    逻辑：
    1. 默认 height_scale=1.0 检测
    2. 计算窗口内 recall = 窗口内峰数 / label总 count
    3. 若 recall < min_recall，以 height_scale=0.7 重试
    4. 若仍不足，以 height_scale=0.5 重试
    """
    total_count = int(label["count"].sum()) if not label.empty else 0
    if total_count <= 0:
        return detect_peaks_stage2(pp, params, ref_profile)

    for scale in [1.0, 0.7, 0.5]:
        pk = detect_peaks_stage2(pp, params, ref_profile, height_scale=scale)
        if pk["idx"].size == 0:
            continue
        win_id = assign_peaks_to_windows(pk, label, lag)
        recall = (win_id >= 0).sum() / total_count
        if recall >= min_recall:
            break
    return pk


def _windows_mask(t: np.ndarray, label: pd.DataFrame, lag: float, pad: float) -> np.ndarray:
    """对给定 lag，构造时间网格上的窗口遮罩（True 表示属于跳跃窗口）。"""
    mask = np.zeros_like(t, dtype=bool)
    if label.empty:
        return mask
    s = label["start"].to_numpy() + lag - pad
    e = label["end"].to_numpy() + lag + pad
    for s_i, e_i in zip(s, e):
        mask |= (t >= s_i) & (t <= e_i)
    return mask


# --- 向后兼容别名 ---
def detect_peaks(pp: dict, params=PARAMS):
    """[兼容] 等价于 detect_peaks_stage2，不传 ref_profile。"""
    return detect_peaks_stage2(pp, params, ref_profile=None)


def _adaptive_pad(label: pd.DataFrame, params=PARAMS) -> np.ndarray:
    """计算每个窗口的自适应 pad：窗口越短 pad 越大，窗口越长 pad 越小。

    逻辑：短窗口标注误差占比大，需要更多容忍；
         长窗口标注精度高，边界误差相对小，pad 可以小。

    公式：pad = min(max_pad, base / max(1, dur/2))
      - 1s窗口: pad = min(1.0, 1.0/0.5) = 1.0s
      - 3s窗口: pad = min(1.0, 1.0/1.5) = 0.67s
      - 6s窗口: pad = min(1.0, 1.0/3.0) = 0.33s
    """
    if label.empty:
        return np.array([])
    max_pad = params.get("label_window_pad_max_s", 1.0)
    base = params.get("label_window_pad_base_s", 1.0)
    durations = (label["end"] - label["start"]).to_numpy()
    pads = base / np.maximum(1.0, durations / 2.0)
    return np.minimum(max_pad, pads)


def _peak_physiological_weights(pp: dict, peaks: dict) -> np.ndarray:
    """计算每个峰的生理学权重：上升时间 < 下降时间的峰权重更高。

    生理学依据：兔子跳跃产生的传感器响应具有"快速上升、缓慢下降"的特征，
    因为跳跃施力瞬间导致电阻快速变化，而着陆后恢复是渐进的。
    器械噪声/伪迹则通常不具有这种不对称性。

    判定方法：对每个峰找到前一个谷值和后一个谷值，
    若 前谷→峰 的时间 < 峰→后谷 的时间，则判定为生理学峰。

    返回：每个峰的权重（生理峰=1.0，非生理峰=0.2）
    """
    if peaks["idx"].size == 0:
        return np.array([])

    d = pp["detrended"]
    t = pp["t"]
    idx = peaks["idx"]
    n = len(d)

    weights = np.full(idx.size, 0.2)  # 默认低权重

    # 对每个峰，在相邻峰之间寻找谷值
    # 第一个峰的前谷：信号起始 ~ 该峰之间的最小值位置
    # 最后一个峰的后谷：该峰 ~ 信号末尾之间的最小值位置
    # 中间峰：前谷在前一个峰~该峰之间，后谷在该峰~下一个峰之间

    for k, pk_idx in enumerate(idx):
        # 确定前谷搜索区间
        left_bound = int(idx[k - 1]) if k > 0 else 0
        # 确定后谷搜索区间
        right_bound = int(idx[k + 1]) if k < len(idx) - 1 else n - 1

        # 前谷：left_bound ~ pk_idx 之间的最小值位置
        if pk_idx <= left_bound:
            continue
        prev_trough_idx = left_bound + int(np.argmin(d[left_bound:pk_idx]))

        # 后谷：pk_idx+1 ~ right_bound 之间的最小值位置
        if right_bound <= pk_idx:
            continue
        next_trough_idx = pk_idx + 1 + int(np.argmin(d[pk_idx + 1:right_bound + 1]))

        # 计算上升时间和下降时间
        rise_time = t[pk_idx] - t[prev_trough_idx]
        fall_time = t[next_trough_idx] - t[pk_idx]

        # 判定：前谷→峰的时间 < 峰→后谷的时间 → 生理学峰
        if rise_time > 0 and fall_time > 0 and rise_time < fall_time:
            weights[k] = 1.0  # 生理学峰，满权重

    return weights


def estimate_lag(pp: dict, peaks: dict, label: pd.DataFrame, params=PARAMS) -> dict:
    """
    时间对齐：在 [lag_min, lag_max] 中网格搜索最优 lag。

    优化目标（窗口峰数匹配，对应 background.md 第三节第4条）：
      对每个候选 lag，将标注窗口平移后，计算每个窗口内检出峰数与标注 count 的差异。
      score(lag) = -Σ|peaks_in_window_i - label_count_i|
      选择 score 最高（即差异之和最小）的 lag。
      若多个 lag 同分，取 |lag| 最小者。

    该准则确保对齐后每个窗口内峰数尽量接近人工标注的跳跃次数，
    而非单纯追求总检出率。
    """
    if label.empty or peaks["idx"].size == 0:
        return {"lag": 0.0, "score": 0.0, "hit_rate": 0.0,
                "n_peaks": int(peaks["idx"].size)}

    t_peak = peaks["t_peak"]
    lags = np.arange(params["lag_search_min_s"],
                     params["lag_search_max_s"] + 1e-9,
                     params["lag_search_step_s"])
    # 自适应 pad
    pads = _adaptive_pad(label, params)
    s_arr = label["start"].to_numpy()
    e_arr = label["end"].to_numpy()
    counts = label["count"].to_numpy().astype(int)
    total_labeled = int(counts.sum())

    scores = np.zeros_like(lags, dtype=float)
    hit_rates = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        s = s_arr + lag - pads
        e = e_arr + lag + pads

        # 逐窗口计算：|窗口内峰数 - 标注 count| 之和
        total_error = 0
        total_hits = 0
        for j in range(len(s_arr)):
            n_in_window = int(np.sum((t_peak >= s[j]) & (t_peak <= e[j])))
            total_error += abs(n_in_window - counts[j])
            total_hits += min(n_in_window, counts[j])

        # score = 负的误差和（越大越好，0 = 完美匹配）
        scores[i] = -float(total_error)
        hit_rates[i] = total_hits / max(1, total_labeled)

    best = float(scores.max())
    cand = np.where(scores >= best - 1e-9)[0]
    j = cand[np.argmin(np.abs(lags[cand]))]
    return {"lag": float(lags[j]),
            "score": float(scores[j]),
            "hit_rate": float(hit_rates[j]),
            "lags": lags, "scores": scores,
            "n_peaks": int(peaks["idx"].size)}


def label_to_intervals(label: pd.DataFrame, lag: float,
                       pad=None, params=PARAMS):
    """返回 (start_arr, end_arr) 已加上 lag 和自适应 pad。

    pad: 若为 float，使用固定值；若为 None，使用自适应计算。
    """
    if label.empty:
        return np.array([]), np.array([])
    if pad is None:
        pads = _adaptive_pad(label, params)
    elif np.isscalar(pad):
        pads = np.full(len(label), pad)
    else:
        pads = np.asarray(pad)
    return (label["start"].to_numpy() + lag - pads,
            label["end"].to_numpy() + lag + pads)


def assign_peaks_to_windows(peaks: dict, label: pd.DataFrame, lag: float,
                            pad=None, params=PARAMS):
    """对每个峰，标记其所在 window 序号（-1 表示窗外）。

    pad: float时用固定值, None时用自适应(min(max_pad, dur×ratio))。
    """
    s_arr, e_arr = label_to_intervals(label, lag, pad=pad, params=params)
    win_id = np.full(peaks["t_peak"].shape, -1, dtype=int)
    for k, tp in enumerate(peaks["t_peak"]):
        m = (tp >= s_arr) & (tp <= e_arr)
        if m.any():
            win_id[k] = int(np.argmax(m))
    return win_id
