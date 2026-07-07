# -*- coding: utf-8 -*-
"""
数据加载与预处理：

1. load_session(rabbit, week)
   - 读取原始 xlsx（无表头），返回 (time[s], resistance[Ω]) numpy 数组
   - 处理：去除 NaN/字符串污染、按时间升序、若时间不严格单调则去重

2. load_label(rabbit, week)
   - 返回 DataFrame: start, end, count

3. preprocess(t, r, params)
   - 重采样到统一频率
   - MAD 离群剔除并插值
   - 动态基线（rolling median）
   - detrended = signal - baseline
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from .config import path_orig, path_label, PARAMS, WEEKS


# --- IO -----------------------------------------------------------------

def _normalize_week_name(name: str) -> str:
    """统一周编号：'1w' -> 'w1'。"""
    n = name.strip().lower()
    if n.startswith("w"):
        return n
    if n.endswith("w"):
        return "w" + n[:-1]
    return n


def _open_book(path) -> dict:
    """读取整个工作簿，返回 {规范化sheet名: DataFrame}。"""
    xf = pd.ExcelFile(path)
    out = {}
    for s in xf.sheet_names:
        out[_normalize_week_name(s)] = xf.parse(s, header=None)
    return out


def load_session(rabbit: str, week: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取一只兔子某一周的原始信号。"""
    book = _open_book(path_orig(rabbit))
    if week not in book:
        raise KeyError(f"{rabbit} 没有 sheet {week}")
    df = book[week]
    if df is None or df.shape[0] == 0 or df.shape[1] < 2:
        # 空 sheet（如 068 w5/w6 未采集）返回空数组，由上游跳过该 session
        return np.array([]), np.array([])
    # 取前两列：时间(s) / 阻值(Ω)，逐元素转 numeric，污染行变 NaN
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    r = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(r)
    t, r = t[mask], r[mask]
    # 时间升序
    idx = np.argsort(t)
    t, r = t[idx], r[idx]
    # 去除时间重复（保留首条）
    _, uniq = np.unique(t, return_index=True)
    uniq = np.sort(uniq)
    t, r = t[uniq], r[uniq]
    return t, r


def load_label(rabbit: str, week: str) -> pd.DataFrame:
    """读取标注表。统一列名为 start/end/count。"""
    xf = pd.ExcelFile(path_label(rabbit))
    sheet_map = {_normalize_week_name(s): s for s in xf.sheet_names}
    if week not in sheet_map:
        return pd.DataFrame(columns=["start", "end", "count"])
    df = xf.parse(sheet_map[week])
    df = df.iloc[:, :3].copy()
    df.columns = ["start", "end", "count"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df["end"] > df["start"]) & (df["count"] > 0)].reset_index(drop=True)
    return df


# --- 预处理 -------------------------------------------------------------

def _rolling_mad(x: np.ndarray, win: int) -> np.ndarray:
    """rolling MAD（中位数绝对偏差），用于阈值。"""
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=max(3, win // 4)).median()
    mad = (s - med).abs().rolling(win, center=True,
                                  min_periods=max(3, win // 4)).median()
    return mad.to_numpy()


def preprocess_raw(t: np.ndarray, r: np.ndarray, params=PARAMS):
    """
    Stage 1 纯原始预处理：不做任何数据清洗，仅做重采样和动态基线。
    对应 background.md 第三节第5条："第一次不做任何的数据清洗先寻峰"。

    处理步骤：
      1. 重采样到统一时间网格（仅插值，不剔除任何点）
      2. 动态基线（滚动低分位数，用于判断峰方向）
      3. detrended = signal - baseline

    与 preprocess_minimal 的区别：
      - 不做硬截断（hard clipping）
      - 不做 MAD 离群剔除
      - 不检测 baseline 不稳定区/快速跃变

    基线设计原则（针对 IPI 精度优化）：
      - 窗口更宽（×3.0）：避免密集跳跃抬高基线，导致小峰被吞没
      - 分位数更低（P15）：确保基线贴近真实静息位，不被峰本身抬高

    返回字典（与 preprocess 兼容的键）：
      t, r_raw, r_clean, baseline, detrended, mad, fs
    """
    fs = params["resample_fs"]
    # 1) 重采样到均匀时间网格
    t_uni = np.arange(t[0], t[-1], 1.0 / fs)
    r_uni = np.interp(t_uni, t, r)

    # 2) 动态基线：滚动低分位数（跟踪基线漂移，不使用固定值）
    # 优化：用更宽窗口（×3.0）+ 更低分位数（P15）
    # 原因：密集跳跃时，较窄窗口的 median/P30 会被峰本身抬高，
    # 导致 detrended 幅度变小、小峰被淡化而漏检。
    win_b = max(11, int(params["baseline_window_s"] * 3.0 * fs))  # 25s*3=75s窗口
    if win_b % 2 == 0:
        win_b += 1
    bq = 0.15  # Stage 1 专用：P15（比 Stage 2 的 P30 更低，确保基线不吞峰）
    baseline = (pd.Series(r_uni)
                .rolling(win_b, center=True, min_periods=win_b // 3)
                .quantile(bq)
                .bfill().ffill().to_numpy())

    # 3) detrended
    detrended = r_uni - baseline
    mad_global = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
    if mad_global == 0:
        mad_global = float(np.nanstd(detrended)) or 1.0

    return {
        "fs": fs,
        "t": t_uni,
        "r_raw": r_uni,
        "r_clean": r_uni,  # 无清洗，等于原始
        "baseline": baseline,
        "detrended": detrended,
        "mad": mad_global,
        "outlier_ratio": 0.0,
        "hard_outlier_ratio": 0.0,
        "unstable_mask": np.zeros(len(t_uni), dtype=bool),
        "unstable_ratio": 0.0,
        "fast_rise_mask": np.zeros(len(t_uni), dtype=bool),
        "fast_rise_ratio": 0.0,
    }


def preprocess_minimal(t: np.ndarray, r: np.ndarray, params=PARAMS):
    """
    Stage 1 最小化预处理：仅做重采样 + 硬截断 + 粗略基线，保留更多原始峰信息。
    用于 Stage 1 寻峰（时域特征提取）。

    与完整 preprocess() 的区别：
      - 不做 MAD 局部离群剔除（避免删除带噪声的真实峰）
      - baseline 窗口较宽（仅跟踪慢漂移）
      - 不检测 baseline 不稳定区/快速跃变

    返回字典（与 preprocess 兼容的键）：
      t, r_raw, r_clean, baseline, detrended, mad, fs
    """
    fs = params["resample_fs"]
    # 1) 重采样
    t_uni = np.arange(t[0], t[-1], 1.0 / fs)
    r_uni = np.interp(t_uni, t, r)

    # 2) 仅做硬截断（去除明显传感器伪迹）
    r_med0 = float(np.nanmedian(r_uni))
    hi = params.get("hard_clip_high", 1.5)
    lo = params.get("hard_clip_low", 1.0 / 1.5)
    hard_bad = (r_uni > r_med0 * hi) | (r_uni < r_med0 * lo)
    r_clean = r_uni.copy()
    if hard_bad.any():
        good_idx = np.where(~hard_bad)[0]
        if good_idx.size > 5:
            r_clean[hard_bad] = np.interp(
                np.where(hard_bad)[0], good_idx, r_uni[good_idx])

    # 3) 宽松基线（较长窗口，仅跟踪慢漂移）
    win_b = max(11, int(params["baseline_window_s"] * 1.5 * fs))  # 比正式清洗宽50%
    if win_b % 2 == 0:
        win_b += 1
    bq = float(params.get("baseline_quantile", 0.5))
    baseline = (pd.Series(r_clean)
                .rolling(win_b, center=True, min_periods=win_b // 2)
                .quantile(bq)
                .bfill().ffill().to_numpy())

    # 4) detrended
    detrended = r_clean - baseline
    mad_global = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
    if mad_global == 0:
        mad_global = float(np.nanstd(detrended)) or 1.0

    return {
        "fs": fs,
        "t": t_uni,
        "r_raw": r_uni,
        "r_clean": r_clean,
        "baseline": baseline,
        "detrended": detrended,
        "mad": mad_global,
        "outlier_ratio": float(hard_bad.mean()),
        "hard_outlier_ratio": float(hard_bad.mean()),
        "unstable_mask": np.zeros(len(t_uni), dtype=bool),
        "unstable_ratio": 0.0,
        "fast_rise_mask": np.zeros(len(t_uni), dtype=bool),
        "fast_rise_ratio": 0.0,
    }


def preprocess(t: np.ndarray, r: np.ndarray, params=PARAMS,
               ref_profile: dict = None):
    """
    Stage 2 完整预处理。可接收标准运动参考特征来自适应调整清洗参数。

    参数：
      ref_profile: 标准运动参考特征 (get_reference_profile 返回值)。
        若提供，将根据标准峰幅值与噪声水平的比例关系调整离群剔除参数，
        使清洗更具针对性，在去噪的同时保护真实峰。

    返回字典：
      t_uni, r_uni          重采样后的时间/原始阻值
      r_clean               离群剔除并插值后的阻值
      baseline              动态基线（rolling median）
      detrended             r_clean - baseline
      mad_global            detrended 的全局 MAD（用于检测阈值）
      ref_profile           传入的参考特征（便于下游使用）
    """
    fs = params["resample_fs"]
    # 1) 重采样到均匀时间网格
    t_uni = np.arange(t[0], t[-1], 1.0 / fs)
    r_uni = np.interp(t_uni, t, r)

    # --- 根据标准运动参考自适应调整参数 ---
    # 如果提供了 ref_profile，根据标准峰幅值与噪声比例调整 outlier_mad_k。
    # 理由：自由运动中噪声更大，但峰幅值与标准运动相似。
    # 若 SNR 较低（噪声大），需要提高 outlier_mad_k 以避免误删峰。
    outlier_mad_k = params["outlier_mad_k"]
    if ref_profile is not None:
        ref_snr = ref_profile.get("snr", 10.0)
        # SNR 高（标准运动干净）: 保持原参数
        # SNR 低: 适当放宽 outlier_mad_k，避免把峰当离群剔掉
        if ref_snr < 5.0:
            outlier_mad_k = max(outlier_mad_k, 8.0)
        elif ref_snr < 10.0:
            outlier_mad_k = max(outlier_mad_k, 7.0)

    # 2a) 生物学合理性硬截断：电阻不应超过全局中位数的 hard_clip_high 倍或低于 hard_clip_low 倍，
    #     这种数量级跳变属于传感器瞬态/接触不良/饱和等非生物伪迹（背景文档明确允许剔除）。
    r_med0 = float(np.nanmedian(r_uni))
    hi = params.get("hard_clip_high", 2.0)
    lo = params.get("hard_clip_low", 1.0 / 3.0)
    hard_bad = (r_uni > r_med0 * hi) | (r_uni < r_med0 * lo)
    r_uni_h = r_uni.copy()
    if hard_bad.any():
        good_idx0 = np.where(~hard_bad)[0]
        if good_idx0.size > 5:
            r_uni_h[hard_bad] = np.interp(np.where(hard_bad)[0], good_idx0,
                                          r_uni[good_idx0])

    # 2b) MAD 离群点：以 r_uni_h 为基准，与 rolling median 偏离 > k * rolling_MAD 的点视为污染
    win_out = max(5, int(params["outlier_window_s"] * fs))
    s = pd.Series(r_uni_h)
    rmed = s.rolling(win_out, center=True, min_periods=3).median()
    rmad = (s - rmed).abs().rolling(win_out, center=True, min_periods=3).median()
    global_mad = float(np.nanmedian(np.abs(r_uni_h - np.nanmedian(r_uni_h)))) or 1.0
    thr = outlier_mad_k * np.maximum(rmad.to_numpy(), global_mad * 0.5)
    bad = np.abs(r_uni_h - rmed.to_numpy()) > thr
    r_clean = r_uni_h.copy()
    if bad.any():
        good_idx = np.where(~bad)[0]
        if good_idx.size > 5:
            r_clean[bad] = np.interp(np.where(bad)[0], good_idx, r_uni_h[good_idx])
    bad = bad | hard_bad        # 合并两类污染

    # 3) 动态基线：滚动低分位数（窗 ~25 s，能滤掉单次跳跃同时跟踪慢漂移）
    #    v4：从 P50 改为 P30（默认）。信号大部分时间在静息位、偶尔向上跳起，
    #    median 会被峰本身抬高。低分位数更贴近真静息位，detrended 幅度不被吞掉。
    win_b = max(11, int(params["baseline_window_s"] * fs))
    if win_b % 2 == 0:
        win_b += 1
    bq = float(params.get("baseline_quantile", 0.5))
    baseline = (pd.Series(r_clean)
                .rolling(win_b, center=True, min_periods=win_b // 2)
                .quantile(bq)
                .to_numpy())
    # 边缘 NaN 用最近邻填充
    baseline = pd.Series(baseline).bfill().ffill().to_numpy()

    # 4) detrended
    detrended = r_clean - baseline
    mad_global = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
    if mad_global == 0:
        mad_global = float(np.nanstd(detrended)) or 1.0

    # 5) baseline 不稳定区域检测：|dR_baseline/dt| 超过 k×MAD（每秒变化多个噪底）
    #    表明发生台阶事件，该区域不参与寻峰。
    db_dt = np.abs(np.gradient(baseline)) * fs
    k_unstable = params.get("baseline_unstable_k", 10.0)
    unstable = db_dt > k_unstable * mad_global
    # 边界调肥：向外扩展 dilate_s 秒，避免台阶上升/下降两侧的过渡区间被误判
    dilate_s = params.get("baseline_unstable_dilate_s", 1.0)
    dilate_n = max(1, int(dilate_s * fs))
    if unstable.any():
        kernel = np.ones(2 * dilate_n + 1, dtype=int)
        unstable = np.convolve(unstable.astype(int), kernel, mode="same") > 0
    unstable_ratio = float(unstable.mean())

    # 6) 峰值增长速率限制：|d(detrended)/dt| 超过 k×MAD/s 视为传感器瞬态/接触跳变。
    #    与 unstable_mask 互补——后者依赖 baseline 已跟随的台阶，此处捕获 baseline
    #    尚未跟上的瞬时跃变（典型表现：detrended 一两步内跳数十~数百 MAD）。
    dd_dt = np.abs(np.gradient(detrended)) * fs
    k_rise = params.get("peak_rise_rate_mad_k", 80.0)
    fast_rise = dd_dt > k_rise * mad_global
    rise_dilate_s = params.get("peak_rise_dilate_s", 0.3)
    rise_dilate_n = max(1, int(rise_dilate_s * fs))
    if fast_rise.any():
        kernel = np.ones(2 * rise_dilate_n + 1, dtype=int)
        fast_rise = np.convolve(fast_rise.astype(int), kernel, mode="same") > 0
    fast_rise_ratio = float(fast_rise.mean())

    return {
        "fs": fs,
        "t": t_uni,
        "r_raw": r_uni,
        "r_clean": r_clean,
        "baseline": baseline,
        "detrended": detrended,
        "mad": mad_global,
        "outlier_ratio": float(bad.mean()),
        "hard_outlier_ratio": float(hard_bad.mean()),
        "unstable_mask": unstable,
        "unstable_ratio": unstable_ratio,
        "fast_rise_mask": fast_rise,
        "fast_rise_ratio": fast_rise_ratio,
        "ref_profile": ref_profile,
    }