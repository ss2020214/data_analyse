# -*- coding: utf-8 -*-
"""
030 和 088 标准运动细节图 — 每只兔子一张大图含 2 个子图。

子图 1（左）: w1/w3/w6 各取一段含 5 个连续峰 + 首峰前谷 + 末峰后谷的
             ΔR/R₀ 时间序列。三周时段对齐到同一 x 轴段（相对时间 0 起点）。
子图 2（右）: 标准运动 mean±std errorbar（与 optimized_v3 中
             delta_r_r0_*_optimized.png 的标准运动统计子图一致）。

ΔR/R₀ = (R - R₀) / R₀，R₀ 为 P30 滚动分位数动态基线。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from .config import WEEKS, OUT_FIG
from .delta_r_r0_plot import _extract_standard_motion_peaks, WEEK_COLORS
from .standard_motion import load_standard_session, _estimate_fs

# ── 字体重置（必须在 import delta_r_r0_plot 之后，因其会覆盖字体设置）──
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 输出路径 ──────────────────────────────────────────────
OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────
TARGET_WEEKS = ["w1", "w3", "w6"]
N_PEAKS = 5  # 每段包含的峰数


# ============================================================
# 1. 标准运动 detrend + 寻峰/谷（与 _extract_standard_motion_peaks 一致）
# ============================================================
def _detrend_and_peaks(t: np.ndarray, r: np.ndarray,
                       count: float = 0) -> dict | None:
    """对一组标准运动数据做 P30 基线 detrend 并寻峰/谷。

    返回 dict: t, r, baseline, detrended, delta_r_r0, peaks, valleys, fs
    """
    if len(t) < 20:
        return None

    fs = _estimate_fs(t)
    win_b = max(11, int(5.0 * fs))
    if win_b % 2 == 0:
        win_b += 1
    # 边界处理：点数不足以支撑完整窗口时缩小
    if len(t) < win_b:
        win_b = max(5, (2 * len(t)) // 3)
        if win_b % 2 == 0:
            win_b += 1
    baseline = (pd.Series(r)
                .rolling(win_b, center=True, min_periods=max(3, win_b // 2))
                .quantile(0.30)
                .bfill().ffill().to_numpy())

    detrended = r - baseline

    # 寻峰（自适应阈值衰减：2×MAD → 1.5×MAD → 1.0×MAD）
    mad = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
    if mad <= 0:
        mad = float(np.nanstd(detrended)) or 1.0
    min_dist_s = 1.0
    distance = max(1, int(min_dist_s * fs))

    height = 2.0 * mad
    peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)
    if count > 0 and peaks_idx.size < count * 0.5:
        height = 1.5 * mad
        peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)
    if count > 0 and peaks_idx.size < count * 0.5:
        height = 1.0 * mad
        peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)

    # 谷（detrended 的局部极小）
    valleys_idx, _ = find_peaks(-detrended, distance=distance)

    # ΔR/R₀
    bl_safe = np.where(np.abs(baseline) > 1e-6, baseline, np.nan)
    delta_r_r0 = (r - baseline) / bl_safe

    return {
        "t": t, "r": r, "baseline": baseline, "detrended": detrended,
        "delta_r_r0": delta_r_r0, "peaks": peaks_idx,
        "valleys": valleys_idx, "fs": fs,
    }


# ============================================================
# 2. 枚举含 5 个连续峰 + 首尾谷的候选段
# ============================================================
def _enumerate_5peak_segments(info: dict) -> list:
    """枚举一组数据中所有含 N_PEAKS 个连续峰 + 首尾谷的候选段。

    对每个连续 N_PEAKS 峰窗口：
    - 段起点 = 首峰前最近的谷
    - 段终点 = 末峰后最近的谷

    返回候选段列表，每个元素为 dict:
        t_seg, dr_seg, peak_times, peak_dr, duration
    """
    if info is None:
        return []
    peaks = info["peaks"]
    t = info["t"]
    valleys = info["valleys"]
    if peaks.size < N_PEAKS:
        return []

    candidates = []
    for i in range(peaks.size - N_PEAKS + 1):
        seg_peaks = peaks[i:i + N_PEAKS]
        first_peak = int(seg_peaks[0])
        last_peak = int(seg_peaks[-1])

        # 首峰前最近的谷
        pre_valleys = valleys[valleys < first_peak]
        if pre_valleys.size > 0:
            start_idx = int(pre_valleys[-1])
        else:
            # 回退：向前延伸约 1 个平均峰间隔
            avg_int = (t[last_peak] - t[first_peak]) / (N_PEAKS - 1)
            start_t = t[first_peak] - avg_int
            start_idx = max(0, int(np.searchsorted(t, start_t)))

        # 末峰后最近的谷
        post_valleys = valleys[valleys > last_peak]
        if post_valleys.size > 0:
            end_idx = int(post_valleys[0])
        else:
            avg_int = (t[last_peak] - t[first_peak]) / (N_PEAKS - 1)
            end_t = t[last_peak] + avg_int
            end_idx = min(len(t) - 1, int(np.searchsorted(t, end_t)))

        if end_idx <= start_idx:
            continue

        t_seg = t[start_idx:end_idx + 1]
        dr_seg = info["delta_r_r0"][start_idx:end_idx + 1]
        peak_times = t[seg_peaks]
        peak_dr = info["delta_r_r0"][seg_peaks]
        duration = float(t_seg[-1] - t_seg[0])

        candidates.append({
            "t_seg": t_seg,
            "dr_seg": dr_seg,
            "peak_times": peak_times,
            "peak_dr": peak_dr,
            "duration": duration,
        })

    return candidates


def _get_week_candidates(rabbit: str, week: str) -> list:
    """获取某周所有组的所有候选 5 峰段。"""
    try:
        groups = load_standard_session(rabbit, week)
    except Exception:
        return []

    all_candidates = []
    for grp in groups:
        t = grp["t"]
        r = grp["r"]
        count = grp["count"]
        if not np.isfinite(count):
            count = 0
        info = _detrend_and_peaks(t, r, count)
        if info is None:
            continue
        cands = _enumerate_5peak_segments(info)
        all_candidates.extend(cands)
    return all_candidates


# 每周最多保留的候选数（按时长排序后截断，防止组合爆炸）
MAX_CANDIDATES_PER_WEEK = 30


def _select_consistent_segments(rabbit: str) -> dict:
    """为三周选取时间跨度尽可能一致的 5 峰段。

    策略：
    1. 每周枚举所有候选 5 峰段（含首尾谷），按时长排序后取最短的若干个
    2. 暴力搜索所有跨周组合，选取时长范围（max-min）最小的组合
    3. 范围相同时优先平均时长更短（更接近"最短段"的语义）

    返回 {week: (t_seg, dr_seg, peak_times, peak_dr)}
    """
    week_candidates = {}
    for week in TARGET_WEEKS:
        candidates = _get_week_candidates(rabbit, week)
        if candidates:
            # 按时长排序，保留最短的 MAX_CANDIDATES_PER_WEEK 个
            candidates.sort(key=lambda c: c["duration"])
            if len(candidates) > MAX_CANDIDATES_PER_WEEK:
                candidates = candidates[:MAX_CANDIDATES_PER_WEEK]
            week_candidates[week] = candidates
            print(f"    {week}: {len(candidates)} 个候选段, "
                  f"时长 {candidates[0]['duration']:.2f}-"
                  f"{candidates[-1]['duration']:.2f}s")

    if not week_candidates:
        return {}

    weeks = list(week_candidates.keys())

    # 不足两周有数据：每周独立取最短段
    if len(weeks) < 2:
        w = weeks[0]
        c = week_candidates[w][0]  # 已按时长排序
        return {w: (c["t_seg"], c["dr_seg"], c["peak_times"], c["peak_dr"])}

    # 暴力搜索：找到跨周时长范围最小的组合
    best_combo = None
    best_score = (np.inf, np.inf)  # (duration_range, avg_duration)

    for combo in product(*(week_candidates[w] for w in weeks)):
        durs = [c["duration"] for c in combo]
        dur_range = max(durs) - min(durs)
        avg_dur = sum(durs) / len(durs)
        score = (dur_range, avg_dur)
        if score < best_score:
            best_score = score
            best_combo = dict(zip(weeks, combo))

    if best_combo is None:
        # 安全回退：每周独立取最短段
        result = {}
        for w, cands in week_candidates.items():
            c = cands[0]
            result[w] = (c["t_seg"], c["dr_seg"], c["peak_times"], c["peak_dr"])
        return result

    result = {}
    for week, cand in best_combo.items():
        result[week] = (cand["t_seg"], cand["dr_seg"],
                        cand["peak_times"], cand["peak_dr"])
    return result


# ============================================================
# 3. 绘图
# ============================================================
def plot_rabbit_standard_detail(rabbit: str):
    """为单只兔子绘制标准运动 2 子图大图。"""
    print(f"\n[{rabbit}] 提取数据...")

    # ── 子图 1 数据：三周 5 峰时段（时间跨度一致化） ──
    print("  枚举候选段...")
    segments = _select_consistent_segments(rabbit)
    for week in TARGET_WEEKS:
        if week in segments:
            t_seg = segments[week][0]
            dur = t_seg[-1] - t_seg[0]
            print(f"  {week}: 段长 {len(t_seg)} 点, "
                  f"时段 {t_seg[0]:.2f}-{t_seg[-1]:.2f}s, "
                  f"跨度 {dur:.2f}s, {len(segments[week][2])} 峰")
        else:
            print(f"  {week}: 未找到含 {N_PEAKS} 峰的段")

    # ── 子图 2 数据：标准运动 mean±std ──
    df_std = _extract_standard_motion_peaks(rabbit)
    print(f"  标准运动峰值点: {len(df_std)} 个, "
          f"覆盖 {df_std['week'].nunique() if not df_std.empty else 0} 周")

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle(f"Rabbit {rabbit} — 标准运动 ΔR/R₀",
                 fontsize=14, fontweight='bold')

    # ── 子图 1: 三周时段不重叠排列 ──
    ax = axes[0]
    x_offset = 0.0
    gap = 3.0  # 段间隔（秒）
    tick_positions = []
    tick_labels = []
    for week in TARGET_WEEKS:
        if week not in segments:
            continue
        t_seg, dr_seg, peak_times, peak_dr = segments[week]
        t_rel = t_seg - t_seg[0]  # 相对时间，每段从 0 开始
        seg_dur = float(t_rel[-1]) if len(t_rel) > 0 else 5.0
        color = WEEK_COLORS.get(week, "gray")
        ax.plot(t_rel + x_offset, dr_seg, color=color, lw=1.2, alpha=0.9, label=week)
        # 标注 5 个峰
        peak_t_rel = peak_times - t_seg[0]
        ax.scatter(peak_t_rel + x_offset, peak_dr, color=color, s=40, zorder=5,
                   edgecolors='white', linewidths=0.6)
        # 段间分隔线
        if x_offset > 0:
            ax.axvline(x_offset - gap / 2, color="gray", lw=0.5, ls=":",
                       alpha=0.5)
        # 记录该段中心位置用于 x 轴标签
        tick_positions.append(x_offset + seg_dur / 2)
        tick_labels.append(week)
        x_offset += seg_dur + gap

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("time", fontsize=11)
    ax.set_ylabel("ΔR/R₀", fontsize=11)
    ax.set_title("标准运动 w1/w3/w6 时段\n(5 峰 + 首峰前谷 + 末峰后谷)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # ── 子图 2: mean±std（与 optimized_v3 一致） ──
    ax = axes[1]
    if not df_std.empty:
        ms, ss, ws = [], [], []
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                ms.append(sub.mean())
                ss.append(sub.std())
                ws.append(int(w[1:]))
        if ms:
            ax.errorbar(ws, ms, yerr=ss, fmt='o-', color="#1E90FF",
                        capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week", fontsize=11)
    ax.set_ylabel("ΔR/R₀", fontsize=11)
    ax.set_title("Standard (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.2)

    fig.tight_layout()
    save_path = OUT_DIR / f"delta_r_r0_{rabbit}_standard_detail.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("030, 088 标准运动细节图 (5峰时段对齐 + mean±std)")
    print("=" * 60)
    for rabbit in ["030", "088"]:
        try:
            plot_rabbit_standard_detail(rabbit)
        except Exception as e:
            print(f"  [ERROR] {rabbit}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"完成! 图表已保存到: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
