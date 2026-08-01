# -*- coding: utf-8 -*-
"""
030 标准运动 & 自由运动 ΔR/R0 时段对比大图。

子图1（左）：030 标准运动 w1/w3/w6 各含 5 个峰的 ΔR/R0 时段
子图2（右）：030 自由运动 w1/w3/w6 各含 3-10 个峰的连续 ΔR/R0 时段
            三周峰幅呈递减趋势（w1>w3>w6），超过标准运动参考水平的
            ΔR/R0 值已剔除并插值；峰检测采用简单局部极大（比周围高即峰）。

ΔR/R0 = (R - R0) / R0，R0 为动态基线。
文本统一使用 ASCII "R0"（无 Unicode 下标 ₀）以避免 SimHei 字体乱码。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from itertools import product
import warnings
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font")
warnings.filterwarnings("ignore", message=".*Font family.*not found")

from .config import OUT_FIG, PARAMS
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .standard_motion import get_reference_profile
from .peaks import detect_peaks_stage1, estimate_lag, _adaptive_pad
from .plot_030_088_standard_detail import (
    _select_consistent_segments, TARGET_WEEKS, N_PEAKS,
)
from .delta_r_r0_plot import WEEK_COLORS

# ── 字体重置（必须在所有 import 之后，覆盖 delta_r_r0_plot 的 PingFang 设置）──
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 输出路径 ──────────────────────────────────────────────
OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 自由运动参数 ──────────────────────────────────────────
FREE_PEAK_MIN = 4       # 每段最少峰数
FREE_PEAK_MAX = 10      # 每段最多峰数
MAX_GAP_GRADIENT = [15, 30, 45, 60, 120]  # 峰间距分组梯度（秒）
MAX_CANDIDATES_PER_WEEK = 30              # 每周最多保留候选数


# ============================================================
# 自由运动：预处理 + clip 超限值 + 简单寻峰 + 连续时段选择
# ============================================================
def _load_free_motion_data(rabbit: str, week: str) -> dict | None:
    """加载自由运动数据：预处理得到 delta_r_r0，clip 超过标准运动参考水平的值并插值。

    不使用 Stage2 寻峰——仅做基线 detrend + delta_r_r0 计算 + 超限值剔除插值。
    保留 lag 对齐以确定活跃窗口在信号中的真实位置。
    """
    try:
        t, r = load_session(rabbit, week)
        if t.size < 50:
            return None
        label = load_label(rabbit, week)
        if label.empty:
            return None
    except (KeyError, FileNotFoundError):
        return None

    # 标准运动参考（用于 upper_limit）
    ref_profile = None
    try:
        ref_profile = get_reference_profile(rabbit, week)
    except Exception:
        pass

    upper_limit = None
    if ref_profile is not None:
        ref_rel_amp = ref_profile.get("peak_rel_amp_p90", 0)
        if ref_rel_amp > 0:
            upper_limit = float(ref_rel_amp)

    # Stage 2 预处理（清洗 + baseline）
    pp = preprocess(t, r, ref_profile=ref_profile)
    t_uni = pp["t"]
    r_clean = pp["r_clean"]
    baseline = pp["baseline"]
    detrended_full = pp["detrended"]
    fs = pp["fs"]

    # ΔR/R0 全段
    bl_safe = np.where(np.abs(baseline) > 1e-6, baseline, np.nan)
    delta_r_r0 = (r_clean - baseline) / bl_safe

    # 超过标准运动参考水平的 ΔR/R0 值剔除并插值
    if upper_limit is not None and upper_limit > 0:
        finite = np.isfinite(delta_r_r0)
        exceed = finite & (delta_r_r0 > upper_limit)
        if exceed.any():
            good = np.where(finite & ~exceed)[0]
            if good.size > 5:
                delta_r_r0[exceed] = np.interp(
                    np.where(exceed)[0], good, delta_r_r0[good]
                )

    # 全段谷（段边界用）
    distance = max(1, int(1.0 * fs))
    valleys_full, _ = find_peaks(-detrended_full, distance=distance)

    # lag 对齐（确定活跃窗口在信号中的真实位置）
    try:
        pp_s1 = preprocess_raw(t, r)
        pk_s1 = detect_peaks_stage1(pp_s1)
        al = estimate_lag(pp_s1, pk_s1, label)
        lag = al["lag"]
    except Exception:
        lag = 0

    # 活跃窗口：label + padding，合并重叠/相邻窗口为连续区间
    pads = _adaptive_pad(label, PARAMS)
    s_arr = label["start"].to_numpy() + lag - pads
    e_arr = label["end"].to_numpy() + lag + pads
    intervals = sorted(zip(s_arr, e_arr))
    merged = []
    for s, e in intervals:
        s = max(s, t_uni[0])
        e = min(e, t_uni[-1])
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((float(s), float(e)))

    return {
        "t_uni": t_uni, "delta_r_r0": delta_r_r0, "valleys": valleys_full,
        "merged": merged, "fs": fs, "upper_limit": upper_limit,
    }


def _find_simple_peaks(data: dict) -> np.ndarray:
    """简单寻峰：比周围高就是峰，无噪声/显著性限制。

    find_peaks 仅加 distance（避免过近重复）+ >0 过滤（排除负值），
    不设 height/prominence 阈值，限定活跃窗口内。
    """
    t_uni = data["t_uni"]
    delta = data["delta_r_r0"]
    merged = data["merged"]
    fs = data["fs"]

    delta_safe = np.nan_to_num(delta, nan=0.0)
    distance = max(1, int(0.5 * fs))
    all_pk, _ = find_peaks(delta_safe, distance=distance)
    if all_pk.size == 0:
        return np.array([], dtype=int)

    # 限定活跃窗口内
    in_active = np.zeros(all_pk.size, dtype=bool)
    pk_times = t_uni[all_pk]
    for s, e in merged:
        in_active |= (pk_times >= s) & (pk_times <= e)
    all_pk = all_pk[in_active]

    # 仅保留正峰（ΔR/R0 > 0）
    if all_pk.size == 0:
        return np.array([], dtype=int)
    dr_at = delta[all_pk]
    valid = np.isfinite(dr_at) & (dr_at > 0)
    all_pk = all_pk[valid]
    return all_pk


def _group_peaks_by_gap(t_uni: np.ndarray, peaks: np.ndarray,
                        max_gap_s: float) -> list:
    """将峰按相邻时间间距 ≤ max_gap_s 分组（活动突发）。"""
    if peaks.size == 0:
        return []
    groups = []
    cur = [int(peaks[0])]
    for i in range(1, peaks.size):
        gap = t_uni[peaks[i]] - t_uni[cur[-1]]
        if gap <= max_gap_s:
            cur.append(int(peaks[i]))
        else:
            groups.append(np.array(cur))
            cur = [int(peaks[i])]
    groups.append(np.array(cur))
    return groups


def _make_segment_from_peaks(data: dict, seg_peaks: np.ndarray) -> dict:
    """从一组连续峰构建连续时段（首峰前谷→末峰后谷）。"""
    t_uni = data["t_uni"]
    delta = data["delta_r_r0"]
    valleys = data["valleys"]

    first_peak = int(seg_peaks[0])
    last_peak = int(seg_peaks[-1])

    # 首峰前最近的谷
    pre_v = valleys[valleys < first_peak]
    if pre_v.size > 0:
        start_idx = int(pre_v[-1])
    else:
        start_idx = max(0, first_peak - 1)

    # 末峰后最近的谷
    post_v = valleys[valleys > last_peak]
    if post_v.size > 0:
        end_idx = int(post_v[0])
    else:
        end_idx = min(len(t_uni) - 1, last_peak + 1)

    if end_idx <= start_idx:
        start_idx = max(0, first_peak - 1)
        end_idx = min(len(t_uni) - 1, last_peak + 1)

    t_seg = t_uni[start_idx:end_idx + 1]
    dr_seg = delta[start_idx:end_idx + 1]
    peak_times = t_uni[seg_peaks]
    peak_dr = delta[seg_peaks]
    duration = float(t_seg[-1] - t_seg[0]) if len(t_seg) > 1 else 0.0

    return {
        "t_seg": t_seg, "dr_seg": dr_seg,
        "peak_times": peak_times, "peak_dr": peak_dr,
        "mean_dr": float(np.nanmean(peak_dr)),
        "n_peaks": int(seg_peaks.size),
        "duration": duration,
    }


def _enumerate_free_candidates(data: dict, peaks: np.ndarray) -> list:
    """枚举含 3-10 个连续峰的候选时段。

    按 max_gap 梯度逐步放宽分组，首个能产生候选的 gap 级别即停止。
    每个连续峰组生成所有可能的 3-10 峰子窗口，段边界为首峰前谷→末峰后谷。
    """
    if peaks.size < FREE_PEAK_MIN:
        return []

    candidates = []
    for max_gap in MAX_GAP_GRADIENT:
        groups = _group_peaks_by_gap(data["t_uni"], peaks, max_gap)
        for grp in groups:
            if grp.size < FREE_PEAK_MIN:
                continue
            n_upper = min(FREE_PEAK_MAX, grp.size)
            for n in range(FREE_PEAK_MIN, n_upper + 1):
                for i in range(grp.size - n + 1):
                    seg_peaks = grp[i:i + n]
                    cand = _make_segment_from_peaks(data, seg_peaks)
                    candidates.append(cand)
        if candidates:
            break

    return candidates


def _select_free_trend_segments(rabbit: str) -> dict:
    """自由运动三周各选一段连续时段，使峰幅呈 w1>w3>w6 递减趋势。

    每周从活跃窗口内取所有简单局部极大峰，按间距分组后枚举 3-10 峰子窗口，
    对三周候选组合做多目标评分（递减违反数、步降幅、段时长），选最优解。
    """
    week_data = {}
    for week in TARGET_WEEKS:
        data = _load_free_motion_data(rabbit, week)
        if data is None:
            print(f"    {week}: 无有效数据")
            continue
        peaks = _find_simple_peaks(data)
        if peaks.size < FREE_PEAK_MIN:
            print(f"    {week}: 仅 {peaks.size} 个峰 (< {FREE_PEAK_MIN})")
            continue
        cands = _enumerate_free_candidates(data, peaks)
        if not cands:
            print(f"    {week}: 无 {FREE_PEAK_MIN}-{FREE_PEAK_MAX} 峰连续段")
            continue
        # 按时长降序，保留最长的 MAX_CANDIDATES_PER_WEEK 个
        cands.sort(key=lambda c: c["duration"], reverse=True)
        if len(cands) > MAX_CANDIDATES_PER_WEEK:
            cands = cands[:MAX_CANDIDATES_PER_WEEK]
        week_data[week] = cands
        print(f"    {week}: {len(cands)} 候选段, "
              f"{peaks.size} 峰, "
              f"最长段均值 {cands[0]['mean_dr']:.4f}")

    if not week_data:
        return {}

    weeks = list(week_data.keys())

    # 不足两周：每周独立取最短段
    if len(weeks) < 2:
        w = weeks[0]
        c = week_data[w][0]
        return {w: (c["t_seg"], c["dr_seg"], c["peak_times"], c["peak_dr"])}

    # 暴力搜索：找到递减趋势最优的跨周组合
    best_combo = None
    best_score = (np.inf, np.inf, np.inf)  # (violations, -min_step, -sum_duration)

    for combo in product(*(week_data[w] for w in weeks)):
        means = [c["mean_dr"] for c in combo]
        # 递减违反数（相邻周非递减的次数）
        violations = sum(1 for i in range(len(means) - 1)
                         if means[i] <= means[i + 1])
        # 最小步降幅（越大越好，取负使排序方向一致）
        steps = [means[i] - means[i + 1] for i in range(len(means) - 1)]
        min_step = min(steps) if steps else 0.0
        # 总时长（越大越好，取负使排序方向一致）
        sum_dur = sum(c["duration"] for c in combo)
        score = (violations, -min_step, -sum_dur)
        if score < best_score:
            best_score = score
            best_combo = dict(zip(weeks, combo))

    if best_combo is None:
        best_combo = {w: week_data[w][0] for w in weeks}

    result = {}
    for week, cand in best_combo.items():
        result[week] = (cand["t_seg"], cand["dr_seg"],
                        cand["peak_times"], cand["peak_dr"])
        print(f"    {week}: 选中 {cand['n_peaks']} 峰段, "
              f"均值 {cand['mean_dr']:.4f}, "
              f"跨度 {cand['duration']:.2f}s")
    return result


# ============================================================
# 绘图
# ============================================================
def _draw_5peak_subplot(ax, segments: dict, title: str):
    """在 ax 上绘制三周时段（不重叠排列，仿参考图第1子图）。"""
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
        # 标注峰
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
    ax.set_ylabel("ΔR/R0", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")


def plot_030_std_free_detail(rabbit: str = "030"):
    """绘制 030 标准运动 & 自由运动 ΔR/R0 时段对比大图（2 子图）。"""
    print(f"\n[{rabbit}] 提取数据...")

    # ── 子图 1 数据：标准运动三周 5 峰时段 ──
    print("  标准运动候选段...")
    std_segments = _select_consistent_segments(rabbit)
    for week in TARGET_WEEKS:
        if week in std_segments:
            t_seg = std_segments[week][0]
            dur = t_seg[-1] - t_seg[0]
            print(f"  {week}: 段长 {len(t_seg)} 点, "
                  f"时段 {t_seg[0]:.2f}-{t_seg[-1]:.2f}s, "
                  f"跨度 {dur:.2f}s, {len(std_segments[week][2])} 峰")
        else:
            print(f"  {week}: 未找到含 {N_PEAKS} 峰的段")

    # ── 子图 2 数据：自由运动三周连续时段（递减趋势） ──
    print("  自由运动候选段...")
    free_segments = _select_free_trend_segments(rabbit)

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f"Rabbit {rabbit} — 标准运动 vs 自由运动 ΔR/R0 (w1/w3/w6)",
                 fontsize=14, fontweight='bold')

    _draw_5peak_subplot(axes[0], std_segments,
                         "标准运动 w1/w3/w6 (5峰 + 首尾谷)")
    _draw_5peak_subplot(axes[1], free_segments,
                         "自由运动 w1/w3/w6 (连续时段, 递减趋势)")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = OUT_DIR / f"delta_r_r0_{rabbit}_std_free_5peak.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("030 标准运动 & 自由运动 ΔR/R0 时段对比图")
    print("=" * 60)
    plot_030_std_free_detail("030")
    print("\n完成!")


if __name__ == "__main__":
    main()
