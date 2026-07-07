# -*- coding: utf-8 -*-
"""
全部兔子 ΔR/R₀ 随周数变化趋势图。

每只兔子一张图，3×2 子图：
  (e) 左上：标准运动散点图 — 峰值点 ΔR/R₀，按周着色
  (f) 右上：标准运动统计图 — 每周均值+误差棒
  (g) 左中：自由运动散点图 — 活跃窗口内采样点 ΔR/R₀
  (h) 右中：自由运动统计图 — 每周均值+误差棒
  (i) 左下：自由运动窗口内峰值散点图 — 仅活跃窗口内的 Stage 2 峰值点
  (j) 右下：自由运动窗口内峰值统计图 — 每周均值+误差棒

ΔR/R₀ = (R - R₀) / R₀
R₀ 统一为该时刻的局部动态基线（P30滚动分位数）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from scipy.signal import find_peaks
from pathlib import Path

from .config import GROUPS, WEEKS, OUT_FIG, PARAMS, RABBIT2GROUP, ALL_RABBITS
from .preprocess import load_session, load_label, preprocess
from .peaks import (detect_peaks_stage1, detect_peaks_stage1_adaptive,
                    estimate_lag, assign_peaks_to_windows)
from .standard_motion import load_standard_session, _estimate_fs

# ============================================================
# 配色：W1-W6
# ============================================================
WEEK_COLORS = {
    "w1": "#FF8C00",  # 橙
    "w2": "#DC143C",  # 红
    "w3": "#228B22",  # 绿
    "w4": "#1E90FF",  # 蓝
    "w5": "#8B6914",  # 深黄
    "w6": "#9932CC",  # 紫
}

OUT_DIR = OUT_FIG / "delta_r_r0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 标准运动数据提取
# ============================================================
def _extract_standard_motion_peaks(rabbit: str, max_per_week: int = 15) -> pd.DataFrame:
    """提取标准运动中所有周的峰值 ΔR/R₀。

    寻峰约束：
    - 物理下限：最小峰间距 = 1.0s
    - 自适应阈值：2×MAD → 1.5×MAD → 1.0×MAD
    - 数量上限：检出数 ≤ 标注count，超出则时间分段+段内最大

    绘图采样：
    - 每周最多取 max_per_week 个峰，按时间均匀选取
    - 若峰数 < max_per_week，则全部保留

    返回 DataFrame: week, time_s, delta_r_r0
    """
    records = []
    for week in WEEKS:
        try:
            groups = load_standard_session(rabbit, week)
        except (FileNotFoundError, KeyError, Exception):
            continue

        week_peaks = []  # 收集本周所有组的峰

        for grp in groups:
            t = grp["t"]
            r = grp["r"]
            count = grp["count"]
            if len(t) < 20:
                continue
            if not np.isfinite(count):
                count = 0

            # P30 滚动基线（5s窗口，自适应：记录太短时自动缩小）
            fs = _estimate_fs(t)
            win_b = max(11, int(5.0 * fs))
            if win_b % 2 == 0:
                win_b += 1
            # 边界处理：点数不足以支撑完整窗口时，缩小至点数的 2/3
            if len(t) < win_b:
                win_b = max(5, (2 * len(t)) // 3)
                if win_b % 2 == 0:
                    win_b += 1
            baseline = (pd.Series(r)
                        .rolling(win_b, center=True, min_periods=max(3, win_b // 2))
                        .quantile(0.30)
                        .bfill().ffill().to_numpy())

            detrended = r - baseline

            # 寻峰（物理下限 + 自适应阈值 + 数量上限）
            mad = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
            if mad <= 0:
                mad = float(np.nanstd(detrended)) or 1.0
            min_dist_s = 1.0
            distance = max(1, int(min_dist_s * fs))

            # 自适应阈值衰减：默认 2×MAD，检出不足时逐级降低
            height = 2.0 * mad
            peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)
            if count > 0 and peaks_idx.size < count * 0.5:
                height = 1.5 * mad
                peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)
            if count > 0 and peaks_idx.size < count * 0.5:
                height = 1.0 * mad
                peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)

            # 数量上限：时间分段 + 段内取最大（保证时间分布均匀）
            if count > 0 and peaks_idx.size > int(count):
                n_sel = int(count)
                t_peaks = t[peaks_idx]
                seg_edges = np.linspace(t_peaks.min() - 0.01,
                                        t_peaks.max() + 0.01, n_sel + 1)
                selected = []
                used = set()
                for si in range(n_sel):
                    mask = (t_peaks >= seg_edges[si]) & (t_peaks < seg_edges[si + 1])
                    seg_idx = np.where(mask)[0]
                    seg_idx = [x for x in seg_idx if x not in used]
                    if seg_idx:
                        best = max(seg_idx, key=lambda x: detrended[peaks_idx[x]])
                        selected.append(best)
                        used.add(best)
                # 空段补充：从剩余未选峰中取幅值最大的
                if len(selected) < n_sel:
                    remaining = [i for i in range(len(peaks_idx)) if i not in used]
                    remaining.sort(key=lambda x: detrended[peaks_idx[x]], reverse=True)
                    for ri in remaining:
                        if len(selected) >= n_sel:
                            break
                        selected.append(ri)
                        used.add(ri)
                peaks_idx = peaks_idx[np.sort(selected)]

            if peaks_idx.size == 0:
                continue

            # 计算 ΔR/R₀
            bl_at_peak = baseline[peaks_idx]
            bl_safe = np.where(np.abs(bl_at_peak) > 1e-6, bl_at_peak, np.nan)
            delta_r_r0 = (r[peaks_idx] - bl_at_peak) / bl_safe

            for i in range(len(peaks_idx)):
                if np.isfinite(delta_r_r0[i]) and delta_r_r0[i] > 0:
                    week_peaks.append({
                        "week": week,
                        "time_s": float(t[peaks_idx[i]]),
                        "delta_r_r0": float(delta_r_r0[i]),
                    })

        # 每周最多取 max_per_week 个峰，按时间均匀选取
        if week_peaks:
            df_week = pd.DataFrame(week_peaks)
            if len(df_week) > max_per_week:
                # 时间分段选取：将时间轴等分为 max_per_week 段，每段取一个峰
                df_week = df_week.sort_values("time_s").reset_index(drop=True)
                t_arr = df_week["time_s"].to_numpy()
                seg_edges = np.linspace(t_arr.min() - 0.01,
                                        t_arr.max() + 0.01, max_per_week + 1)
                sel_idx = []
                for si in range(max_per_week):
                    mask = (t_arr >= seg_edges[si]) & (t_arr < seg_edges[si + 1])
                    candidates = np.where(mask)[0]
                    candidates = [c for c in candidates if c not in sel_idx]
                    if candidates:
                        # 段内取 ΔR/R₀ 最大的
                        best = max(candidates,
                                   key=lambda x: df_week.iloc[x]["delta_r_r0"])
                        sel_idx.append(best)
                # 空段补充：从剩余未选中取 ΔR/R₀ 最大的
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


# ============================================================
# 2. 自由运动数据提取
# ============================================================
def _extract_free_motion_points(rabbit: str, n_per_week: int = 30) -> pd.DataFrame:
    """提取自由运动活跃窗口内采样点的 ΔR/R₀。

    每周采样 n_per_week 个点，按时间均匀选取（时间分段 + 段内随机）。
    时间分段仅基于活跃窗口内的时间，静止段不参与分段。
    使用标准运动参考进行上限过滤：排除超过标准运动峰值的异常点。
    返回 DataFrame: week, time_s, delta_r_r0
    """
    from .preprocess import preprocess_raw
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

        # 获取标准运动参考（用于上限过滤）
        ref_profile = None
        try:
            ref_profile = get_reference_profile(rabbit, week)
        except Exception:
            pass

        # Stage 2 预处理（传入ref_profile以获得更好的清洗）
        pp = preprocess(t, r, ref_profile=ref_profile)
        t_uni = pp["t"]
        r_clean = pp["r_clean"]
        baseline = pp["baseline"]

        # 对齐
        pp_s1 = preprocess_raw(t, r)
        pk_s1 = detect_peaks_stage1(pp_s1)
        al = estimate_lag(pp_s1, pk_s1, label)
        lag = al["lag"]

        # 确定活跃窗口内的采样点
        from .peaks import _adaptive_pad
        pads = _adaptive_pad(label, PARAMS)
        s_arr = label["start"].to_numpy() + lag - pads
        e_arr = label["end"].to_numpy() + lag + pads

        # 找窗口内所有点的索引
        in_window = np.zeros(len(t_uni), dtype=bool)
        for s_i, e_i in zip(s_arr, e_arr):
            in_window |= (t_uni >= s_i) & (t_uni <= e_i)

        idx_in = np.where(in_window)[0]
        if idx_in.size < 5:
            continue

        # 计算窗口内所有点的 ΔR/R₀
        bl_in = baseline[idx_in]
        bl_safe = np.where(np.abs(bl_in) > 1e-6, bl_in, np.nan)
        delta_r_r0 = (r_clean[idx_in] - bl_in) / bl_safe

        # 上限过滤：使用标准运动的相对幅值作为上界
        # 标准运动下用力更大，其峰值是自由运动的上限
        # 允许 1.2× 容差（考虑测量误差）
        upper_limit = None
        if ref_profile is not None:
            ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
            if ref_rel_amp > 0:
                upper_limit = ref_rel_amp * 1.2

        # 过滤有效值
        valid = np.isfinite(delta_r_r0) & (delta_r_r0 > 0)
        if upper_limit is not None:
            valid = valid & (delta_r_r0 <= upper_limit)

        positive = valid
        n_positive = int(positive.sum())

        if n_positive >= n_per_week:
            # 时间分段 + 段内随机选取
            pool_idx = np.where(positive)[0]
            pool_times = t_uni[idx_in[pool_idx]]
            rng = np.random.default_rng(seed=hash((rabbit, week)) & 0xFFFF)

            # 将活跃窗口内的时间轴等分为 n_per_week 段
            seg_edges = np.linspace(pool_times.min() - 0.001,
                                    pool_times.max() + 0.001, n_per_week + 1)
            chosen = []
            used = set()
            for si in range(n_per_week):
                mask = (pool_times >= seg_edges[si]) & (pool_times < seg_edges[si + 1])
                seg_cands = np.where(mask)[0]
                seg_cands = [c for c in seg_cands if c not in used]
                if seg_cands:
                    pick = rng.choice(seg_cands)
                    chosen.append(pool_idx[pick])
                    used.add(pick)
            # 空段补充：从剩余未选点中随机补充
            if len(chosen) < n_per_week:
                remaining = [i for i in range(len(pool_idx)) if i not in used]
                if remaining:
                    need = n_per_week - len(chosen)
                    extra = rng.choice(remaining, size=min(need, len(remaining)),
                                       replace=False)
                    for ei in extra:
                        chosen.append(pool_idx[ei])
            chosen = np.array(chosen)
        elif n_positive > 0:
            chosen = np.where(positive)[0]
        else:
            # 没有正值，取所有有效值
            fallback = np.isfinite(delta_r_r0)
            if upper_limit is not None:
                fallback = fallback & (delta_r_r0 <= upper_limit)
            chosen = np.where(fallback)[0][:n_per_week] if fallback.any() else np.array([], dtype=int)

        for ci in chosen:
            records.append({
                "week": week,
                "time_s": float(t_uni[idx_in[ci]]),
                "delta_r_r0": float(delta_r_r0[ci]),
            })

    return pd.DataFrame(records)


# ============================================================
# 3. 自由运动活跃窗口内峰值点提取
# ============================================================
def _extract_free_motion_peaks_in_window(rabbit: str) -> pd.DataFrame:
    """提取自由运动中仅在活跃窗口内的生理学峰值点 ΔR/R₀。

    与 _extract_free_motion_points 的区别：
    - 该函数取窗口内所有采样点
    - 本函数仅取 Stage 2 检出的峰值点（已内置生理学峰过滤）

    处理流程：
    1. lag 对齐确定窗口位置
    2. 自适应 Stage 2 寻峰（检出不足时逐级降低阈值）
    3. 筛选落在活跃窗口内的峰
    4. 计算峰值处 ΔR/R₀ 并应用上限过滤

    返回 DataFrame: week, time_s, delta_r_r0
    """
    from .preprocess import preprocess_raw
    from .standard_motion import get_reference_profile
    from .peaks import (detect_peaks_stage2_adaptive, assign_peaks_to_windows)

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

        # 获取标准运动参考
        ref_profile = None
        try:
            ref_profile = get_reference_profile(rabbit, week)
        except Exception:
            pass

        # Stage 2 预处理
        pp = preprocess(t, r, ref_profile=ref_profile)
        t_uni = pp["t"]
        r_clean = pp["r_clean"]
        baseline = pp["baseline"]

        # lag 对齐（先算 lag，再做自适应检测）
        pp_s1 = preprocess_raw(t, r)
        pk_s1 = detect_peaks_stage1(pp_s1)
        al = estimate_lag(pp_s1, pk_s1, label)
        lag = al["lag"]

        # 自适应 Stage 2 寻峰（已内置生理学峰过滤 + 自动降阈值）
        pk = detect_peaks_stage2_adaptive(pp, label, lag, ref_profile=ref_profile)
        if pk["idx"].size == 0:
            continue

        # 筛选落在活跃窗口内的峰
        win_id = assign_peaks_to_windows(pk, label, lag)
        in_window = win_id >= 0
        if not in_window.any():
            continue

        peak_idx = pk["idx"][in_window]

        # 计算峰值处 ΔR/R₀
        bl_at_peak = baseline[peak_idx]
        bl_safe = np.where(np.abs(bl_at_peak) > 1e-6, bl_at_peak, np.nan)
        delta_r_r0 = (r_clean[peak_idx] - bl_at_peak) / bl_safe

        # 上限过滤：标准运动峰值 × 1.2
        upper_limit = None
        if ref_profile is not None:
            ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
            if ref_rel_amp > 0:
                upper_limit = ref_rel_amp * 1.2

        for i in range(len(peak_idx)):
            val = delta_r_r0[i]
            if not np.isfinite(val) or val <= 0:
                continue
            if upper_limit is not None and val > upper_limit:
                continue
            records.append({
                "week": week,
                "time_s": float(t_uni[peak_idx[i]]),
                "delta_r_r0": float(val),
            })

    return pd.DataFrame(records)


# ============================================================
# 4. 绘图
# ============================================================
def plot_delta_r_r0_per_rabbit(rabbit: str):
    """为单只兔子生成 ΔR/R₀ 趋势图（6子图，3×2布局）。

    行1: 标准运动（散点 + 均值±std）
    行2: 自由运动全部采样点（散点 + 均值±std）
    行3: 自由运动活跃窗口内生理学峰值点（散点 + 均值±std）
    """

    print(f"  [{rabbit}] 提取标准运动数据...")
    df_std = _extract_standard_motion_peaks(rabbit)
    print(f"    标准运动: {len(df_std)} 个峰值点")

    print(f"  [{rabbit}] 提取自由运动数据...")
    df_free = _extract_free_motion_points(rabbit, n_per_week=30)
    print(f"    自由运动: {len(df_free)} 个采样点")

    print(f"  [{rabbit}] 提取自由运动窗口内生理学峰值...")
    df_free_peaks = _extract_free_motion_peaks_in_window(rabbit)
    print(f"    自由运动生理学峰: {len(df_free_peaks)} 个峰值点")

    # 创建 3x2 子图
    fig, axes = plt.subplots(3, 2, figsize=(12, 13),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle(f"Rabbit {rabbit} — ΔR/R₀ 随周数变化", fontsize=14, fontweight='bold')

    # --- (e) 左上：标准运动散点图 ---
    ax = axes[0, 0]
    if not df_std.empty:
        # 按周排列，每周一个时间段
        x_offset = 0
        week_positions = {}
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            # 归一化到相对时间
            t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7,
                       label=w)
            week_positions[w] = x_offset + (t_rel.max() if t_rel.size else 0) / 2
            x_offset += (t_rel.max() if t_rel.size else 5) + 5
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard movement", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- (f) 右上：标准运动统计图 ---
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

    # --- (g) 左下：自由运动散点图 ---
    ax = axes[1, 0]
    if not df_free.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_free[df_free["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=15, alpha=0.6,
                       label=w)
            x_offset += (t_rel.max() if t_rel.size else 5) + 10
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- (h) 右中：自由运动统计图 ---
    ax = axes[1, 1]
    if not df_free.empty:
        means, stds, weeks_valid = [], [], []
        for w in WEEKS:
            sub = df_free[df_free["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                means.append(sub.mean())
                stds.append(sub.std())
                weeks_valid.append(int(w[1:]))
        if means:
            ax.errorbar(weeks_valid, means, yerr=stds,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.2)

    # --- (i) 左下：自由运动窗口内峰值散点图 ---
    ax = axes[2, 0]
    if not df_free_peaks.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_free_peaks[df_free_peaks["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7,
                       label=w)
            x_offset += (t_rel.max() if t_rel.size else 5) + 10
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement (physio peaks in active windows)", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- (j) 右下：自由运动窗口内峰值统计图 ---
    ax = axes[2, 1]
    if not df_free_peaks.empty:
        means, stds, weeks_valid = [], [], []
        for w in WEEKS:
            sub = df_free_peaks[df_free_peaks["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                means.append(sub.mean())
                stds.append(sub.std())
                weeks_valid.append(int(w[1:]))
        if means:
            ax.errorbar(weeks_valid, means, yerr=stds,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free physio-peaks (mean\u00b1std)", fontsize=11)
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.2)

    fig.tight_layout()
    save_path = OUT_DIR / f"delta_r_r0_{rabbit}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    已保存: {save_path}")
    return save_path


# ============================================================
# 5. 主入口
# ============================================================
def main():
    """为所有兔子生成 ΔR/R₀ 趋势图。"""
    from .config import ALL_RABBITS

    print("=" * 60)
    print("全部兔子 ΔR/R₀ 随周数变化趋势图")
    print("=" * 60)

    for rabbit in ALL_RABBITS:
        print(f"\n处理兔子 {rabbit} ({RABBIT2GROUP[rabbit]})...")
        try:
            plot_delta_r_r0_per_rabbit(rabbit)
        except Exception as e:
            print(f"    [ERROR] {rabbit}: {e}")

    print("\n" + "=" * 60)
    print(f"完成! 图表已保存到: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
