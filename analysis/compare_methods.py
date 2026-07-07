# -*- coding: utf-8 -*-
"""
对比分析：新方法 vs 旧方法的 Stage 2 窗口内数据点在信号上的分布。

为每只兔子（030/071/075/020）的每周（w1/w4/w6）生成一张对比图：
- 上子图：原始信号 + Stage 2 peaks + 旧方法窗口（蓝色底色）+ 旧方法选中点
- 下子图：原始信号 + Stage 2 peaks + 新方法窗口（绿色底色）+ 新方法选中点
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Songti SC', 'PingFang SC', 'Heiti SC',
                                       'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from .config import GROUPS, OUT_FIG, OUT_INTER, PARAMS
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .peaks import (detect_peaks_stage1, detect_peaks_stage2,
                    estimate_lag, _adaptive_pad, assign_peaks_to_windows)
from .standard_motion import get_reference_profile
from .experiment_window import estimate_lag_golden, GOLDEN_PEAKS

# ============================================================
OUT_DIR = OUT_FIG / "delta_r_r0" / "free_motion_detail_new_method"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RABBITS = ["030", "071", "075", "020"]
TARGET_WEEKS = ["w1", "w4", "w6"]

# 预加载新方法 lag
COMPARISON_CSV = OUT_INTER / "experiment_golden_window" / "comparison_results.csv"


def _load_new_lags() -> dict:
    df = pd.read_csv(COMPARISON_CSV)
    return {(str(row["rabbit"]).zfill(3), row["week"]): row["new_lag"]
            for _, row in df.iterrows()}


def plot_signal_comparison(rabbit: str, week: str, new_lag: float):
    """
    生成单个 session 的信号级对比图。
    上：旧方法窗口 + Stage 2 选中点
    下：新方法窗口 + Stage 2 选中点
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
    detrended = pp["detrended"]

    # Stage 2 寻峰
    pk = detect_peaks_stage2(pp, ref_profile=ref_profile)

    # 旧方法 lag
    pp_s1 = preprocess_raw(t, r)
    pk_s1 = detect_peaks_stage1(pp_s1)
    al_old = estimate_lag(pp_s1, pk_s1, label)
    old_lag = al_old["lag"]

    # 计算两种方法的窗口
    pads = _adaptive_pad(label, PARAMS)
    s_old = label["start"].to_numpy() + old_lag - pads
    e_old = label["end"].to_numpy() + old_lag + pads
    s_new = label["start"].to_numpy() + new_lag - pads
    e_new = label["end"].to_numpy() + new_lag + pads

    # 计算 ΔR/R₀ for Stage 2 peaks
    bl_safe = np.where(np.abs(baseline) > 1e-6, baseline, np.nan)
    delta_r_r0_all = (r_clean - baseline) / bl_safe

    # Stage 2 峰在旧窗口内的归属
    win_old = assign_peaks_to_windows(pk, label, old_lag)
    in_old = win_old >= 0
    # Stage 2 峰在新窗口内的归属
    # 手动计算（因为 new_lag 不走标准流程）
    in_new = np.zeros(pk["idx"].size, dtype=bool)
    for j in range(len(s_new)):
        in_new |= (pk["t_peak"] >= s_new[j]) & (pk["t_peak"] <= e_new[j])

    # --- 绘图 ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for ax_idx, (ax, s_arr, e_arr, in_win, lag_val, method, color) in enumerate([
        (axes[0], s_old, e_old, in_old, old_lag,
         f"旧方法 (physio-hit-rate, lag={old_lag:+.1f}s)", "#1f77b4"),
        (axes[1], s_new, e_new, in_new, new_lag,
         f"新方法 (黄金峰覆盖, lag={new_lag:+.1f}s)", "#2ca02c"),
    ]):
        # 信号
        ax.plot(t_uni, detrended, color="#bbb", lw=0.5, zorder=1)

        # 窗口底色
        for s_i, e_i in zip(s_arr, e_arr):
            ax.axvspan(s_i, e_i, alpha=0.15, color=color, zorder=0)

        # Stage 2 peaks
        if pk["idx"].size:
            # 窗口外峰（灰色）
            out_win = ~in_win
            if out_win.any():
                ax.scatter(pk["t_peak"][out_win], detrended[pk["idx"][out_win]],
                           c="#ccc", s=12, zorder=3, marker="^", alpha=0.5,
                           label=f"窗口外峰 (n={out_win.sum()})")
            # 窗口内峰（强调色）
            if in_win.any():
                ax.scatter(pk["t_peak"][in_win], detrended[pk["idx"][in_win]],
                           c=color, s=25, zorder=4, marker="^",
                           label=f"窗口内峰 (n={in_win.sum()})")

        # 统计信息
        n_in = int(in_win.sum()) if pk["idx"].size else 0
        n_total = int(pk["idx"].size) if pk["idx"].size else 0

        # 窗口内峰的 ΔR/R₀ 统计
        if n_in > 0:
            dr_in = delta_r_r0_all[pk["idx"][in_win]]
            dr_valid = dr_in[np.isfinite(dr_in) & (dr_in > 0)]
            mean_dr = float(dr_valid.mean()) if dr_valid.size > 0 else 0
            info_text = (f"窗口内峰: {n_in}/{n_total}  "
                         f"ΔR/R₀ mean={mean_dr:.4f}")
        else:
            info_text = f"窗口内峰: 0/{n_total}"

        ax.text(0.02, 0.95, info_text,
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_ylabel("ΔR (detrended)")
        ax.set_title(f"{rabbit} {week} — {method}")
        ax.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_path = OUT_DIR / f"compare_signal_{rabbit}_{week}.png"
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def run_comparison():
    """为所有目标 session 生成信号级对比图 + 数值统计。"""
    new_lags = _load_new_lags()
    stats = []

    print("=" * 70)
    print("信号级对比：新方法 vs 旧方法窗口内 Stage 2 数据点分布")
    print("=" * 70)

    for rabbit in RABBITS:
        for week in TARGET_WEEKS:
            lag = new_lags.get((rabbit, week))
            if lag is None:
                continue

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
            pk = detect_peaks_stage2(pp, ref_profile=ref_profile)
            baseline = pp["baseline"]
            r_clean = pp["r_clean"]

            pp_s1 = preprocess_raw(t, r)
            pk_s1 = detect_peaks_stage1(pp_s1)
            al_old = estimate_lag(pp_s1, pk_s1, label)
            old_lag = al_old["lag"]

            pads = _adaptive_pad(label, PARAMS)
            # 旧方法
            s_old = label["start"].to_numpy() + old_lag - pads
            e_old = label["end"].to_numpy() + old_lag + pads
            in_old = np.zeros(pk["idx"].size, dtype=bool)
            for j in range(len(s_old)):
                in_old |= (pk["t_peak"] >= s_old[j]) & (pk["t_peak"] <= e_old[j])
            # 新方法
            s_new = label["start"].to_numpy() + lag - pads
            e_new = label["end"].to_numpy() + lag + pads
            in_new = np.zeros(pk["idx"].size, dtype=bool)
            for j in range(len(s_new)):
                in_new |= (pk["t_peak"] >= s_new[j]) & (pk["t_peak"] <= e_new[j])

            # ΔR/R₀ 统计
            bl_safe = np.where(np.abs(baseline) > 1e-6, baseline, np.nan)
            dr_all = (r_clean - baseline) / bl_safe

            def _dr_stats(mask):
                if not mask.any():
                    return 0, 0.0, 0.0
                dr = dr_all[pk["idx"][mask]]
                valid = dr[np.isfinite(dr) & (dr > 0)]
                return int(valid.size), float(valid.mean()) if valid.size else 0.0, \
                       float(valid.std()) if valid.size >= 2 else 0.0

            n_old, mean_old, std_old = _dr_stats(in_old)
            n_new, mean_new, std_new = _dr_stats(in_new)

            stats.append({
                "rabbit": rabbit, "week": week,
                "old_lag": old_lag, "new_lag": lag,
                "total_s2_peaks": int(pk["idx"].size),
                "old_in_window": int(in_old.sum()),
                "new_in_window": int(in_new.sum()),
                "old_dr_n": n_old, "old_dr_mean": mean_old, "old_dr_std": std_old,
                "new_dr_n": n_new, "new_dr_mean": mean_new, "new_dr_std": std_new,
            })

            # 生成对比图
            plot_signal_comparison(rabbit, week, lag)
            print(f"[done] {rabbit} {week}  "
                  f"旧: {int(in_old.sum())}/{pk['idx'].size} peaks (ΔR/R₀={mean_old:.4f})  "
                  f"新: {int(in_new.sum())}/{pk['idx'].size} peaks (ΔR/R₀={mean_new:.4f})")

    # 保存统计表
    df = pd.DataFrame(stats)
    df.to_csv(OUT_DIR / "method_comparison_stats.csv", index=False)

    # 打印汇总
    print("\n" + "=" * 70)
    print(f"{'rabbit':<6} {'week':<4} {'S2_peaks':<9} "
          f"{'旧窗内':>6} {'旧mean':>8} {'旧std':>7}  "
          f"{'新窗内':>6} {'新mean':>8} {'新std':>7}  {'Δmean':>7}")
    print("-" * 70)
    for _, row in df.iterrows():
        d_mean = row["new_dr_mean"] - row["old_dr_mean"]
        print(f"{row['rabbit']:<6} {row['week']:<4} {row['total_s2_peaks']:<9} "
              f"{row['old_in_window']:>6} {row['old_dr_mean']:>8.4f} {row['old_dr_std']:>7.4f}  "
              f"{row['new_in_window']:>6} {row['new_dr_mean']:>8.4f} {row['new_dr_std']:>7.4f}  "
              f"{d_mean:>+7.4f}")

    print(f"\n对比图和统计表已保存到: {OUT_DIR}")
    return df


if __name__ == "__main__":
    run_comparison()
