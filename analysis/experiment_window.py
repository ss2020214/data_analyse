# -*- coding: utf-8 -*-
"""
实验：以人工挑选峰覆盖率为核心目标的窗口偏移优化。

与原流程的区别：
- 原方法：physio-hit-rate（生理学峰加权命中率），搜索范围 ±2s
- 新方法：最大化人工标注的"黄金峰"落入窗口的比率，搜索范围 ±5s

不修改原有 analysis 流程，独立验证。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .config import GROUPS, OUT_INTER, PARAMS, RABBIT2GROUP, WEEKS
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .peaks import (detect_peaks_stage1, detect_peaks_stage1_adaptive,
                    detect_peaks_stage2, estimate_lag, assign_peaks_to_windows,
                    _adaptive_pad)
from .standard_motion import get_reference_profile
from .plot import _robust_ylim

# ============================================================
# 人工挑选的黄金峰（Stage 1 编号，1-based）
# ============================================================
GOLDEN_PEAKS = {
    ("030", "w1"): [13, 14, 15, 16, 18, 25, 26, 27, 28, 41, 42, 43,
                    61, 62, 65, 67, 72],
    ("030", "w4"): [6, 10, 11, 12, 19, 20, 21, 23, 26, 31, 33, 57, 58],
    ("030", "w6"): [8, 30, 32, 33, 56, 57, 78, 83, 84, 85, 94, 98,
                    106, 107, 108, 109, 110, 192],
    ("071", "w1"): [17, 18, 21, 49, 89, 104, 116, 136, 145, 147],
    ("071", "w4"): [1, 2, 3, 4, 10, 18, 23, 25, 32, 34, 35, 36, 37,
                    42, 43, 50, 51, 52, 53, 54, 55, 56],
    ("071", "w6"): [1, 34, 43, 45, 56, 57, 58, 66],
    ("075", "w1"): [27, 28, 43, 44, 58, 60, 66, 68, 72, 88],
    ("075", "w4"): [17, 21, 29, 30, 31, 48, 49, 50, 66, 67],
    ("075", "w6"): [7, 8, 15, 37, 48, 65, 73],
    ("020", "w1"): [8, 12, 21, 25, 26, 27, 30, 31, 33, 36,
                    42, 43, 44, 61, 66, 71],
    ("020", "w4"): [15, 23, 33, 45, 56, 72, 73, 82, 89, 90],
    ("020", "w6"): [17, 23, 27, 31, 45, 47, 54, 59, 62, 72, 73, 77,
                    88, 94, 105, 106, 107, 113, 114, 120, 127, 133],
}

# 输出目录
OUT_EXP = OUT_INTER / "experiment_golden_window"
OUT_EXP.mkdir(parents=True, exist_ok=True)


# ============================================================
# 核心算法：以黄金峰覆盖率为目标的 lag 估计
# ============================================================

def estimate_lag_golden(peaks: dict, label: pd.DataFrame,
                        golden_indices: list,
                        params=None) -> dict:
    """
    窗口偏移优化：最大化人工挑选的黄金峰落入标注窗口的比率。

    参数
    ----
    peaks : Stage 1 寻峰结果
    label : 标注窗口 DataFrame
    golden_indices : 黄金峰编号列表（1-based）
    params : 参数字典（可覆盖搜索范围）

    返回
    ----
    dict: lag, golden_hit_rate, golden_hits, golden_total, lags, scores, ...
    """
    if params is None:
        params = dict(PARAMS)

    # 搜索范围扩展到 ±5s
    lag_min = params.get("exp_lag_search_min_s", -5.0)
    lag_max = params.get("exp_lag_search_max_s", 5.0)
    lag_step = params.get("lag_search_step_s", 0.1)

    if label.empty or peaks["idx"].size == 0:
        return {"lag": 0.0, "golden_hit_rate": 0.0,
                "golden_hits": 0, "golden_total": 0,
                "n_peaks": int(peaks["idx"].size)}

    t_peak = peaks["t_peak"]

    # 筛选有效的黄金峰索引（1-based → 0-based，且不超范围）
    valid_golden = [i - 1 for i in golden_indices
                    if 1 <= i <= len(t_peak)]
    if not valid_golden:
        return {"lag": 0.0, "golden_hit_rate": 0.0,
                "golden_hits": 0, "golden_total": len(golden_indices),
                "n_peaks": int(peaks["idx"].size)}

    t_golden = t_peak[valid_golden]  # 黄金峰的时间坐标
    golden_total = len(t_golden)

    # 自适应 pad
    pads = _adaptive_pad(label, params)
    s_arr = label["start"].to_numpy()
    e_arr = label["end"].to_numpy()

    lags = np.arange(lag_min, lag_max + 1e-9, lag_step)
    scores = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        s = s_arr + lag - pads
        e = e_arr + lag + pads
        # 计算黄金峰有多少落入某个窗口
        in_any_window = np.zeros(golden_total, dtype=bool)
        for j in range(len(s)):
            in_any_window |= (t_golden >= s[j]) & (t_golden <= e[j])
        scores[i] = float(in_any_window.sum()) / golden_total

    best_score = float(scores.max())
    # 同分时选 |lag| 最小的
    cand = np.where(scores >= best_score - 1e-9)[0]
    j = cand[np.argmin(np.abs(lags[cand]))]

    # 计算最优 lag 下的详细命中情况
    best_lag = float(lags[j])
    s_best = s_arr + best_lag - pads
    e_best = e_arr + best_lag + pads
    golden_in_window = np.zeros(golden_total, dtype=bool)
    golden_window_id = np.full(golden_total, -1, dtype=int)
    for jj in range(len(s_best)):
        mask = (t_golden >= s_best[jj]) & (t_golden <= e_best[jj])
        golden_in_window |= mask
        golden_window_id[mask & (golden_window_id < 0)] = jj

    return {
        "lag": best_lag,
        "golden_hit_rate": float(golden_in_window.sum()) / golden_total,
        "golden_hits": int(golden_in_window.sum()),
        "golden_total": golden_total,
        "golden_in_window": golden_in_window,
        "golden_window_id": golden_window_id,
        "golden_indices_0based": valid_golden,
        "lags": lags,
        "scores": scores,
        "n_peaks": int(peaks["idx"].size),
    }


# ============================================================
# 对比图：新方法 vs 旧方法
# ============================================================

def plot_comparison(rabbit, week, pp_s1, pk_s1, label,
                    al_old, al_new, golden_indices, save_path):
    """
    生成对比图：
    - 上：Stage 1 信号 + 旧方法窗口 + 峰编号（黄金峰高亮）
    - 下：Stage 1 信号 + 新方法窗口 + 峰编号（黄金峰高亮）
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    t1 = pp_s1["t"]
    d1 = pp_s1["detrended"]

    valid_golden_0 = [i - 1 for i in golden_indices
                      if 1 <= i <= len(pk_s1["t_peak"])]
    is_golden = np.zeros(pk_s1["idx"].size, dtype=bool)
    for g in valid_golden_0:
        is_golden[g] = True

    for ax_idx, (ax, al, method_name) in enumerate(zip(
            axes,
            [al_old, al_new],
            ["旧方法 (physio-hit-rate, ±2s)", "新方法 (黄金峰覆盖率, ±5s)"])):

        lag = al["lag"]
        ax.plot(t1, d1, color="#888", lw=0.5)

        # 绘制窗口
        if not label.empty:
            pads = _adaptive_pad(label, PARAMS)
            s_arr = label["start"].to_numpy() + lag - pads
            e_arr = label["end"].to_numpy() + lag + pads
            for s_i, e_i in zip(s_arr, e_arr):
                ax.axvspan(s_i, e_i, alpha=0.15, color="#1f77b4")

        # 绘制峰：普通峰(灰) + 黄金峰(红)
        if pk_s1["idx"].size:
            # 非黄金峰
            non_golden = ~is_golden
            if non_golden.any():
                ax.scatter(pk_s1["t_peak"][non_golden],
                           d1[pk_s1["idx"][non_golden]],
                           c="#aaa", s=10, zorder=3, marker="v", alpha=0.5)
            # 黄金峰
            if is_golden.any():
                ax.scatter(pk_s1["t_peak"][is_golden],
                           d1[pk_s1["idx"][is_golden]],
                           c="#e63946", s=30, zorder=5, marker="v",
                           label=f"黄金峰 (n={is_golden.sum()})")

            # 标注黄金峰编号
            for g_idx in valid_golden_0:
                tp = pk_s1["t_peak"][g_idx]
                amp = d1[pk_s1["idx"][g_idx]]
                ax.annotate(str(g_idx + 1), xy=(tp, amp),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=6, ha="center", color="#e63946",
                            fontweight="bold")

        # 统计黄金峰命中
        if pk_s1["idx"].size and not label.empty:
            t_golden = pk_s1["t_peak"][valid_golden_0]
            pads = _adaptive_pad(label, PARAMS)
            s_arr = label["start"].to_numpy() + lag - pads
            e_arr = label["end"].to_numpy() + lag + pads
            in_win = np.zeros(len(valid_golden_0), dtype=bool)
            for jj in range(len(s_arr)):
                in_win |= (t_golden >= s_arr[jj]) & (t_golden <= e_arr[jj])
            n_hit = int(in_win.sum())
            n_total = len(valid_golden_0)
        else:
            n_hit, n_total = 0, 0

        ylim = _robust_ylim(d1, d1[pk_s1["idx"]] if pk_s1["idx"].size else None)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_ylabel("ΔR (Stage 1)")
        ax.set_title(f"{rabbit} {week} — {method_name}  |  "
                     f"lag={lag:+.1f}s  "
                     f"黄金峰命中={n_hit}/{n_total} "
                     f"({n_hit/max(n_total,1)*100:.0f}%)")
        ax.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================

def run_experiment():
    """运行实验：对所有有黄金峰标注的 session 进行新旧方法对比。"""
    results = []

    for (rabbit, week), golden_ids in sorted(GOLDEN_PEAKS.items()):
        try:
            t, r = load_session(rabbit, week)
        except KeyError:
            print(f"[skip] {rabbit} {week} — 数据不存在")
            continue
        if t.size < 50:
            continue
        label = load_label(rabbit, week)

        ref_profile = None
        try:
            ref_profile = get_reference_profile(rabbit, week)
        except Exception:
            pass

        # Stage 1 寻峰（与原流程一致）
        pp_s1 = preprocess_raw(t, r)
        pk_s1 = detect_peaks_stage1(pp_s1)

        # === 旧方法：physio-hit-rate, ±2s ===
        al_old = estimate_lag(pp_s1, pk_s1, label)

        # === 新方法：黄金峰覆盖率, ±5s ===
        al_new = estimate_lag_golden(pk_s1, label, golden_ids)

        # 计算旧方法对黄金峰的覆盖情况
        valid_golden_0 = [i - 1 for i in golden_ids
                          if 1 <= i <= len(pk_s1["t_peak"])]
        t_golden = pk_s1["t_peak"][valid_golden_0]

        # 旧方法下黄金峰命中率
        pads = _adaptive_pad(label, PARAMS)
        s_old = label["start"].to_numpy() + al_old["lag"] - pads
        e_old = label["end"].to_numpy() + al_old["lag"] + pads
        in_old = np.zeros(len(valid_golden_0), dtype=bool)
        for j in range(len(s_old)):
            in_old |= (t_golden >= s_old[j]) & (t_golden <= e_old[j])
        old_golden_hit = int(in_old.sum())

        results.append({
            "rabbit": rabbit,
            "week": week,
            "golden_total": len(valid_golden_0),
            "old_lag": al_old["lag"],
            "old_score": al_old.get("score", 0),
            "old_golden_hits": old_golden_hit,
            "old_golden_rate": old_golden_hit / max(len(valid_golden_0), 1),
            "new_lag": al_new["lag"],
            "new_golden_hits": al_new["golden_hits"],
            "new_golden_rate": al_new["golden_hit_rate"],
        })

        # 生成对比图
        save_path = OUT_EXP / f"compare_{rabbit}_{week}.png"
        plot_comparison(rabbit, week, pp_s1, pk_s1, label,
                        al_old, al_new, golden_ids, save_path)

        print(f"[done] {rabbit} {week}  "
              f"旧lag={al_old['lag']:+.1f}s 黄金命中={old_golden_hit}/{len(valid_golden_0)}  |  "
              f"新lag={al_new['lag']:+.1f}s 黄金命中={al_new['golden_hits']}/{al_new['golden_total']}")

    # 输出对比表
    df = pd.DataFrame(results)
    df.to_csv(OUT_EXP / "comparison_results.csv", index=False)

    # 打印汇总
    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'rabbit':<6} {'week':<4} {'N_golden':<9} "
          f"{'旧lag':>6} {'旧命中':>8} {'旧率':>6}  "
          f"{'新lag':>6} {'新命中':>8} {'新率':>6}  {'改善':>6}")
    print("-" * 70)
    for _, row in df.iterrows():
        improve = row["new_golden_rate"] - row["old_golden_rate"]
        print(f"{row['rabbit']:<6} {row['week']:<4} {row['golden_total']:<9} "
              f"{row['old_lag']:>+5.1f}s {row['old_golden_hits']:>4}/{row['golden_total']:<3} "
              f"{row['old_golden_rate']:>5.0%}  "
              f"{row['new_lag']:>+5.1f}s {row['new_golden_hits']:>4}/{row['golden_total']:<3} "
              f"{row['new_golden_rate']:>5.0%}  "
              f"{improve:>+5.0%}")

    print(f"\n结果已保存到: {OUT_EXP}")
    return df


if __name__ == "__main__":
    run_experiment()
