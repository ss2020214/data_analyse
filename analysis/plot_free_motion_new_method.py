# -*- coding: utf-8 -*-
"""
使用新方法（黄金峰窗口对齐）绘制自由运动 ΔR/R₀ 散点图和趋势统计图。

仅绘制 w1、w4、w6 三周，使用 experiment_golden_window 实验得到的新 lag 值。
数据来源：新窗口内 Stage 2 检出的峰值点的 ΔR/R₀。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from .config import GROUPS, OUT_FIG, OUT_INTER, PARAMS
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .peaks import (detect_peaks_stage1, detect_peaks_stage2, _adaptive_pad)
from .standard_motion import get_reference_profile
from .experiment_window import estimate_lag_golden, GOLDEN_PEAKS

# ============================================================
# 配色与输出目录
# ============================================================
WEEK_COLORS = {
    "w1": "#FF8C00",  # 橙
    "w4": "#1E90FF",  # 蓝
    "w6": "#9932CC",  # 紫
}

TARGET_WEEKS = ["w1", "w4", "w6"]
RABBITS = ["030", "071", "075", "020"]

OUT_DIR = OUT_FIG / "delta_r_r0" / "free_motion_detail_new_method"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取预计算的新方法 lag
COMPARISON_CSV = OUT_INTER / "experiment_golden_window" / "comparison_results.csv"


def _load_new_lags() -> dict:
    """从实验结果CSV读取新方法的lag值，返回 {(rabbit, week): lag}。"""
    df = pd.read_csv(COMPARISON_CSV)
    return {(str(row["rabbit"]).zfill(3), row["week"]): row["new_lag"]
            for _, row in df.iterrows()}


def _extract_free_motion_new_window(rabbit: str, week: str, new_lag: float,
                                    n_per_week: int = 50) -> pd.DataFrame:
    """
    使用新方法lag提取自由运动活跃窗口内 Stage 2 峰值点的 ΔR/R₀。

    与原方法区别：lag 来自黄金峰优化（±5s搜索），而非 physio-hit-rate。
    数据来源：Stage 2 检出的峰值点中，落在新窗口内的峰。
    """
    try:
        t, r = load_session(rabbit, week)
        if t.size < 50:
            return pd.DataFrame()
        label = load_label(rabbit, week)
        if label.empty:
            return pd.DataFrame()
    except (KeyError, FileNotFoundError):
        return pd.DataFrame()

    # 获取标准运动参考
    ref_profile = None
    try:
        ref_profile = get_reference_profile(rabbit, week)
    except Exception:
        pass

    # Stage 2 预处理 + 寻峰
    pp = preprocess(t, r, ref_profile=ref_profile)
    pk = detect_peaks_stage2(pp, ref_profile=ref_profile)
    t_uni = pp["t"]
    r_clean = pp["r_clean"]
    baseline = pp["baseline"]

    if pk["idx"].size == 0:
        return pd.DataFrame()

    # 使用新方法的 lag 确定活跃窗口
    pads = _adaptive_pad(label, PARAMS)
    s_arr = label["start"].to_numpy() + new_lag - pads
    e_arr = label["end"].to_numpy() + new_lag + pads

    # 找窗口内的 Stage 2 峰
    t_peak = pk["t_peak"]
    in_window = np.zeros(t_peak.size, dtype=bool)
    for s_i, e_i in zip(s_arr, e_arr):
        in_window |= (t_peak >= s_i) & (t_peak <= e_i)

    peak_idx_in = pk["idx"][in_window]
    t_peak_in = t_peak[in_window]
    if peak_idx_in.size == 0:
        return pd.DataFrame()

    # 计算峰值点的 ΔR/R₀
    bl_at_peak = baseline[peak_idx_in]
    bl_safe = np.where(np.abs(bl_at_peak) > 1e-6, bl_at_peak, np.nan)
    delta_r_r0 = (r_clean[peak_idx_in] - bl_at_peak) / bl_safe

    # 上限过滤：使用标准运动参考
    upper_limit = None
    if ref_profile is not None:
        ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
        if ref_rel_amp > 0:
            upper_limit = ref_rel_amp * 1.2

    # 过滤有效值：仅保留 ΔR/R₀ > 0 的正向响应峰
    valid = np.isfinite(delta_r_r0) & (delta_r_r0 > 0)
    if upper_limit is not None:
        valid = valid & (delta_r_r0 <= upper_limit)

    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return pd.DataFrame()

    records = []
    for vi in valid_idx:
        records.append({
            "week": week,
            "time_s": float(t_peak_in[vi]),
            "delta_r_r0": float(delta_r_r0[vi]),
        })
    return pd.DataFrame(records)


def plot_rabbit_free_motion(rabbit: str, new_lags: dict):
    """为单只兔子绘制自由运动 ΔR/R₀ 图（散点 + 统计）。

    包含 w1、w4、w6 三周数据。
    """
    # 提取三周数据
    all_data = []
    for week in TARGET_WEEKS:
        lag = new_lags.get((rabbit, week))
        if lag is None:
            print(f"    [skip] {rabbit} {week} — 无新方法lag")
            continue
        df_week = _extract_free_motion_new_window(rabbit, week, lag)
        if not df_week.empty:
            all_data.append(df_week)
            print(f"    {rabbit} {week}: {len(df_week)} 个峰值点, lag={lag:+.1f}s")
        else:
            print(f"    [warn] {rabbit} {week}: 无有效数据")

    if not all_data:
        print(f"    [ERROR] {rabbit}: 三周均无数据")
        return None

    df_all = pd.concat(all_data, ignore_index=True)

    # 绘图：1×2 子图（左散点，右统计）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle(f"Rabbit {rabbit} — Free movement ΔR/R₀ (新窗口方法, w1/w4/w6)",
                 fontsize=13, fontweight='bold')

    # --- 左：散点图（按周着色，横轴为实际时间） ---
    ax = axes[0]
    x_offset = 0
    for w in TARGET_WEEKS:
        sub = df_all[df_all["week"] == w]
        if sub.empty:
            x_offset += 10
            continue
        t_local = sub["time_s"].to_numpy()
        t_rel = t_local - t_local.min() if t_local.size > 0 else t_local
        ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                   c=WEEK_COLORS.get(w, "gray"), s=18, alpha=0.65,
                   label=w)
        x_offset += (t_rel.max() if t_rel.size else 5) + 15

    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement (新窗口对齐)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylim(bottom=0)

    # --- 右：均值±标准差统计图 ---
    ax = axes[1]
    means, stds, weeks_valid = [], [], []
    for w in TARGET_WEEKS:
        sub = df_all[df_all["week"] == w]["delta_r_r0"]
        if sub.size >= 2:
            means.append(sub.mean())
            stds.append(sub.std())
            weeks_valid.append(int(w[1:]))
    if means:
        ax.errorbar(weeks_valid, means, yerr=stds,
                    fmt='o-', color="#1E90FF", capsize=5, lw=2, markersize=9)
    ax.set_xlabel("week")
    ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free (mean±std)", fontsize=11)
    ax.set_xticks([1, 4, 6])
    ax.grid(alpha=0.2)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_path = OUT_DIR / f"{rabbit}_new_method.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    已保存: {save_path}")
    return save_path


def main():
    """为正常组4只兔子生成新方法自由运动 ΔR/R₀ 图。"""
    print("=" * 60)
    print("新方法（黄金峰窗口对齐）— 自由运动 ΔR/R₀ 趋势图")
    print("=" * 60)

    new_lags = _load_new_lags()
    print(f"已加载 {len(new_lags)} 组新方法 lag 值\n")

    for rabbit in RABBITS:
        print(f"\n处理兔子 {rabbit}...")
        try:
            plot_rabbit_free_motion(rabbit, new_lags)
        except Exception as e:
            import traceback
            print(f"    [ERROR] {rabbit}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"完成! 图表已保存到: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
