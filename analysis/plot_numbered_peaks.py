# -*- coding: utf-8 -*-
"""
为正常组生成带峰编号标注的 dual_stage 图。
每个峰标注序号（1,2,3...），同时输出编号-时间对照 CSV。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .config import GROUPS, OUT_INTER, PARAMS, RABBIT2GROUP
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .peaks import (detect_peaks_stage1, detect_peaks_stage1_adaptive,
                    detect_peaks_stage2, estimate_lag, assign_peaks_to_windows)
from .standard_motion import get_reference_profile
from .plot import _robust_ylim

NORMAL_RABBITS = GROUPS["normal"]  # ["030", "075", "071", "020"]
WEEKS = ["w1", "w2", "w3", "w4", "w5", "w6"]

OUT_DIR = OUT_INTER / "numbered_peaks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_dual_stage_numbered(rabbit, week, pp_s1, pk_s1, pp, pk, label, align,
                             save_path, ref_profile=None):
    """双阶段寻峰对比图 + 峰编号标注。

    Stage 1 峰编号用 S1-1, S1-2, ...
    Stage 2 峰编号用 S2-1, S2-2, ...
    返回两个 DataFrame: (stage1_peaks_info, stage2_peaks_info)
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    lag = align.get("lag", 0.0)

    def _draw_label_windows(ax):
        if not label.empty:
            s_arr = label["start"].to_numpy() + lag
            e_arr = label["end"].to_numpy() + lag
            for s_i, e_i in zip(s_arr, e_arr):
                ax.axvspan(s_i, e_i, alpha=0.12, color="#1f77b4")

    records_s1 = []
    records_s2 = []

    # --- Stage 1 ---
    ax = axes[0]
    t1 = pp_s1["t"]
    d1 = pp_s1["detrended"]
    ax.plot(t1, d1, color="#666", lw=0.6, label="detrended (minimal)")
    ax.axhline(pp_s1["mad"] * 0.8, color="#ff7f0e", ls=":", lw=0.8,
               label=f"S1 height thr (0.8×MAD={pp_s1['mad']*0.8:.0f})")

    if pk_s1["idx"].size:
        win_id_s1 = assign_peaks_to_windows(pk_s1, label, lag)
        in_win = win_id_s1 >= 0
        out_win = ~in_win
        n_in = int(in_win.sum())
        n_out = int(out_win.sum())
        if out_win.any():
            ax.scatter(pk_s1["t_peak"][out_win], d1[pk_s1["idx"][out_win]],
                       c="#333333", s=14, zorder=4, marker="v", alpha=0.7,
                       label=f"窗口外 (n={n_out})")
        if in_win.any():
            ax.scatter(pk_s1["t_peak"][in_win], d1[pk_s1["idx"][in_win]],
                       c="#e63946", s=22, zorder=5, marker="v",
                       label=f"窗口内 (n={n_in})")

        # 标注峰编号
        for i, (tp, idx_i) in enumerate(zip(pk_s1["t_peak"], pk_s1["idx"])):
            peak_num = i + 1
            records_s1.append({
                "peak_id": f"S1-{peak_num}",
                "time_s": float(tp),
                "amplitude": float(d1[idx_i]),
                "in_window": bool(in_win[i]),
                "window_id": int(win_id_s1[i]) if in_win[i] else -1,
            })
            # 标注文字（小字号，避免过度拥挤）
            ax.annotate(str(peak_num), xy=(tp, d1[idx_i]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=5, ha="center", color="#222",
                        fontweight="bold")

    _draw_label_windows(ax)
    ylim1 = _robust_ylim(d1, d1[pk_s1["idx"]] if pk_s1["idx"].size else None)
    if ylim1 is not None:
        ax.set_ylim(*ylim1)
    ax.set_ylabel("ΔR (Stage 1)")
    ax.set_title(f"Rabbit {rabbit} — {week}  |  "
                 f"Stage 1 (minimal preprocess)  "
                 f"n_peaks={pk_s1['idx'].size}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # --- Stage 2 ---
    ax = axes[1]
    t2 = pp["t"]
    d2 = pp["detrended"]
    ax.plot(t2, d2, color="#444", lw=0.6, label="detrended (cleaned)")
    h_k = PARAMS["peak_height_mad_k"]
    ax.axhline(pp["mad"] * h_k, color="#ff7f0e", ls="--", lw=0.8,
               label=f"S2 height thr ({h_k}×MAD={pp['mad']*h_k:.0f})")

    if pk["idx"].size:
        win_id_s2 = assign_peaks_to_windows(pk, label, lag)
        in_win2 = win_id_s2 >= 0
        out_win2 = ~in_win2
        n_in2 = int(in_win2.sum())
        n_out2 = int(out_win2.sum())
        if out_win2.any():
            ax.scatter(pk["t_peak"][out_win2], d2[pk["idx"][out_win2]],
                       c="#333333", s=14, zorder=4, marker="^", alpha=0.7,
                       label=f"窗口外 (n={n_out2})")
        if in_win2.any():
            ax.scatter(pk["t_peak"][in_win2], d2[pk["idx"][in_win2]],
                       c="#2ca02c", s=22, zorder=5, marker="^",
                       label=f"窗口内 (n={n_in2})")

        # 标注峰编号
        for i, (tp, idx_i) in enumerate(zip(pk["t_peak"], pk["idx"])):
            peak_num = i + 1
            records_s2.append({
                "peak_id": f"S2-{peak_num}",
                "time_s": float(tp),
                "amplitude": float(d2[idx_i]),
                "in_window": bool(in_win2[i]),
                "window_id": int(win_id_s2[i]) if in_win2[i] else -1,
            })
            ax.annotate(str(peak_num), xy=(tp, d2[idx_i]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=5, ha="center", color="#222",
                        fontweight="bold")

    _draw_label_windows(ax)
    ylim2 = _robust_ylim(d2, d2[pk["idx"]] if pk["idx"].size else None)
    if ylim2 is not None:
        ax.set_ylim(*ylim2)
    ref_txt = ""
    if ref_profile:
        ref_txt = (f"  ref: amp={ref_profile.get('peak_amp_median',0):.0f}Ω "
                   f"SNR={ref_profile.get('snr',0):.1f}")
    ax.set_ylabel("ΔR (Stage 2)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Stage 2 (full clean, ref-guided)  "
                 f"n_peaks={pk['idx'].size}  "
                 f"lag={lag:+.1f}s  hit={align.get('hit_rate',0):.2f}"
                 f"{ref_txt}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return pd.DataFrame(records_s1), pd.DataFrame(records_s2)


def run_normal_numbered():
    """对正常组全部 session 生成带编号的图，并汇总峰信息表。"""
    all_s1 = []
    all_s2 = []

    for rabbit in NORMAL_RABBITS:
        for week in WEEKS:
            try:
                t, r = load_session(rabbit, week)
            except KeyError:
                continue
            if t.size < 50:
                continue
            label = load_label(rabbit, week)

            ref_profile = None
            try:
                ref_profile = get_reference_profile(rabbit, week)
            except Exception:
                pass

            # Stage 1
            pp_s1 = preprocess_raw(t, r)
            pk_s1 = detect_peaks_stage1(pp_s1)
            al = estimate_lag(pp_s1, pk_s1, label)
            pk_s1 = detect_peaks_stage1_adaptive(pp_s1, label, al["lag"])

            # Stage 2
            pp = preprocess(t, r, ref_profile=ref_profile)
            pk = detect_peaks_stage2(pp, ref_profile=ref_profile)

            save_path = OUT_DIR / f"dual_numbered_{rabbit}_{week}.png"
            df_s1, df_s2 = plot_dual_stage_numbered(
                rabbit, week, pp_s1, pk_s1, pp, pk, label, al,
                save_path, ref_profile=ref_profile)

            if not df_s1.empty:
                df_s1 = df_s1.assign(rabbit=rabbit, week=week)
                all_s1.append(df_s1)
            if not df_s2.empty:
                df_s2 = df_s2.assign(rabbit=rabbit, week=week)
                all_s2.append(df_s2)

            print(f"[done] {rabbit} {week}  S1={pk_s1['idx'].size}  "
                  f"S2={pk['idx'].size}  lag={al['lag']:+.1f}s")

    # 汇总保存
    if all_s1:
        pd.concat(all_s1, ignore_index=True).to_csv(
            OUT_DIR / "peak_index_stage1.csv", index=False)
    if all_s2:
        pd.concat(all_s2, ignore_index=True).to_csv(
            OUT_DIR / "peak_index_stage2.csv", index=False)

    print(f"\n完成！图片和对照表已保存到: {OUT_DIR}")


if __name__ == "__main__":
    run_normal_numbered()
