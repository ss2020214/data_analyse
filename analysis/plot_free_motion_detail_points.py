# -*- coding: utf-8 -*-
"""
自由运动选点诊断图：在 Stage 2 的 ΔR/R₀ 曲线上标注被选中的采样点。

每只兔子每周一张图，展示：
- Stage 2 detrended 信号转换为 ΔR/R₀ 的时序曲线
- 活跃窗口高亮区域
- 被选中的采样点（红色圆点）
- 上限阈值线（橙色虚线）

用于诊断选点分布是否合理、为何某些周标准差过大。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from .config import WEEKS, PARAMS, OUT_FIG, RABBIT2GROUP
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .peaks import (detect_peaks_stage1, detect_peaks_stage2,
                    estimate_lag, _adaptive_pad, assign_peaks_to_windows)
from .standard_motion import get_reference_profile

OUT_DIR = OUT_FIG / "delta_r_r0" / "free_motion_detail"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_free_motion_detail(rabbit: str, weeks=None):
    """为指定兔子绘制每周的自由运动选点诊断图。

    参数：
        rabbit: 兔子编号
        weeks: 指定周列表，None 则绘制全部周
    """
    if weeks is None:
        weeks = WEEKS

    for week in weeks:
        try:
            t, r = load_session(rabbit, week)
            if t.size < 50:
                continue
            label = load_label(rabbit, week)
            if label.empty:
                continue
        except (KeyError, FileNotFoundError):
            continue

        # 标准运动参考
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

        # Stage 2 寻峰
        pk = detect_peaks_stage2(pp, ref_profile=ref_profile)

        # lag 对齐
        pp_s1 = preprocess_raw(t, r)
        pk_s1 = detect_peaks_stage1(pp_s1)
        al = estimate_lag(pp_s1, pk_s1, label)
        lag = al["lag"]

        # 活跃窗口
        pads = _adaptive_pad(label, PARAMS)
        s_arr = label["start"].to_numpy() + lag - pads
        e_arr = label["end"].to_numpy() + lag + pads

        # 窗口内遮罩
        in_window = np.zeros(len(t_uni), dtype=bool)
        for s_i, e_i in zip(s_arr, e_arr):
            in_window |= (t_uni >= s_i) & (t_uni <= e_i)

        # 计算全信号的 ΔR/R₀
        bl_safe = np.where(np.abs(baseline) > 1e-6, baseline, np.nan)
        delta_r_r0_full = (r_clean - baseline) / bl_safe

        # 窗口内选点逻辑（复现 _extract_free_motion_points）
        idx_in = np.where(in_window)[0]
        if idx_in.size < 5:
            continue

        bl_in = baseline[idx_in]
        bl_in_safe = np.where(np.abs(bl_in) > 1e-6, bl_in, np.nan)
        delta_r_r0_win = (r_clean[idx_in] - bl_in) / bl_in_safe

        upper_limit = None
        if ref_profile is not None:
            ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
            if ref_rel_amp > 0:
                upper_limit = ref_rel_amp * 1.2

        valid = np.isfinite(delta_r_r0_win) & (delta_r_r0_win > 0)
        if upper_limit is not None:
            valid = valid & (delta_r_r0_win <= upper_limit)

        n_per_week = 50
        n_positive = int(valid.sum())
        if n_positive >= n_per_week:
            pool_idx = np.where(valid)[0]
            rng = np.random.default_rng(seed=hash((rabbit, week)) & 0xFFFF)
            chosen = rng.choice(pool_idx, size=n_per_week, replace=False)
        elif n_positive > 0:
            chosen = np.where(valid)[0]
        else:
            fallback = np.isfinite(delta_r_r0_win)
            if upper_limit is not None:
                fallback = fallback & (delta_r_r0_win <= upper_limit)
            chosen = np.where(fallback)[0][:n_per_week] if fallback.any() else np.array([], dtype=int)

        # Stage 2 峰值落在窗口内的（已内置生理学峰过滤）
        win_id = assign_peaks_to_windows(pk, label, lag)
        peaks_in_win = win_id >= 0

        # --- 绘图 ---
        fig, ax = plt.subplots(figsize=(14, 5))

        # 绘制全信号 ΔR/R₀ 曲线
        ax.plot(t_uni, delta_r_r0_full, color="gray", lw=0.6, alpha=0.7,
                label="ΔR/R₀ (Stage 2)")

        # 高亮活跃窗口
        for s_i, e_i in zip(s_arr, e_arr):
            ax.axvspan(s_i, e_i, alpha=0.12, color="skyblue")

        # 标注被选中的采样点
        chosen_global_idx = idx_in[chosen]
        ax.scatter(t_uni[chosen_global_idx], delta_r_r0_full[chosen_global_idx],
                   c="red", s=30, zorder=5, alpha=0.8,
                   label=f"选中采样点 (n={len(chosen)})")

        # 标注 Stage 2 窗口内峰值（已经是生理学峰）
        if peaks_in_win.any():
            pk_idx_in = pk["idx"][peaks_in_win]
            ax.scatter(t_uni[pk_idx_in], delta_r_r0_full[pk_idx_in],
                       c="green", s=50, marker="^", zorder=6, alpha=0.8,
                       label=f"生理学峰 (n={peaks_in_win.sum()})")

        # 上限线
        if upper_limit is not None:
            ax.axhline(upper_limit, color="orange", ls="--", lw=1.5,
                       label=f"上限 (ref×1.2={upper_limit:.4f})")

        # 零线
        ax.axhline(0, color="gray", ls="--", lw=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔR/R₀")
        ax.set_title(f"Rabbit {rabbit} — {week} — 自由运动选点详情  "
                     f"(lag={lag:.1f}s, group={RABBIT2GROUP[rabbit]})",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.2)

        # y 轴裁剪：避免极端值压缩可视区域
        y_data = delta_r_r0_full[np.isfinite(delta_r_r0_full)]
        if y_data.size > 0:
            p5, p95 = np.percentile(y_data, [2, 98])
            margin = (p95 - p5) * 0.3
            ax.set_ylim(p5 - margin, p95 + margin)

        fig.tight_layout()
        save_path = OUT_DIR / f"{rabbit}_{week}_points_detail.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    已保存: {save_path}")


def main():
    """为 015 和 065 生成自由运动选点诊断图。"""
    print("=" * 60)
    print("自由运动选点诊断图")
    print("=" * 60)

    for rabbit in ["015", "065"]:
        print(f"\n处理兔子 {rabbit} ({RABBIT2GROUP[rabbit]})...")
        plot_free_motion_detail(rabbit)

    print(f"\n完成! 图表已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
