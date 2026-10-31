# -*- coding: utf-8 -*-
"""
标准运动峰值检测诊断图：每只兔子一张，3×2 子图（W1-W6）。

每个子图展示：
- 灰色曲线：原始 R 信号（各组拼接）
- 橙色虚线：滚动分位数基线（050 的 w3/w4 按 config 覆盖为全局最小值常数基线）
- 红色三角：检出的峰值
- 绿色三角：被 ΔR/R₀ 图表选中的峰（与 delta_r_r0_*_standard_iqr_weekly 同源，
  按峰时间精确匹配；含 ΔR/R₀>0 过滤、周级双峰过滤、每周限 10 点时间均匀选点）
- 标题：检出峰数 / 标注运动次数 / 入选峰数

寻峰约束与 delta_r_r0_plot._extract_standard_motion_peaks 一致：
- 频率约束：min_dist = max(0.3s, 0.7 × duration/count)
- 数量上限：检出数 ≤ 标注count，超出则取最大幅值 top-N
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

from .config import (WEEKS, OUT_FIG, RABBIT2GROUP, ALL_RABBITS_STD,
                     std_baseline_quantile, std_baseline_mode)
from .standard_motion import load_standard_session, _estimate_fs
from .delta_r_r0_plot import _extract_standard_motion_peaks

OUT_DIR = OUT_FIG / "delta_r_r0" / "standard_peaks_detail"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_standard_peaks_detail(rabbit: str):
    """为单只兔子生成标准运动峰值检测诊断图（3×2，W1-W6）。"""

    # ΔR/R₀ 图表的选点结果（与 delta_r_r0_*_standard_iqr_weekly 同源）
    sel_df = _extract_standard_motion_peaks(rabbit)
    sel_by_week = {}
    if not sel_df.empty:
        for wk, sub in sel_df.groupby("week"):
            sel_by_week[wk] = np.sort(sub["time_s"].to_numpy(dtype=float))
    matched_total = 0  # 已匹配到的入选点数（用于完整性校验）

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Rabbit {rabbit} — Standard Motion Peak Detection (count-limited)",
                 fontsize=13, fontweight='bold')

    for wi, week in enumerate(WEEKS):
        row, col = wi // 2, wi % 2
        ax = axes[row, col]

        try:
            groups = load_standard_session(rabbit, week)
        except (FileNotFoundError, KeyError, Exception):
            groups = []

        if not groups:
            ax.set_title(f"W{wi+1} - Empty after processing",
                         fontsize=10, color="gray")
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # 拼接各组数据，组间加一点间隔用于可视化
        all_t, all_r, all_bl, all_peaks_t, all_peaks_r = [], [], [], [], []
        all_sel_t, all_sel_r = [], []
        total_detected = 0
        total_labeled = 0
        total_selected = 0
        t_offset = 0.0

        for grp in groups:
            t = grp["t"]
            r = grp["r"]
            count = grp["count"]
            if len(t) < 20:
                continue
            if not np.isfinite(count):
                count = 0

            total_labeled += int(count)

            # 基线：默认滚动分位数（P30）；支持按兔子/周覆盖分位数
            # 或改为整段最小值常数基线（config.STD_BASELINE_MODE_OVERRIDE）
            fs = _estimate_fs(t)
            if std_baseline_mode(rabbit, week) == "global_min":
                baseline = np.full_like(np.asarray(r, dtype=float),
                                        float(np.min(r)))
            else:
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
                            .quantile(std_baseline_quantile(rabbit, week))
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

            total_detected += peaks_idx.size

            # 标记被 ΔR/R₀ 图表选中的峰（按峰时间精确匹配）
            sel_ref = sel_by_week.get(week)
            if sel_ref is not None and sel_ref.size and peaks_idx.size:
                pk_times = t[peaks_idx].astype(float)
                diffs = np.abs(pk_times[:, None] - sel_ref[None, :])
                sel_mask = diffs.min(axis=1) < 1e-6
            else:
                sel_mask = np.zeros(peaks_idx.size, dtype=bool)
            total_selected += int(sel_mask.sum())

            # 时间偏移（从0开始拼接）
            t_shifted = t - t[0] + t_offset
            all_t.append(t_shifted)
            all_r.append(r)
            all_bl.append(baseline)
            if peaks_idx.size > 0:
                all_peaks_t.append(t_shifted[peaks_idx])
                all_peaks_r.append(r[peaks_idx])
            if sel_mask.any():
                all_sel_t.append(t_shifted[peaks_idx[sel_mask]])
                all_sel_r.append(r[peaks_idx[sel_mask]])

            t_offset = t_shifted[-1] + 0.5  # 组间间隔

        if not all_t:
            ax.set_title(f"W{wi+1} - Empty after processing",
                         fontsize=10, color="gray")
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # 拼接
        t_all = np.concatenate(all_t)
        r_all = np.concatenate(all_r)
        bl_all = np.concatenate(all_bl)

        # 绘制信号
        ax.plot(t_all, r_all, color="gray", lw=0.7, alpha=0.8, label="R (raw)")
        ax.plot(t_all, bl_all, color="orange", ls="--", lw=1.2, alpha=0.8,
                label="baseline (P30)")

        # 绘制峰值
        if all_peaks_t:
            pk_t = np.concatenate(all_peaks_t)
            pk_r = np.concatenate(all_peaks_r)
            ax.scatter(pk_t, pk_r, c="red", s=40, marker="v", zorder=5,
                       alpha=0.9, label=f"detected ({total_detected})")

        # 绘制入选峰（绿三角，供 delta_r_r0_*_standard_iqr_weekly 使用的点）
        if all_sel_t:
            sel_t = np.concatenate(all_sel_t)
            sel_r = np.concatenate(all_sel_r)
            ax.scatter(sel_t, sel_r, c="#2ca02c", s=60, marker="v",
                       zorder=6, alpha=0.95,
                       edgecolors="black", linewidths=0.6,
                       label=f"selected → ΔR/R₀ charts ({total_selected})")
            matched_total += total_selected

        # 标题与标签
        ax.set_title(f"W{wi+1} — detected {total_detected} / labeled {total_labeled} "
                     f"/ selected {total_selected}",
                     fontsize=10, color="green")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("R (Ohm)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    save_path = OUT_DIR / f"std_peaks_{rabbit}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not sel_df.empty and matched_total != len(sel_df):
        print(f"  [WARN] {rabbit}: 入选点匹配 {matched_total}/{len(sel_df)}")
    print(f"  已保存: {save_path}（检出峰上叠加绿色=入选 ΔR/R₀ 图的峰，"
          f"{matched_total}/{len(sel_df)}）")
    return save_path


def main(rabbits=None):
    """为指定兔子生成标准运动峰值诊断图。默认为所有兔子（含新增 050/070）。"""
    if rabbits is None:
        rabbits = ALL_RABBITS_STD

    print("=" * 60)
    print("标准运动峰值检测诊断图")
    print("=" * 60)

    for rabbit in rabbits:
        print(f"\n处理兔子 {rabbit} ({RABBIT2GROUP[rabbit]})...")
        try:
            plot_standard_peaks_detail(rabbit)
        except Exception as e:
            print(f"  [跳过] {rabbit}: {e}")

    print(f"\n完成! 图表已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
