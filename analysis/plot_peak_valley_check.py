# -*- coding: utf-8 -*-
"""峰-谷配对验证：导出结构谷中间结果 CSV + 峰谷局部放大图。

用途：核对文档法 ΔR/R₀ 的谷值寻峰是否正确：
    delta_r_r0 = (peak_r - valley_r) / R0
其中 valley_r 是该峰关联的「结构谷」（find_peaks 的 left_base 处阻值）。

产出（存 output/tables/ 与 output/figures/peak_valley_check/）：
  1. peak_valley_collect.csv：每峰一行
     rabbit, group, week, peak_time_s, peak_r, valley_time_s, valley_r,
     left_base_idx, peak_idx, R0_val, delta_r_r0
  2. 每只兔子一张峰谷局部放大图（<= MAX_PEAKS 个峰的窗口，画原始阻值 +
     detrended，标注峰点/谷点/R0 基线）。

CLI:
    python -m analysis.plot_peak_valley_check --rabbits 030 --all
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

from .config import (GROUPS, WEEKS, OUT_FIG, OUT_TAB, RABBIT2GROUP,
                     ALL_RABBITS)
from .peak_trough_delta import (extract_rest_resistance_r0, _rough_detrend,
                                _select_standard_peaks, peak_trough_delta,
                                _est_fs, _find_left_bases)

OUT_PV = OUT_FIG / "peak_valley_check"
OUT_PV.mkdir(parents=True, exist_ok=True)
MAX_PEAKS = 40   # 每只兔子最多画多少峰的窗口
WIN_S = 1.5      # 每个峰局部窗口半宽（秒）


def collect_peak_valley(rabbit: str, params: dict = None) -> pd.DataFrame:
    """对单只兔子标准运动收集每峰的结构谷中间结果（不做采样/双峰过滤），
    返回 DataFrame: rabbit, group, week, peak_time_s, peak_r, valley_time_s,
    valley_r, left_base_idx, peak_idx, R0_val, delta_r_r0。
    """
    from .standard_motion import load_standard_session, _estimate_fs
    p = params or {}
    records = []
    for week in WEEKS:
        try:
            groups = load_standard_session(rabbit, week)
        except Exception:
            continue
        for gi, grp in enumerate(groups):
            t = grp["t"]
            r = grp["r"]
            count = grp.get("count", 0)
            if len(t) < 20:
                continue
            if not np.isfinite(count):
                count = 0
            R0, _ = extract_rest_resistance_r0(
                t, r, active_mask=None,
                r_quantile=p.get("doc_r0_quantile", 0.20), params=p)
            fs = _estimate_fs(t)
            detrended = _rough_detrend(
                r, fs, win_s=p.get("doc_detrend_window_s", 5.0))
            mad = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
            if mad <= 0:
                mad = float(np.nanstd(detrended)) or 1.0
            peaks_idx = _select_standard_peaks(detrended, mad, count, fs, p)
            if peaks_idx.size == 0:
                continue
            delta, valley_r, left_base = peak_trough_delta(
                r, detrended, peaks_idx, R0, fs, p)
            for i in range(len(peaks_idx)):
                pb, vb = peaks_idx[i], left_base[i]
                if not np.isfinite(delta[i]) or delta[i] <= 0:
                    continue
                records.append({
                    "rabbit": rabbit,
                    "group": RABBIT2GROUP.get(rabbit, ""),
                    "week": week,
                    "group_idx": gi,
                    "peak_time_s": float(t[pb]),
                    "peak_idx": int(pb),
                    "peak_r": float(r[pb]),
                    "valley_time_s": float(t[vb]),
                    "left_base_idx": int(vb),
                    "valley_r": float(valley_r[i]),
                    "R0_val": float(R0),
                    "delta_r_r0": float(delta[i]),
                })
    return pd.DataFrame(records)


def _plot_rabbit(rabbit: str, df: pd.DataFrame):
    """画单只兔子峰谷局部放大图（峰点/谷点/R0 标注）。"""
    if df.empty:
        print(f"    [skip] {rabbit}: 无峰数据")
        return
    # 抽样子峰数
    n_show = min(MAX_PEAKS, len(df))
    step = max(1, len(df) // MAX_PEAKS) if len(df) > MAX_PEAKS else 1
    show_idx = list(range(0, len(df), step))[:n_show]

    cols = 4
    rows = int(np.ceil(len(show_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows),
                             squeeze=False)
    axes = axes.ravel()

    # 每峰对应的原始 r / detrended 放在 map 缓存
    sig_cache = {}

    def _sig(week, gi):
        key = (week, gi)
        if key not in sig_cache:
            from .standard_motion import load_standard_session, _estimate_fs
            grp = load_standard_session(rabbit, week)[gi]
            fs = _estimate_fs(grp["t"])
            det = _rough_detrend(grp["r"], fs)
            sig_cache[key] = (grp["t"], grp["r"], det)
        return sig_cache[key]

    for ai, i in enumerate(show_idx):
        ax = axes[ai]
        row = df.iloc[i]
        t, r, det = _sig(row["week"], int(row["group_idx"]))
        # 窗口同时覆盖峰与谷（谷可能在峰左/右远处），保护越界
        pk_idx = int(row["peak_idx"])
        vl_idx = int(row["left_base_idx"])
        cm = int((pk_idx + vl_idx) / 2)
        win = max(5, int(WIN_S * _est_fs(t)))
        lo = max(0, cm - win)
        hi = min(len(t), cm + win + 1)
        if lo >= hi:
            lo, hi = max(0, pk_idx - 1), min(len(t), pk_idx + 2)
        seg_t = t[lo:hi]
        # 纵轴对齐绘图顺序（每组完整曲线）
        seg_r = r[lo:hi]
        seg_det = det[lo:hi]
        # 峰/谷在段内的相对索引
        ipk = pk_idx - lo
        ivl = vl_idx - lo
        ipk = int(np.clip(ipk, 0, len(seg_t) - 1))
        ivl = int(np.clip(ivl, 0, len(seg_t) - 1))
        # 原始阻值 R：放在主轴，聚焦峰谷局部（放大差异）
        ax.plot(seg_t, seg_r, '-', color="#1E90FF", lw=1.2, label="R")
        ax.axhline(row["R0_val"], color="#228B22", lw=1, ls="--", alpha=0.8,
                   label=f"R0={row['R0_val']:.1f}")
        ax.plot(seg_t[ipk], seg_r[ipk], 'o', color="#DC143C", ms=7,
                label=f"peak R={row['peak_r']:.1f}")
        ax.plot(seg_t[ivl], seg_r[ivl], 'v', color="#9932CC", ms=8,
                label=f"valley R={row['valley_r']:.1f}")
        ax.set_title(
            f"w{row['week'][1:]} t={row['peak_time_s']:.1f}s\n"
            f"ΔR/R0={row['delta_r_r0']:.3f} (pk{row['peak_idx']}-vl{row['left_base_idx']})",
            fontsize=9)
        ax.set_xlabel("t(s)")
        ax.set_ylabel("R(ohm)")
        ax.grid(alpha=0.2)
        # detrended 移到独立右轴（隐藏刻度），避免其数值范围拉大主轴的 y 缩放
        ax2 = ax.twinx()
        ax2.plot(seg_t, seg_det, '-', color="#BBBBBB", lw=1.0, alpha=0.7)
        ax2.set_yticks([])
        # y 轴聚焦峰谷局部：以 (peak, valley, R0) 为界加对称 padding
        pk_v = float(row["peak_r"]); vl_v = float(row["valley_r"])
        r0_v = float(row["R0_val"])
        span = max(abs(pk_v - vl_v), 1e-9)
        pad = max(span * 0.20, abs(vl_v) * 1e-3, 1e-6)
        lo_v = min(vl_v, r0_v) - pad
        hi_v = max(pk_v, r0_v) + pad
        ax.set_ylim(lo_v, hi_v)
        if ai == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper right")
    # 空子图关闭
    for ai in range(len(show_idx), rows * cols):
        axes[ai].axis("off")
    fig.suptitle(f"Rabbit {rabbit} ({RABBIT2GROUP.get(rabbit, '')}) "
                 f"peak-valley paired ({n_show} shown)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUT_PV / f"peak_valley_{rabbit}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    峰谷图: {path}  (n_total={len(df)}, 展示{n_show})")


def run(rabbits):
    print("\n================ 峰谷配对验证（标准运动） ================")
    all_df = []
    for rabbit in rabbits:
        try:
            df = collect_peak_valley(rabbit)
            print(f"    兔子 {rabbit}: {len(df)} 峰谷对")
        except Exception as e:
            print(f"    [ERROR] {rabbit}: {e}")
            continue
        if df.empty:
            continue
        all_df.append(df)
        try:
            _plot_rabbit(rabbit, df)
        except Exception as e:
            print(f"    [skip 峰谷图 {rabbit}] {e}")

    if all_df:
        out = pd.concat(all_df, ignore_index=True)
        path = OUT_TAB / "peak_valley_collect.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n谷值中间结果表已保存: {path}  (共 {len(out)} 行)")
        # 全局谷值合理性统计
        ok_r = out["valley_r"] <= out["peak_r"]
        ok_pos = (out["peak_time_s"] - out["valley_time_s"]) >= 0
        print(f"  [抽查] valley<=peak 比例={ok_r.mean():.3f}  "
              f"谷在峰左比例={ok_pos.mean():.3f}  "
              f"delta>0 比例={(out['delta_r_r0']>0).mean():.3f}")
    else:
        print("    未收集到任何峰谷数据")


def main(argv=None):
    parser = argparse.ArgumentParser(description="峰谷配对验证")
    parser.add_argument("--rabbits", nargs="+", default=None,
                        help="兔子列表，如 030 068")
    parser.add_argument("--all", action="store_true", help="所有兔子")
    args = parser.parse_args(argv)
    rabbits = ALL_RABBITS if (args.all or not args.rabbits) else args.rabbits
    run(rabbits)


if __name__ == "__main__":
    main()