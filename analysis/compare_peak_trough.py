# -*- coding: utf-8 -*-
"""
文档法 ΔR/R₀（峰值−谷值 / 全局 R₀）与旧法（动态基线 / 逐峰）对比。

产出（存 output/figures/delta_r_r0_compare/ 与 output/tables/）：
  1. 逐峰散点：新旧两法 ΔR/R₀ 对照（XY 相关 + R²），标准 / 自由各一张。
  2. 每周均值±std 双线对比（标准 / 自由），新旧各一条线。
  3. 汇总表 new_vs_old_compare.csv（rabbit, week, old_mean, new_mean,
     delta_pct, corr, n）。
  4. 分组（normal/pathological/infection/sham）周趋势并排小图。

CLI:
    python -m analysis.compare_peak_trough --rabbits 030 --weeks w1..w6 --all
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

from .config import (GROUPS, WEEKS, OUT_FIG, OUT_TAB, RABBIT2GROUP,
                     ALL_RABBITS)
from .peak_trough_delta import (extract_standard_motion_delta,
                                extract_free_motion_delta)
from .delta_r_r0_plot import (_extract_standard_motion_peaks,
                              _extract_free_motion_peaks_in_window)

OUT_CMP = OUT_FIG / "delta_r_r0_compare"
OUT_CMP.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 配对与相关系数
# ---------------------------------------------------------------------------
def _match_standard(old: pd.DataFrame, new: pd.DataFrame,
                    tol_s: float = 1.0) -> pd.DataFrame:
    """按 week + 最近 time_s 将新旧标准法逐峰配对。

    两法寻峰参数一致、仅 detrend 基线窗口不同，峰位置接近；对每个新峰
    在该周内找时间最近的旧峰，距离 < tol_s 视为同一物理峰。
    """
    paired = []
    if old is None or new is None or old.empty or new.empty:
        return pd.DataFrame(paired)
    for w in WEEKS:
        o = old[old["week"] == w]
        n = new[new["week"] == w]
        if o.empty or n.empty:
            continue
        o_t = o["time_s"].to_numpy()
        n_t = n["time_s"].to_numpy()
        for i, nt in enumerate(n_t):
            dt = np.abs(o_t - nt)
            j = int(np.argmin(dt))
            if dt[j] <= tol_s:
                paired.append({
                    "week": w,
                    "old_time_s": float(o_t[j]),
                    "new_time_s": float(nt),
                    "old_delta": float(o.iloc[j]["delta_r_r0"]),
                    "new_delta": float(n.iloc[i]["delta_r_r0"]),
                })
    return pd.DataFrame(paired)


def _corr_r2(x: np.ndarray, y: np.ndarray):
    """皮尔逊相关系数 r 与决定系数 R²。"""
    if len(x) < 3:
        return np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(r):
        return np.nan, np.nan
    return float(r), float(r ** 2)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _plot_scatter(pairs: pd.DataFrame, title: str, name: str):
    """逐峰配对散点（新旧 XY 对照）。"""
    if pairs.empty:
        print(f"    [skip] {name} 无配对数据")
        return
    r, r2 = _corr_r2(pairs["old_delta"].to_numpy(),
                     pairs["new_delta"].to_numpy())
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(pairs["old_delta"], pairs["new_delta"], s=18, alpha=0.6,
               c="#1E90FF", edgecolors="k", linewidths=0.3)
    lim = [min(0, pairs["old_delta"].min(), pairs["new_delta"].min()),
           max(pairs["old_delta"].max(), pairs["new_delta"].max())]
    ax.plot(lim, lim, "r--", lw=1, label="y=x")
    ax.set_xlabel("Old ΔR/R0 (动态基线)")
    ax.set_ylabel("New ΔR/R0 (峰值−谷值 / R0)")
    ax.set_title(f"{title}\nn={len(pairs)}, r={r:.3f}, R2={r2:.3f}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = OUT_CMP / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    散点图: {path}  (r={r:.3f}, R2={r2:.3f}, n={len(pairs)})")


def _weekly_rows(df: pd.DataFrame):
    """按周聚合 mean/std。"""
    rows = []
    for w in WEEKS:
        sub = df[df["week"] == w]["delta_r_r0"]
        if sub.size >= 2:
            rows.append({"week": int(w[1:]), "mean": sub.mean(),
                         "std": sub.std(), "n": sub.size})
    return rows


def _plot_weekly(old: pd.DataFrame, new: pd.DataFrame, title: str, name: str):
    """每周均值±std 双线对比。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    if not old.empty:
        ot = _weekly_rows(old)
        if ot:
            w_ = [r["week"] for r in ot]
            ax.errorbar(w_, [r["mean"] for r in ot], yerr=[r["std"] for r in ot],
                        fmt='o-', color="#999999", capsize=4, lw=2,
                        markersize=7, label="Old")
    if not new.empty:
        nt = _weekly_rows(new)
        if nt:
            w_ = [r["week"] for r in nt]
            ax.errorbar(w_, [r["mean"] for r in nt], yerr=[r["std"] for r in nt],
                        fmt='s--', color="#DC143C", capsize=4, lw=2,
                        markersize=7, label="New")
    ax.set_xlabel("week")
    ax.set_ylabel("ΔR/R0")
    ax.set_title(title)
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = OUT_CMP / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    双线图: {path}")


def _group_trend(agg: pd.DataFrame, kind: str):
    """分组周趋势并排小图（new 法，已跨兔子聚合，含 rabbit 列）。"""
    if agg.empty:
        return
    groups = ["normal", "pathological", "infection", "sham"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, g in zip(axes.ravel(), groups):
        dg = agg[agg["rabbit"].isin(GROUPS[g])]
        if dg.empty:
            ax.set_title(f"{g} (无数据)", fontsize=10)
            ax.set_xticks(range(1, 7))
            ax.grid(alpha=0.2)
            continue
        means, stds, weeks_valid = [], [], []
        for w in WEEKS:
            sub = dg[dg["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                means.append(sub.mean())
                stds.append(sub.std())
                weeks_valid.append(int(w[1:]))
        if means:
            ax.errorbar(weeks_valid, means, yerr=stds, fmt='o-',
                        color="#228B22", capsize=4, lw=2, markersize=7)
        ax.set_xlabel("week")
        ax.set_ylabel("ΔR/R0")
        ax.set_title(f"{g}", fontsize=10)
        ax.set_xticks(range(1, 7))
        ax.grid(alpha=0.2)
    fig.suptitle(f"Group weekly trend - New doc-method ({kind})", fontsize=13)
    fig.tight_layout()
    path = OUT_CMP / f"group_trend_{kind}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    分组图: {path}")


# ---------------------------------------------------------------------------
# 单只兔子对比
# ---------------------------------------------------------------------------
def compare_rabbit(rabbit: str) -> dict:
    """对单只兔子计算新旧两法各项对比，返回 dict:
    {"summary": 汇总行, "new_std": 新法标准逐峰df(含rabbit),
     "new_free": 新法自由df(含rabbit)}。
    """
    print(f"\n===== 兔子 {rabbit} ({RABBIT2GROUP[rabbit]}) =====")

    # --- 标准运动 ---
    old_std = _extract_standard_motion_peaks(rabbit)
    new_std = extract_standard_motion_delta(rabbit)
    pairs_std = _match_standard(old_std, new_std)

    # --- 自由运动（physio_peak，新旧同寻峰可直接配对）---
    old_free = _extract_free_motion_peaks_in_window(rabbit)
    new_free = extract_free_motion_delta(rabbit, mode="physio_peak")
    pairs_free = pd.DataFrame()
    if not old_free.empty and not new_free.empty:
        joined = []
        for w in WEEKS:
            o = old_free[old_free["week"] == w]
            n = new_free[new_free["week"] == w]
            if o.empty or n.empty:
                continue
            o_t = o["time_s"].to_numpy()
            n_t = n["time_s"].to_numpy()
            # 同寻峰逻辑 → 按时间最近配对
            for i, nt in enumerate(n_t):
                dt = np.abs(o_t - nt)
                j = int(np.argmin(dt))
                if dt[j] <= 0.5:
                    joined.append({
                        "week": w,
                        "old_delta": float(o.iloc[j]["delta_r_r0"]),
                        "new_delta": float(n.iloc[i]["delta_r_r0"]),
                    })
        pairs_free = pd.DataFrame(joined)

    # --- 图 ---
    _plot_scatter(pairs_std, f"Rabbit {rabbit} - Standard paired",
                  f"scatter_std_{rabbit}.png")
    _plot_scatter(pairs_free, f"Rabbit {rabbit} - Free physio-peaks paired",
                  f"scatter_free_{rabbit}.png")
    _plot_weekly(old_std, new_std, f"Rabbit {rabbit} - Standard weekly mean±std",
                 f"weekly_std_{rabbit}.png")
    _plot_weekly(old_free, new_free,
                 f"Rabbit {rabbit} - Free weekly mean±std",
                 f"weekly_free_{rabbit}.png")

    # --- 汇总行 ---
    row = {"rabbit": rabbit, "group": RABBIT2GROUP[rabbit]}
    for key, pairs in (("std", pairs_std), ("free", pairs_free)):
        if pairs is None or pairs.empty:
            row[f"corr_{key}"] = np.nan
            row[f"r2_{key}"] = np.nan
            row[f"n_{key}"] = 0
            continue
        r_, r2_ = _corr_r2(pairs["old_delta"].to_numpy(),
                           pairs["new_delta"].to_numpy())
        row[f"corr_{key}"] = r_
        row[f"r2_{key}"] = r2_
        row[f"n_{key}"] = len(pairs)

    # 带 rabbit 列的逐峰数据，供分组趋势图聚合
    new_std_r = new_std.copy() if not new_std.empty else None
    new_free_r = new_free.copy() if not new_free.empty else None
    if new_std_r is not None:
        new_std_r["rabbit"] = rabbit
    if new_free_r is not None:
        new_free_r["rabbit"] = rabbit
    return {"summary": row, "new_std": new_std_r, "new_free": new_free_r}


def run_all_compare(rabbits):
    """对所有指定兔子跑对比并汇总表。"""
    print("\n================ 文档法 vs 旧法 全量对比 ================")
    rows = []
    new_std_all = []
    new_free_all = []
    for rabbit in rabbits:
        try:
            res = compare_rabbit(rabbit)
            rows.append(res["summary"])
            if res["new_std"] is not None:
                new_std_all.append(res["new_std"])
            if res["new_free"] is not None:
                new_free_all.append(res["new_free"])
        except Exception as e:
            print(f"    [ERROR] {rabbit}: {e}")
            rows.append({"rabbit": rabbit, "group": RABBIT2GROUP[rabbit],
                         "error": str(e)})

    df = pd.DataFrame(rows)
    df["rabbit"] = df["rabbit"].astype(str)
    path = OUT_TAB / "new_vs_old_compare.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表已保存: {path}")

    # 分组周趋势：跨兔子聚合 new 法逐峰数据
    try:
        if new_std_all:
            _group_trend(pd.concat(new_std_all, ignore_index=True), "std")
        if new_free_all:
            _group_trend(pd.concat(new_free_all, ignore_index=True), "free")
    except Exception as e:
        print(f"    [skip 分组图] {e}")
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="文档法 vs 旧法 ΔR/R₀ 对比")
    parser.add_argument("--rabbits", nargs="+", default=None,
                        help="兔子列表，如 030 068")
    parser.add_argument("--all", action="store_true",
                        help="对所有兔子运行对比")
    args = parser.parse_args(argv)

    if args.all or not args.rabbits:
        rabbits = ALL_RABBITS
    else:
        rabbits = args.rabbits

    run_all_compare(rabbits)


if __name__ == "__main__":
    main()