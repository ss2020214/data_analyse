# -*- coding: utf-8 -*-
"""
W1 基准对齐版汇总图。

动机：兔子之间的「初始（w1）水平」差异较大，直接比较绝对值会混淆
「初始状态的个体差异」与「随时间的变化趋势」。
本脚本对每只兔子的每个特征做差值对齐：
        v'_w(i) = v_w(i) - v_w1
使所有兔子在 w1 处统一为 0，仅保留相对 w1 的变化趋势，便于在
同一坐标轴下对比 6 周的形状（趋势）而非起点。

为何用差值而不是比值/百分比：
- 部分特征 w1 数值接近 0（如 amp_norm_mean ≈ 0.005），比值会数值爆炸；
- 差值保持原量纲，正负号代表「相对 w1 增加/减少」，直观且稳健；
- 若需要百分比，可在 ratio 模式下用 (v - v_w1) / max(|v_w1|, eps) 输出，
  本脚本同时保留该可选实现。

输出：output/figures/summary_4panels_w1aligned.png
      output/tables/features_w1_aligned.csv  （对齐后的 long-format 表）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from analysis.config import OUT_TAB, OUT_FIG

rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC",
                               "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

GROUP_COLOR = {"normal": "#2ca02c", "pathological": "#d62728",
               "infection": "#ff7f0e", "sham": "#1f77b4"}
LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
GROUP_ORDER = ["normal", "pathological", "infection", "sham"]


def align_to_w1(df: pd.DataFrame, feature: str, mode: str = "diff") -> pd.DataFrame:
    """对每只兔子做 w1 基准对齐。

    mode:
      - "diff"  : v - v_w1   （量纲保留，w1 处恒为 0）
      - "ratio" : (v - v_w1) / max(|v_w1|, eps)  （百分比变化，无量纲）
    没有 w1 数据的兔子会被剔除并打印告警。
    """
    out = []
    skipped = []
    for r, g in df.groupby("rabbit"):
        w1_row = g[g["week"] == "w1"]
        if w1_row.empty or pd.isna(w1_row[feature].iloc[0]):
            skipped.append(r)
            continue
        v0 = float(w1_row[feature].iloc[0])
        gg = g.copy()
        if mode == "diff":
            gg[feature] = gg[feature] - v0
        elif mode == "ratio":
            denom = max(abs(v0), 1e-9)
            gg[feature] = (gg[feature] - v0) / denom
        else:
            raise ValueError(f"unknown mode {mode}")
        out.append(gg)
    if skipped:
        print(f"  [warn] feature={feature} 缺 w1 的兔子已剔除: {skipped}")
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def _per_week_mean_sem(df, group, feature, weeks):
    means, sems = [], []
    sub = df[df["group"] == group]
    for w in weeks:
        v = sub[sub["week"] == w][feature].dropna()
        means.append(v.mean() if len(v) else np.nan)
        sems.append(v.std(ddof=1) / np.sqrt(max(1, len(v))) if len(v) > 1 else 0)
    return np.array(means), np.array(sems)


def _draw_panel(ax, df_aligned, feat, title, weeks, rabbit_styles, groups_present):
    for g in groups_present:
        mu, se = _per_week_mean_sem(df_aligned, g, feat, weeks)
        x = np.arange(len(weeks))
        n_rab = df_aligned[df_aligned["group"] == g]["rabbit"].nunique()
        grp_label = f"{g} (n={n_rab}, " + ("mean\u00b1SEM" if n_rab > 1 else "single") + ")"
        ax.errorbar(x, mu, yerr=se, marker="o", lw=2.4, capsize=3,
                    color=GROUP_COLOR[g], label=grp_label, zorder=5)
        sub = df_aligned[df_aligned["group"] == g]
        for r in sorted(sub["rabbit"].astype(str).unique()):
            sr = (sub[sub["rabbit"].astype(str) == r]
                  .set_index("week")[feat].reindex(weeks).to_numpy())
            ax.plot(x, sr, color=GROUP_COLOR[g], alpha=0.55,
                    marker=".", lw=0.9, ls=rabbit_styles[r],
                    label=f"{g[0].upper()}-{r}")
            valid = ~np.isnan(sr)
            if valid.any():
                j = int(np.where(valid)[0][-1])
                ax.annotate(r, xy=(x[j], sr[j]),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=GROUP_COLOR[g],
                            va="center", ha="left", alpha=0.85)
    # w1 = 0 参考线
    ax.axhline(0, color="#888", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(np.arange(len(weeks))); ax.set_xticklabels(weeks)
    ax.set_xlabel("Week")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.85)


def main(mode: str = "diff"):
    df = pd.read_csv(OUT_TAB / "features_per_session.csv")
    weeks = sorted(df["week"].unique(), key=lambda x: int(x[1:]))
    groups_present = [g for g in GROUP_ORDER if (df["group"] == g).any()]

    panels = [
        ("amp_norm_mean",         "(a) Δ Peak amplitude / baseline\n(\u76f8\u5bf9 w1, \u4e0a\u5347\u4e3a\u6b63)"),
        ("amp_cv",                "(b) Δ Amplitude CV\n(\u76f8\u5bf9 w1, \u4e0a\u5347 \u2192 \u8d8a\u4e0d\u7a33)"),
        ("burst_density_per_min", "(c) Δ Bursts per minute\n(\u76f8\u5bf9 w1, \u4e0a\u5347 \u2192 \u8d8a\u6d3b\u8dc3)"),
        ("ipi_cv",                "(d) Δ Inter-peak interval CV\n(\u76f8\u5bf9 w1, \u4e0b\u964d \u2192 \u8282\u5f8b\u8d8a\u89c4\u5219)"),
    ]

    # 为每只兔子预分配线型（4 个子图保持一致）
    rabbit_styles = {}
    for g in groups_present:
        rs = sorted(df[df["group"] == g]["rabbit"].astype(str).unique())
        for i, r in enumerate(rs):
            rabbit_styles[r] = LINE_STYLES[i % len(LINE_STYLES)]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    axes = axes.ravel()

    # 同时收集对齐后的长表写入 CSV
    long_rows = []
    for ax, (feat, title) in zip(axes, panels):
        df_a = align_to_w1(df, feat, mode=mode)
        _draw_panel(ax, df_a, feat, title, weeks, rabbit_styles, groups_present)
        for _, r in df_a[["rabbit", "group", "week", feat]].iterrows():
            long_rows.append({"rabbit": r["rabbit"], "group": r["group"],
                              "week": r["week"], "feature": feat,
                              "value_aligned": r[feat], "mode": mode})

    suptitle = ("W1-aligned trend (\u6240\u6709\u5154\u5b50 w1 \u7edf\u4e00\u96f6\u70b9, \u53ea\u770b\u8d8b\u52bf)\n"
                "Normal (n=6) vs Pathological (n=3) vs Sham (n=1) \u2014 6 weeks")
    fig.suptitle(suptitle, fontsize=12, y=1.00)
    fig.tight_layout()
    out_path = OUT_FIG / "summary_4panels_w1aligned.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("Saved", out_path)

    pd.DataFrame(long_rows).to_csv(OUT_TAB / "features_w1_aligned.csv",
                                   index=False)
    print("Saved", OUT_TAB / "features_w1_aligned.csv")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "diff"
    main(mode=mode)
