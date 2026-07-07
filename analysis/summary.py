# -*- coding: utf-8 -*-
"""
\u751f\u6210\u53d1\u8868\u7ea7\u6700\u7ec8\u6c47\u603b\u56fe\uff1a4 \u4e2a\u5b50\u56fe + \u8de8\u5468\u5bf9\u6bd4\u3002
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from analysis.config import OUT_TAB, OUT_FIG, GROUPS
from analysis.plot import rabbit_color_marker

rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC",
                               "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

GROUP_ORDER = ["normal", "pathological", "infection", "sham"]


def main():
    df = pd.read_csv(OUT_TAB / "features_per_session.csv")
    weeks = sorted(df["week"].unique(), key=lambda x: int(x[1:]))

    panels = [
        ("amp_norm_mean",         "(a) Peak amplitude / baseline\n(发力幅度, 越大 → 跳跃越剧烈)"),
        ("amp_cv",                "(b) Amplitude CV\n(发力均一性, 越高 → 越不稳)"),
        ("burst_density_per_min", "(c) Bursts per minute\n(跨度/跳跃频次, 越高 → 越活跃)"),
        ("ipi_cv",                "(d) Inter-peak interval CV\n(节律规整度, 越高 → 越不规则)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    axes = axes.ravel()

    groups_present = [g for g in GROUP_ORDER if (df["group"] == g).any()]
    colors, markers, prefix = rabbit_color_marker(df)

    for ax, (feat, title) in zip(axes, panels):
        legend_handles = {g: [] for g in groups_present}
        for g in groups_present:
            sub = df[df["group"] == g]
            for r in sorted(sub["rabbit"].astype(str).unique()):
                sr = (sub[sub["rabbit"].astype(str) == r]
                      .set_index("week")[feat].reindex(weeks).to_numpy())
                x = np.arange(len(weeks))
                line, = ax.plot(x, sr, color=colors[r], marker=markers[g],
                                lw=1.5, markersize=5.5, alpha=0.9,
                                label=f"{prefix[g]}-{r}")
                legend_handles[g].append(line)
                valid = ~np.isnan(sr)
                if valid.any():
                    j = int(np.where(valid)[0][-1])
                    ax.annotate(f"{prefix[g]}-{r}", xy=(x[j], sr[j]),
                                xytext=(5, 0), textcoords="offset points",
                                fontsize=7, color=colors[r],
                                va="center", ha="left")
        ax.set_xticks(np.arange(len(weeks))); ax.set_xticklabels(weeks)
        ax.set_xlabel("Week")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        handles, labels = [], []
        for g in groups_present:
            for h in legend_handles[g]:
                handles.append(h); labels.append(h.get_label())
        ax.legend(handles, labels, fontsize=7, loc="best", ncol=2,
                  framealpha=0.85,
                  title="● normal  ■ pathological  ◆ infection  ▲ sham",
                  title_fontsize=7.5)

    n_counts = {g: df[df['group']==g]['rabbit'].nunique() for g in groups_present}
    title_parts = [f"{g.title()} (n={n_counts[g]})" for g in groups_present]
    fig.suptitle(f"个体轨迹 — {' vs '.join(title_parts)} — 6 weeks",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "summary_4panels.png", dpi=160)
    plt.close(fig)
    print("Saved", OUT_FIG / "summary_4panels.png")

    # 跨周汇总表：三组在所有周的汇总
    pool = []
    feat_cols = ["amp_norm_mean", "amp_mean", "amp_cv", "amp_std",
                 "ipi_median", "ipi_cv", "baseline_drift_pct",
                 "burst_density_per_min", "peaks_in_label",
                 "detect_recall", "duration_s"]
    for g in groups_present:
        sub = df[df["group"] == g]
        row = {"group": g, "n_sessions": len(sub),
               "n_rabbits": sub["rabbit"].nunique()}
        for f in feat_cols:
            row[f"{f}_mean"] = sub[f].mean()
            row[f"{f}_std"] = sub[f].std()
        pool.append(row)
    pd.DataFrame(pool).to_csv(OUT_TAB / "group_overall.csv", index=False)
    print("Saved", OUT_TAB / "group_overall.csv")


if __name__ == "__main__":
    main()
