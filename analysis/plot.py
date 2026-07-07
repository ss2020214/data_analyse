# -*- coding: utf-8 -*-
"""
绘图工具：
1. plot_session  —— 单 session 的中间结果图（原始/基线/检出峰/标注窗）
2. plot_alignment_curve —— 对齐 lag-score 曲线
3. group_compare_plot —— 组间逐周对比（mean ± SEM）
4. heatmap         —— 兔子×周 的特征热图

为兼容 mac 默认环境，强制使用非 GUI 后端 + DejaVu Sans（避免中文字体警告影响渲染）。
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

# 字体：尝试系统中文字体；找不到则用英文 + 数字标注
_zh_candidates = ["PingFang SC", "Heiti SC", "Songti SC",
                  "Arial Unicode MS", "Microsoft YaHei", "Noto Sans CJK SC"]
rcParams["font.sans-serif"] = _zh_candidates + ["DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ===== 特征中文描述字典 =====
# 每张图自动查找并在底部添加生物学含义说明
FEATURE_DESCRIPTION = {
    # 基线/漂移特征
    "baseline_mean":          "基础阻值 — 传感器静息状态的电阻值，反映钢板/组织状态",
    "baseline_drift_pct":     "基线漂移% — 会话内基线变化程度，反映传感器/组织瘕合进程稳定性",
    "baseline_slope_pct_min": "基线斜率 — 基线随时间的线性趋势，正值=逐渐升高",
    "baseline_short_drift_p95":"短期漂移幅度 — 30s窗口内基线波动范围，反映短期稳定性",
    "baseline_zero_cross_per_min": "基线穿越频次 — 基线波动频率，值越大基线越不稳定",
    "baseline_rms":           "基线RMS — 基线波动总能量，越大表示漂移越严重",
    # 峰幅与发力特征
    "amp_mean":               "峰幅均值(Ω) — 跳跃时钢板形变幅度，值越小表示骨强度恢复越好（力回到骨上）",
    "amp_norm_mean":          "峰值/基线 — 无量纲发力指标，值越小表示骨愈合越好（钢板形变减小）",
    "amp_median":             "峰幅中位数 — 中位数发力幅度，抗极端值干扰",
    "amp_cv":                 "峰幅变异系数 — 发力均一性，值越高表示每次跳跃力度越不稳定",
    "amp_std":                "峰幅标准差 — 发力幅度的离散程度",
    # 节律特征
    "ipi_median":             "峰间期中位数(s) — 相邻跳跃的时间间隔，反映运动节律",
    "ipi_cv":                 "峰间期变异系数 — 运动节律规整度，值越低表示节律越规则",
    "burst_density_per_min":  "每分钟突发数 — 运动活跃度，值越高表示越活跃",
    "jumps_per_min":          "每分钟跳跃数 — 累计跳跃频率",
    "jumps_per_active_min":   "活跃段跳跃密度 — 排除静息时间后的运动强度",
    # 峰形态特征
    "peak_fwhm_median":       "峰宽中位数(s) — 跳跃事件持续时间，反映发力模式",
    "peak_rise_time_median":  "上升时间(s) — 从基线到峰值的时间，反映发力速度",
    "peak_fall_time_median":  "下降时间(s) — 从峰值回到基线的时间，反映松弛速度",
    "peak_asymmetry_median":  "峰不对称性 — (fall-rise)/(fall+rise)，正值=下降慢于上升",
    # 频域特征
    "psd_peak_freq_hz":       "主频率(Hz) — 跳跃信号的主导频率，反映运动模式",
    "psd_jump_band_ratio":    "跳跃频带能量比 — 0.5-4Hz频段能量占比，反映跳跃信号纯度",
    # 时序模式
    "burst_duration_median":  "突发持续时间(s) — 每次连续跳跃的持续时长",
    "silence_median_s":       "静息期中位数(s) — 相邻突发之间的休息时间",
    "silence_max_s":          "最长静息期(s) — 会话内最长的不活动时段",
    "silence_cv":             "静息期变异 — 休息时间的规律性，值越低越规则",
    # 峰间关系
    "amp_autocorr_lag1":      "峰幅自相关 — 相邻峰幅值的相关性，反映发力一致性",
    "amp_trend_slope_per_min":"峰幅趋势斜率 — 会话内峰幅随时间的变化，正值=逐渐增大",
    "ipi_trend_slope_s_per_min":"峰间期趋势 — 会话内跳跃间隔的变化，正值=逐渐变慢",
    # v7 新增
    "amp_skewness":           "峰幅偏度 — 峰幅分布形态，正偏=多数小峰偶有大峰",
    "amp_kurtosis":           "峰幅峰度 — 峰幅分布尖锐程度，高值=极端发力更频繁",
    "amp_large_ratio":        "大峰占比 — 超过P75的峰比例，反映强力跳跃频率",
    "peaks_per_burst_median": "每次突发峰数 — 每次连续跳跃包含的跳跃次数",
    "peaks_per_burst_cv":     "突发峰数变异 — 每次突发跳跃次数的稳定性",
    "intra_burst_ipi_median": "突发内峰间期(s) — 同一次连续跳跃内的节律",
    "intra_burst_amp_cv":     "突发内幅值变异 — 同一次连跳内发力的均一性",
    "burst_decay_slope":      "突发衰减斜率 — 连跳过程中峰幅的下降速度，反映疲劳",
    "amp_ratio_first_vs_second_half": "前/后半峰幅比 — >1表示后半段发力减小（疲劳）",
    "ipi_ratio_first_vs_second_half": "前/后半峰间期比 — <1表示后半段节律变慢",
    "warmup_time_to_first_burst": "预热时间(s) — 会话开始到第一次跳跃的时间",
    "sample_entropy":         "样本熵 — 信号复杂度，值越高表示运动模式越不规则",
    "ipi_entropy":            "IPI信息熵 — 跳跃间隔的随机性，值越高节律越不可预测",
    "fano_factor":            "Fano因子 — 30s窗口内峰数的离散度，>1表示聚集性跳跃",
    "recovery_time_median":   "恢复时间(s) — 突发后信号回到基线的时间",
    "post_burst_overshoot":   "突发后基线过冲 — 突发后基线的偏移，反映组织弹性恢复",
    # 检测统计
    "detect_recall":          "检出/标注比 — 算法检出峰数vs人工标注，越接近1越准确",
    "peaks_in_label":         "窗口内检出峰数 — 落入标注窗口的峰总数",
    "hit_rate":               "命中率 — 检出峰落入标注窗口的比例",
    # per-peak 特征
    "amp_rel_baseline":       "ΔR/R₀ — 相对基线的峰值变化率，无量纲跨兔子可比",
    "prominence_rel_baseline":"突出度/基线 — 峰的显著性相对基线的比值",
    "peak_area_rel_baseline": "峰面积/基线 — 单次跳跃事件的总能量指标",
    "peak_fwhm":              "峰宽(s) — 单次跳跃持续时间",
    "peak_rise_time":         "上升时间(s) — 单次跳跃的发力速度",
    "peak_fall_time":         "下降时间(s) — 单次跳跃的松弛速度",
    "peak_asymmetry":         "峰不对称性 — 发力与松弛的时间比例",
    "peak_steepness":         "峰陡度(Ω/s) — 峰高/峰宽，反映发力的锐利程度",
    "intra_burst_ipi":        "突发内峰间期(s) — 同一次连跳内的节律",
}


def _add_feature_description(fig, feature: str):
    """在图底部添加特征含义的中文说明文字。"""
    desc = FEATURE_DESCRIPTION.get(feature, "")
    if desc:
        fig.text(0.5, -0.02, desc, ha="center", va="top",
                 fontsize=8.5, color="#333", style="italic",
                 transform=fig.transFigure)


def plot_session(rabbit, week, pp, peaks, label, align, save_path):
    t = pp["t"]
    duration = float(t[-1] - t[0]) if t.size > 1 else 60.0
    # 根据时长动态调整图宽，确保横轴精细度
    fig_width = max(16, duration / 8)  # 每8秒约1英寸
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    r, b, d = pp["r_clean"], pp["baseline"], pp["detrended"]

    # 顶图：原始 + 基线 + 标注窗（已 lag 平移）
    ax = axes[0]
    ax.plot(t, r, color="#444", lw=0.8, label="resistance (clean)")
    ax.plot(t, b, color="#d62728", lw=1.2, label="baseline")
    # label 窗（lag 平移后）
    if not label.empty:
        s_arr = label["start"].to_numpy() + align["lag"]
        e_arr = label["end"].to_numpy() + align["lag"]
        for s_i, e_i in zip(s_arr, e_arr):
            ax.axvspan(s_i, e_i, alpha=0.15, color="#1f77b4")
    # 检出峰
    if peaks["idx"].size:
        ax.scatter(peaks["t_peak"], r[peaks["idx"]],
                   c="#2ca02c", s=20, zorder=5, label=f"peaks (n={peaks['idx'].size})")
    ax.set_ylabel("Resistance (Ω)")
    ax.set_title(f"Rabbit {rabbit} — {week}    "
                 f"lag={align.get('lag',0):+.2f}s  "
                 f"hit-rate={align.get('hit_rate',0):.2f}  "
                 f"jumps_labeled={int(label['count'].sum()) if not label.empty else 0}")
    ax.legend(loc="upper right", fontsize=8)

    # 底图：detrended + 阈值 + 检出峰
    ax = axes[1]
    ax.plot(t, d, color="#555", lw=0.7, label="detrended")
    ax.axhline(2 * pp["mad"], color="#ff7f0e", ls="--", lw=0.9,
               label="height thr (2·MAD)")
    if peaks["idx"].size:
        ax.scatter(peaks["t_peak"], d[peaks["idx"]],
                   c="#2ca02c", s=18, zorder=5)
    if not label.empty:
        s_arr = label["start"].to_numpy() + align["lag"]
        e_arr = label["end"].to_numpy() + align["lag"]
        for s_i, e_i in zip(s_arr, e_arr):
            ax.axvspan(s_i, e_i, alpha=0.10, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔR (Ω)")
    ax.legend(loc="upper right", fontsize=8)
    # 精细化横轴刻度
    if duration <= 60:
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    elif duration <= 150:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
    elif duration <= 300:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which='minor', alpha=0.2, linestyle=':')
    ax.grid(True, which='major', alpha=0.4, linestyle='-')
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_alignment_curve(rabbit, week, align, save_path):
    if "lags" not in align:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(align["lags"], align["scores"], lw=1.2)
    ax.axvline(align["lag"], color="r", ls="--", lw=1,
               label=f"best={align['lag']:+.2f}s")
    ax.set_xlabel("lag (s)  [shift label by ...]")
    ax.set_ylabel("hit rate (peaks ∈ window)")
    ax.set_title(f"alignment scan — {rabbit} {week}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def _robust_ylim(signal: np.ndarray, peaks_amp: np.ndarray = None,
                 q_lo: float = 0.5, q_hi: float = 97.5, pad: float = 0.15):
    """计算稳健的 y 轴范围，避免极端离群值压缩可视化。

    策略：
    - 上限：使用信号 P97.5，若有峰则取 max(信号P97.5, 峰P95) 以确保大部分峰可见
    - 下限：使用信号 P0.5
    - 加上 15% padding
    """
    s = signal[np.isfinite(signal)]
    if s.size < 10:
        return None
    lo = float(np.percentile(s, q_lo))
    hi = float(np.percentile(s, q_hi))
    # 如果有峰幅值，确保多数峰可见（P95而非最大值）
    if peaks_amp is not None and peaks_amp.size > 2:
        pk_hi = float(np.percentile(peaks_amp[np.isfinite(peaks_amp)], 95))
        hi = max(hi, pk_hi)
    span = hi - lo
    if span <= 0:
        return None
    return (lo - pad * span, hi + pad * span)


def plot_dual_stage(rabbit, week, pp_s1, pk_s1, pp, pk, label, align,
                   save_path, ref_profile=None):
    """双阶段寻峰对比图：上 Stage1(最小化预处理+宽松峰) / 下 Stage2(完整清洗+精确峰)。

    y 轴优化：使用稳健分位数裁剪，避免极端异常峰压缩正常数据的可视范围。
    峰颜色区分：落在活跃窗口内的峰用显著色，窗口外的峰用淡色。
    """
    from .peaks import assign_peaks_to_windows

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    lag = align.get("lag", 0.0)

    # 公共：绘制 label 窗口
    def _draw_label_windows(ax):
        if not label.empty:
            s_arr = label["start"].to_numpy() + lag
            e_arr = label["end"].to_numpy() + lag
            for s_i, e_i in zip(s_arr, e_arr):
                ax.axvspan(s_i, e_i, alpha=0.12, color="#1f77b4")

    # --- Stage 1 ---
    ax = axes[0]
    t1 = pp_s1["t"]
    d1 = pp_s1["detrended"]
    ax.plot(t1, d1, color="#666", lw=0.6, label="detrended (minimal)")
    ax.axhline(pp_s1["mad"] * 0.8, color="#ff7f0e", ls=":", lw=0.8,
               label=f"S1 height thr (0.8×MAD={pp_s1['mad']*0.8:.0f})")
    # 峰的颜色区分：窗口内(红色) vs 窗口外(淡灰)
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
    _draw_label_windows(ax)
    # y 轴稳健裁剪
    ylim1 = _robust_ylim(d1, d1[pk_s1["idx"]] if pk_s1["idx"].size else None)
    if ylim1 is not None:
        ax.set_ylim(*ylim1)
        if pk_s1["idx"].size:
            clipped = d1[pk_s1["idx"]] > ylim1[1]
            n_clipped = int(clipped.sum())
            if n_clipped > 0:
                ax.text(0.02, 0.95, f"⚠ {n_clipped}个异常峰超出显示范围",
                        transform=ax.transAxes, fontsize=8, color="red",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", alpha=0.8))
    ax.set_ylabel("ΔR (Stage 1)")
    ax.set_title(f"Rabbit {rabbit} — {week}  |  "
                 f"Stage 1 (minimal preprocess, loose params)  "
                 f"n_peaks={pk_s1['idx'].size}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # --- Stage 2 ---
    ax = axes[1]
    t2 = pp["t"]
    d2 = pp["detrended"]
    ax.plot(t2, d2, color="#444", lw=0.6, label="detrended (cleaned)")
    from .config import PARAMS
    h_k = PARAMS["peak_height_mad_k"]
    ax.axhline(pp["mad"] * h_k, color="#ff7f0e", ls="--", lw=0.8,
               label=f"S2 height thr ({h_k}×MAD={pp['mad']*h_k:.0f})")
    # 峰的颜色区分：窗口内(绿色) vs 窗口外(淡灰)
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
    _draw_label_windows(ax)
    # y 轴稳健裁剪
    ylim2 = _robust_ylim(d2, d2[pk["idx"]] if pk["idx"].size else None)
    if ylim2 is not None:
        ax.set_ylim(*ylim2)
        if pk["idx"].size:
            clipped = d2[pk["idx"]] > ylim2[1]
            n_clipped = int(clipped.sum())
            if n_clipped > 0:
                ax.text(0.02, 0.95, f"⚠ {n_clipped}个异常峰超出显示范围",
                        transform=ax.transAxes, fontsize=8, color="red",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", alpha=0.8))
    # 标注参考信息
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
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def rabbit_color_marker(df: pd.DataFrame):
    """为每只兔子分配独立颜色（tab20）及 per-rabbit marker。
    - normal       组都用 ●【o】
    - pathological 组都用 ■【s】
    - infection    组都用 ◆【D】
    - sham 组内使用不同三角形，以便区分个体：
        △ ('^') / ▽ ('v') / ◁ ('<') / ▷ ('>')
    返回：
        colors:  dict[str_rabbit -> RGB]
        markers: dict[str_rabbit -> marker_str]
        prefix:  dict[group -> 'N'/'P'/'I'/'S']
    """
    pal = list(plt.get_cmap("tab20").colors)
    sham_pool = ['^', 'v', '<', '>', 'P', 'X', '*']
    group_marker = {"normal": "o", "pathological": "s", "infection": "D"}
    colors, markers = {}, {}
    for g in ["normal", "pathological", "infection", "sham"]:
        rs = sorted(df[df["group"] == g]["rabbit"].astype(str).unique())
        for i, r in enumerate(rs):
            colors[r] = pal[len(colors) % len(pal)]
            if g == "sham":
                markers[r] = sham_pool[i % len(sham_pool)]
            else:
                markers[r] = group_marker[g]
    prefix = {"normal": "N", "pathological": "P", "infection": "I", "sham": "S"}
    return colors, markers, prefix


def _auto_yscale(values: np.ndarray) -> str:
    """启发式选择 y 轴缩放：
    - 全为正且最大/最小跨越 1×10^1.5 → log
    - 含负但量级跨越 2 → symlog
    - 其它 → linear
    """
    v = values[~np.isnan(values)]
    if v.size < 3:
        return "linear"
    if (v > 0).all():
        if v.max() / max(v.min(), 1e-12) >= 30:
            return "log"
    else:
        ratio = np.max(np.abs(v)) / max(np.median(np.abs(v[v != 0])), 1e-12) \
            if (v != 0).any() else 1
        if ratio >= 100:
            return "symlog"
    return "linear"


def feature_trajectory(df: pd.DataFrame, feature: str, title: str,
                       save_path, ylabel=None, yscale: str = "auto"):
    """个体轨迹图：每只兔子一条独立颜色线，组用 marker 形状区分，
    末端标注 N-030 / P-043 / S-068。不再画组均值粗线。

    yscale： 'auto' (默认) / 'linear' / 'log' / 'symlog'
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    weeks = sorted(df["week"].unique(), key=lambda x: int(x[1:]))
    colors, markers, prefix = rabbit_color_marker(df)
    groups_present = [g for g in ["normal", "pathological", "infection", "sham"]
                      if (df["group"] == g).any()]

    legend_handles_by_group = {g: [] for g in groups_present}
    all_values = []
    for g in groups_present:
        sub = df[df["group"] == g]
        for r in sorted(sub["rabbit"].astype(str).unique()):
            sr = (sub[sub["rabbit"].astype(str) == r]
                  .set_index("week")[feature].reindex(weeks).to_numpy())
            all_values.append(sr)
            x = np.arange(len(weeks))
            line, = ax.plot(x, sr,
                            color=colors[r], marker=markers[r],
                            lw=1.5, markersize=5.5, alpha=0.9,
                            label=f"{prefix[g]}-{r}")
            legend_handles_by_group[g].append(line)
            valid = ~np.isnan(sr)
            if valid.any():
                j = int(np.where(valid)[0][-1])
                ax.annotate(f"{prefix[g]}-{r}", xy=(x[j], sr[j]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7.5, color=colors[r],
                            va="center", ha="left")
    # 选择 y 轴缩放
    if yscale == "auto" and all_values:
        flat = np.concatenate(all_values)
        yscale = _auto_yscale(flat)
    if yscale == "log":
        ax.set_yscale("log")
    elif yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(np.arange(len(weeks)))
    ax.set_xticklabels(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel((ylabel or feature) +
                  (f"  [{yscale}]" if yscale not in ("linear",) else ""))
    ax.set_title(title)
    handles, labels = [], []
    for g in groups_present:
        for h in legend_handles_by_group[g]:
            handles.append(h); labels.append(h.get_label())
    ax.legend(handles, labels, fontsize=7, loc="best", ncol=2, framealpha=0.85,
              title="● normal  ■ pathological  ◆ infection  ▲ sham", title_fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    _add_feature_description(fig, feature)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def feature_trajectory_normalized(df: pd.DataFrame, feature: str, title: str,
                                  save_path, ylabel=None, mode: str = "w1"):
    """个体相对轨迹图：每只兔子除以自己的 W1 值（或首个非空值）。
    这样所有个体起点在 1.0，物理量级被消除，只看“相对于自己起点怎么变”。
    mode='w1' 除以 W1；mode='first' 除以首个非空值。接近 0 的特征可能会被放大。
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    weeks = sorted(df["week"].unique(), key=lambda x: int(x[1:]))
    colors, markers, prefix = rabbit_color_marker(df)
    groups_present = [g for g in ["normal", "pathological", "infection", "sham"]
                      if (df["group"] == g).any()]

    legend_handles_by_group = {g: [] for g in groups_present}
    for g in groups_present:
        sub = df[df["group"] == g]
        for r in sorted(sub["rabbit"].astype(str).unique()):
            sr = (sub[sub["rabbit"].astype(str) == r]
                  .set_index("week")[feature].reindex(weeks).to_numpy())
            if mode == "w1":
                ref = sr[0]
            else:
                v = sr[~np.isnan(sr)]
                ref = v[0] if v.size else np.nan
            if not np.isfinite(ref) or abs(ref) < 1e-12:
                continue
            sr_n = sr / ref
            x = np.arange(len(weeks))
            line, = ax.plot(x, sr_n,
                            color=colors[r], marker=markers[r],
                            lw=1.5, markersize=5.5, alpha=0.9,
                            label=f"{prefix[g]}-{r}")
            legend_handles_by_group[g].append(line)
            valid = ~np.isnan(sr_n)
            if valid.any():
                j = int(np.where(valid)[0][-1])
                ax.annotate(f"{prefix[g]}-{r}", xy=(x[j], sr_n[j]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7.5, color=colors[r],
                            va="center", ha="left")
    ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(np.arange(len(weeks)))
    ax.set_xticklabels(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel((ylabel or feature) + "  /  W1")
    ax.set_title(title + "  —  per-rabbit normalized to W1")
    handles, labels = [], []
    for g in groups_present:
        for h in legend_handles_by_group[g]:
            handles.append(h); labels.append(h.get_label())
    ax.legend(handles, labels, fontsize=7, loc="best", ncol=2, framealpha=0.85,
              title="● normal  ■ pathological  ◆ infection  ▲ sham", title_fontsize=7.5)
    ax.grid(alpha=0.25)
    _add_feature_description(fig, feature)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def feature_trajectory_zscore_split(df: pd.DataFrame, feature: str, title: str,
                                    save_path, ylabel=None):
    """分组子图 + z-score 归一化。
    每只兔子的 6 周序列变为 (x - 个体均值) / 个体std，消除个体量级差异，
    只保留"时间形状"。分多个子图分别画各组，不互相拥挤。
    """
    weeks = sorted(df["week"].unique(), key=lambda x: int(x[1:]))
    colors, markers, prefix = rabbit_color_marker(df)
    x = np.arange(len(weeks))

    # 分组数据准备
    groups_main = [g for g in ["normal", "pathological", "infection"] if (df["group"] == g).any()]
    has_sham = (df["group"] == "sham").any()

    fig, axes = plt.subplots(len(groups_main), 1, figsize=(9, 4.3 * len(groups_main)),
                             sharex=True)
    if len(groups_main) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups_main):
        sub = df[df["group"] == g]
        rabbit_list = sorted(sub["rabbit"].astype(str).unique())
        group_z_curves = []
        for r in rabbit_list:
            sr = (sub[sub["rabbit"].astype(str) == r]
                  .set_index("week")[feature].reindex(weeks).to_numpy())
            valid = ~np.isnan(sr)
            if valid.sum() < 2:
                continue
            mu, sd = np.nanmean(sr), np.nanstd(sr, ddof=0)
            if sd < 1e-12:
                z = sr - mu
            else:
                z = (sr - mu) / sd
            group_z_curves.append(z)
            ax.plot(x, z, color=colors[r], marker=markers[r],
                    lw=1.4, markersize=5, alpha=0.85,
                    label=f"{prefix[g]}-{r}")
            # 末端标注
            vld = ~np.isnan(z)
            if vld.any():
                j = int(np.where(vld)[0][-1])
                ax.annotate(f"{prefix[g]}-{r}", xy=(x[j], z[j]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7.5, color=colors[r], va="center", ha="left")

        # 组中位数趋势虚线
        if group_z_curves:
            stack = np.array(group_z_curves)
            med = np.nanmedian(stack, axis=0)
            ax.plot(x, med, color="k", lw=2.5, ls="--", alpha=0.7,
                    label=f"{g} median", zorder=10)
            # 带（IQR）
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
            ax.fill_between(x, q25, q75, color="gray", alpha=0.12)

        # sham 参考线
        if has_sham:
            sub_s = df[df["group"] == "sham"]
            for r in sorted(sub_s["rabbit"].astype(str).unique()):
                sr = (sub_s[sub_s["rabbit"].astype(str) == r]
                      .set_index("week")[feature].reindex(weeks).to_numpy())
                valid = ~np.isnan(sr)
                if valid.sum() < 2:
                    continue
                mu, sd = np.nanmean(sr), np.nanstd(sr, ddof=0)
                z = (sr - mu) / sd if sd > 1e-12 else sr - mu
                ax.plot(x, z, color="#888", marker=markers[r], lw=1, alpha=0.5,
                        ls=":", markersize=4, label=f"S-{r} (sham)")

        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel(f"z-score  ({g})")
        ax.set_title(f"{g.title()} 组 — {title}", fontsize=10)
        ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.8)
        ax.grid(alpha=0.2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(weeks)
    axes[-1].set_xlabel("Week")
    fig.suptitle(f"{title}  —  z-score / 分组子图", fontsize=11, y=1.01)
    _add_feature_description(fig, feature)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def boxplot_groups_at_week(df: pd.DataFrame, feature: str, week_label: str,
                           title: str, save_path,
                           normalize: str = "robust_w1",
                           ref_stats: dict | None = None,
                           ylabel: str | None = None):
    """组间箱线图：在指定「周」上对 normal/pathological/sham 三组画箱线，
    叠加个体散点（同一只兔子使用 trajectory 同色 marker）。

    week_label：取值为 'w1' 或 'last'。
        - 'w1'：所有兔子都在 W1。
        - 'last'：每只兔子使用自己实际有数据的最后一周
          （sham 新增三只=w3，068=w4，normal/pathological=w6）。

    normalize:
        - 'robust_w1' (默认)：以「所有兔子 W1」的 median/MAD 为参考坐标，
          输出 robust z-score = (x - med) / (1.4826*MAD)。
          两张图使用同一参考 → 「W1中位≈0，last-week 偏离位置」即生物学变化。
        - 'minmax_w1'：以全体 W1 的 [P5, P95] 范围归一到 [0,1]。
        - 'none'：不归一化（谨慎使用，可能量纲不同）。
    ref_stats: 外部传入参考统计量（如 W1 的 med/mad/p5/p95），
        存在时优先使用，以保证两张图使用相同参考。
    """
    df = df.copy()
    df["rabbit"] = df["rabbit"].astype(str)

    # 1) 取出每只兔子在指定周的数值
    if week_label == "w1":
        sub = df[df["week"] == "w1"][["rabbit", "group", feature]].copy()
        sub = sub.dropna(subset=[feature])
        x_caption = "W1"
    elif week_label == "last":
        rows = []
        for r, sdf in df.groupby("rabbit"):
            sdf2 = sdf.dropna(subset=[feature])
            if sdf2.empty:
                continue
            sdf2 = sdf2.sort_values("week", key=lambda s: s.map(lambda x: int(x[1:])))
            last = sdf2.iloc[-1]
            rows.append({"rabbit": r, "group": last["group"],
                         feature: last[feature], "week": last["week"]})
        sub = pd.DataFrame(rows)
        x_caption = "each rabbit's last available week"
    else:
        raise ValueError(f"unknown week_label: {week_label}")

    if sub.empty:
        return

    # 2) 计算/使用参考统计量（仅基于 W1 所有兔子）
    if ref_stats is None:
        w1v = df[df["week"] == "w1"][feature].dropna().to_numpy()
        if w1v.size >= 3:
            med = float(np.median(w1v))
            mad = float(np.median(np.abs(w1v - med)))
            if mad < 1e-12:
                mad = float(np.std(w1v)) or 1.0
            p5, p95 = float(np.percentile(w1v, 5)), float(np.percentile(w1v, 95))
            ref_stats = {"med": med, "mad": mad, "p5": p5, "p95": p95}
        else:
            ref_stats = None

    def _zr(x):
        if normalize == "robust_w1" and ref_stats is not None:
            return (x - ref_stats["med"]) / (1.4826 * ref_stats["mad"])
        if normalize == "minmax_w1" and ref_stats is not None:
            denom = ref_stats["p95"] - ref_stats["p5"]
            if abs(denom) < 1e-12:
                return x
            return (x - ref_stats["p5"]) / denom
        return x

    sub = sub.assign(_y=_zr(sub[feature].to_numpy()))

    # 3) 画箱线图 + 散点
    groups_present = [g for g in ["normal", "pathological", "infection", "sham"]
                      if (sub["group"] == g).any()]
    colors_g = {"normal": "#1f77b4", "pathological": "#d62728", "sham": "#7f7f7f"}
    rabbit_colors, markers, prefix = rabbit_color_marker(df)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    box_data = [sub[sub["group"] == g]["_y"].to_numpy() for g in groups_present]
    bp = ax.boxplot(box_data, positions=np.arange(len(groups_present)),
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch, g in zip(bp["boxes"], groups_present):
        patch.set_facecolor(colors_g[g])
        patch.set_alpha(0.28)
        patch.set_edgecolor(colors_g[g])
    for whisker in bp["whiskers"]:
        whisker.set_color("#333")
    for cap in bp["caps"]:
        cap.set_color("#333")

    # 叠加个体散点位置：组位置 ± 随机抖动
    rng = np.random.default_rng(seed=hash(feature) & 0xffff)
    for gi, g in enumerate(groups_present):
        sub_g = sub[sub["group"] == g]
        for _, row in sub_g.iterrows():
            r = row["rabbit"]
            jitter = rng.uniform(-0.15, 0.15)
            ax.scatter(gi + jitter, row["_y"],
                       color=rabbit_colors.get(r, "k"),
                       marker=markers[r], edgecolor="black", linewidth=0.4,
                       s=55, alpha=0.92, zorder=5)
            ax.annotate(f"{prefix[g]}-{r}",
                        xy=(gi + jitter, row["_y"]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=6.5, color="#222", alpha=0.85,
                        va="center", ha="left")

    if normalize == "robust_w1":
        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.55)
        y_label = (ylabel or feature) + "\n(robust z vs all rabbits' W1)"
    elif normalize == "minmax_w1":
        ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.4)
        ax.axhline(1, color="k", lw=0.6, ls=":", alpha=0.4)
        y_label = (ylabel or feature) + "\n(min-max [W1 P5–P95])"
    else:
        y_label = ylabel or feature

    ax.set_xticks(np.arange(len(groups_present)))
    ax.set_xticklabels([f"{g}\n(n={(sub['group']==g).sum()})" for g in groups_present])
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\n@ {x_caption}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def per_rabbit_box_panels(peaks_df: pd.DataFrame, feature: str, title: str,
                          save_path, ylabel: str | None = None,
                          weeks_to_show=("w1", "w3", "last"),
                          normalize: str = "none",
                          min_peaks: int = 3,
                          yscale: str = "linear",
                          y_clip_quantile: float | None = None):
    """大图含 3 个子图（W1 / W3 / each rabbit's last week）。
    每个子图横坐标 = 兔子（按 normal→pathological→sham 排列，组间以
    竖线分隔），每只兔子一个箱，箱包含它在该周所有 in-label 峰的
    该特征分布（per-peak 样本）。
    不画「组均值」箱。

    peaks_df: 需含列 [rabbit, week, group, <feature>] 的 long-format 表。
    normalize:
        - 'none' (默认)：原始单位
        - 'per_rabbit_w1'：每只兔子除以「自己 W1 中位数」，
          同只兔子不同周可以与自己起点比较，消除传感器量纲。
    yscale:
        - 'linear' (默认)
        - 'log'：仅对正值有效（自动过滤 ≤0 的样本，并将 y 下限设为全体
          正值 P1）。适合 *_rel_baseline、peak_steepness 等跨数量级特征，
          视觉上把倍数差异线性化，避免极端离群值压扁正常箱体。
        - 'symlog'：含负值时用，linthresh=median(|x|)/10。
    y_clip_quantile: 若非 None，将 y 轴上限收紧到全体样本的该分位
          （如 0.99 表示去掉 top 1% 极端离群以保持视觉对比），下限同步
          取 1−q 分位（log 模式忽略此项的下限处理）。
    """
    df = peaks_df.copy()
    df["rabbit"] = df["rabbit"].astype(str)

    # 兔子顺序：按组
    rabbit_order, group_for, group_blocks = [], {}, []
    for g in ["normal", "pathological", "infection", "sham"]:
        rs = sorted(df[df["group"] == g]["rabbit"].unique())
        if not rs:
            continue
        group_blocks.append((g, rs))
        for r in rs:
            rabbit_order.append(r)
            group_for[r] = g
    n_r = len(rabbit_order)
    if n_r == 0:
        return

    colors_rabbit, markers, prefix = rabbit_color_marker(df)

    # 预算 per-rabbit W1 中位数（用于归一化）
    w1_med = {}
    for r in rabbit_order:
        w1v = df[(df["rabbit"] == r) & (df["week"] == "w1")][feature].dropna().to_numpy()
        if w1v.size >= min_peaks:
            m = float(np.median(w1v))
            w1_med[r] = m if abs(m) > 1e-12 else np.nan
        else:
            w1_med[r] = np.nan

    def _normalize_vals(r, vals):
        if normalize == "per_rabbit_w1":
            ref = w1_med.get(r, np.nan)
            if not np.isfinite(ref) or abs(ref) < 1e-12:
                return np.array([])
            return vals / ref
        return vals

    # 预先收集所有用于绘制的 y 值（已应用 normalize），用于 log 范围/clip
    _all_vals = []
    for r in rabbit_order:
        for w in weeks_to_show:
            if w == "last":
                sub_w = df[(df["rabbit"] == r) & df[feature].notna()]
                if sub_w.empty:
                    continue
                wlbl = max(sub_w["week"].unique(), key=lambda x: int(x[1:]))
            else:
                wlbl = w
            v = df[(df["rabbit"] == r) & (df["week"] == wlbl)][feature].dropna().to_numpy()
            v = _normalize_vals(r, v)
            if v.size:
                _all_vals.append(v)
    flat = np.concatenate(_all_vals) if _all_vals else np.array([])

    # 计算 y 轴范围（统一 sharey）
    ylim = None
    if yscale == "log":
        pos = flat[flat > 0]
        if pos.size >= 5:
            lo = float(np.percentile(pos, 1))
            hi = float(np.percentile(pos, 99.5))
            if hi > lo > 0:
                ylim = (lo * 0.7, hi * 1.5)
    else:
        # linear / symlog 默认按 P99 裁剪，避免个别极端离群值（如
        # peak_fall_time ~30s、归一化后 ÷W1 偏小 → 某点 60×）把
        # sharey 的其它箱体压成贴近 0 的细线。
        q = float(y_clip_quantile) if y_clip_quantile is not None else 0.99
        if flat.size >= 5:
            hi = float(np.percentile(flat, q * 100))
            lo = float(np.percentile(flat, (1 - q) * 100))
            if hi > lo:
                pad = 0.05 * (hi - lo)
                ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 3, figsize=(max(11, 0.6 * n_r * 2.2 + 2), 5.6),
                             sharey=True)

    week_titles = []
    week_picker = []  # callable(r) -> week_label
    for w in weeks_to_show:
        if w == "last":
            def _last_for(r):
                sub = df[(df["rabbit"] == r) & df[feature].notna()]
                if sub.empty:
                    return None
                return max(sub["week"].unique(), key=lambda x: int(x[1:]))
            week_titles.append("Each rabbit's last week")
            week_picker.append(("last", _last_for))
        else:
            week_titles.append(w.upper())
            week_picker.append((w, lambda r, _w=w: _w))

    for ax, (wk_tag, picker), wt in zip(axes, week_picker, week_titles):
        positions = []
        cursor = 0
        block_separators = []
        # 按组块排列，组之间插入一个空位置
        rabbit_pos = {}
        for bi, (g, rs) in enumerate(group_blocks):
            if bi > 0:
                cursor += 1
                block_separators.append(cursor - 0.5)
            for r in rs:
                rabbit_pos[r] = cursor
                positions.append(cursor)
                cursor += 1

        for r in rabbit_order:
            week_label = picker(r)
            pos = rabbit_pos[r]
            if week_label is None:
                continue
            vals = df[(df["rabbit"] == r) &
                      (df["week"] == week_label)][feature].dropna().to_numpy()
            vals = _normalize_vals(r, vals)
            # log 模式下过滤非正值（log 无定义）
            if yscale == "log" and vals.size:
                vals = vals[vals > 0]
            if vals.size < min_peaks:
                # 样本太少，仅画散点
                if vals.size > 0:
                    ax.scatter([pos] * vals.size, vals,
                               color=colors_rabbit.get(r, "k"),
                               marker=markers.get(r, "o"),
                               s=40, edgecolor="black", linewidth=0.4,
                               alpha=0.85, zorder=4)
                continue
            bp = ax.boxplot([vals], positions=[pos], widths=0.7,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.4))
            face = colors_rabbit.get(r, (0.5, 0.5, 0.5))
            for box in bp["boxes"]:
                box.set_facecolor(face)
                box.set_alpha(0.45)
                box.set_edgecolor("#222")
            for w_ in bp["whiskers"]:
                w_.set_color("#333"); w_.set_linewidth(0.9)
            for c_ in bp["caps"]:
                c_.set_color("#333")
            # 叠加少量间隔散点（最多 30 个，避块拥挤）
            sub_v = vals if vals.size <= 30 else \
                np.random.default_rng(seed=hash((r, wk_tag)) & 0xffff).choice(vals, 30, replace=False)
            rng2 = np.random.default_rng(seed=(hash((r, wk_tag, "j")) & 0xffff))
            jit = rng2.uniform(-0.18, 0.18, size=sub_v.size)
            ax.scatter(np.full_like(jit, pos) + jit, sub_v,
                       color=face, marker=markers.get(r, "o"),
                       s=14, edgecolor="black", linewidth=0.25,
                       alpha=0.55, zorder=3)
            # 在箱上方加上 n=...；y 位置使用「上须」估计而非 nanmax，
            # 避免个别极端离群值把 y 轴掌到顶部。
            try:
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr
                inside = vals[vals <= upper]
                anno_y = float(inside.max()) if inside.size else float(q3)
            except Exception:
                anno_y = float(np.nanmedian(vals))
            ax.text(pos, anno_y, f"n={vals.size}",
                    fontsize=6, ha="center", va="bottom", color="#444")

        for sep in block_separators:
            ax.axvline(sep, color="#888", lw=0.7, ls=":", alpha=0.7)

        # x 轴标签
        x_ticks = [rabbit_pos[r] for r in rabbit_order]
        x_labels = [f"{prefix[group_for[r]]}-{r}" for r in rabbit_order]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(wt, fontsize=10)
        ax.grid(axis="y", alpha=0.25, which="both")
        if yscale == "log":
            ax.set_yscale("log")
        elif yscale == "symlog":
            ref = np.nanmedian(np.abs(flat)) if flat.size else 1.0
            ax.set_yscale("symlog", linthresh=max(ref / 10, 1e-6))
        if ylim is not None:
            ax.set_ylim(*ylim)

    y_label = ylabel or feature
    if normalize == "per_rabbit_w1":
        y_label = y_label + "  (÷ per-rabbit W1 median)"
        for ax in axes:
            ax.axhline(1.0, color="k", lw=0.7, ls=":", alpha=0.5)
    if yscale in ("log", "symlog"):
        y_label = y_label + f"  [{yscale}]"
    axes[0].set_ylabel(y_label)
    fig.suptitle(f"{title}  —  per-rabbit per-peak 分布", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def heatmap_feature(df: pd.DataFrame, feature: str, title: str, save_path):
    """rabbit × week 矩阵热图；normal → pathological → sham 顺序。"""
    pivot = df.pivot(index="rabbit", columns="week", values=feature)
    weeks = sorted(pivot.columns, key=lambda x: int(x[1:]))
    pivot = pivot[weeks]
    # 排序：normal → pathological → infection → sham
    from .config import GROUPS
    order = (GROUPS["normal"] + GROUPS["pathological"]
             + GROUPS.get("infection", []) + GROUPS.get("sham", []))
    pivot = pivot.reindex([r for r in order if r in pivot.index])
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(pivot) + 1.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(weeks))); ax.set_xticklabels(weeks)
    ax.set_yticks(range(len(pivot)))
    def _tag(r):
        if r in GROUPS["normal"]:
            return "N"
        if r in GROUPS["pathological"]:
            return "P"
        if r in GROUPS.get("infection", []):
            return "I"
        return "S"
    yt = [f"{r} ({_tag(r)})" for r in pivot.index]
    ax.set_yticklabels(yt)
    ax.set_title(f"{title}\n(N=normal, P=pathological, I=infection, S=sham)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    # 在各组之间加分界线
    n_normal = sum(1 for r in pivot.index if r in GROUPS["normal"])
    n_path = sum(1 for r in pivot.index if r in GROUPS["pathological"])
    n_infect = sum(1 for r in pivot.index if r in GROUPS.get("infection", []))
    if 0 < n_normal < len(pivot):
        ax.axhline(n_normal - 0.5, color="white", lw=2)
    if 0 < n_normal + n_path < len(pivot):
        ax.axhline(n_normal + n_path - 0.5, color="white", lw=2)
    if 0 < n_normal + n_path + n_infect < len(pivot):
        ax.axhline(n_normal + n_path + n_infect - 0.5, color="white", lw=2)
    _add_feature_description(fig, feature)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
