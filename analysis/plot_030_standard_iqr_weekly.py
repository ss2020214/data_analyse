# -*- coding: utf-8 -*-
"""
030 标准运动散点图 — 带组内 IQR 带 + 均值曲线的独立图（周数 X 轴）。

- X 轴: week1–week6
- 散点: 黑色 50% 透明度
- IQR 带: #99C7CE 50% 透明度
- 均值曲线: 黑色实线
- Y 轴: 百分比
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

from .config import WEEKS, OUT_FIG
from .delta_r_r0_plot import _extract_standard_motion_peaks

# ── 字体配置（必须在 import delta_r_r0_plot 之后重置，因其会覆盖字体设置）──
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ── 输出路径 ──────────────────────────────────────────────
OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 平滑参数 ──────────────────────────────────────────────
N_GRID = 300
SPLINE_KIND = "cubic"


def _build_weekly_positions(df: pd.DataFrame) -> pd.DataFrame:
    """将每周采样点按时间先后在周内均匀排布。"""
    records = []
    for wi, week in enumerate(WEEKS):
        sub = df[df["week"] == week].sort_values("time_s")
        n = len(sub)
        if n == 0:
            continue
        week_center = wi + 1
        x_positions = np.linspace(week_center - 0.35, week_center + 0.35, n)
        for i, (_, row) in enumerate(sub.iterrows()):
            records.append({
                "week": week,
                "x": x_positions[i],
                "delta_r_r0": row["delta_r_r0"],
                "time_s": row["time_s"],
            })
    return pd.DataFrame(records)


def _compute_iqr_band(df_plot: pd.DataFrame) -> dict | None:
    """计算每周 Q1/Q3 并生成平滑 IQR 带数据。"""
    x_centers = []
    q1_vals = []
    q3_vals = []

    for wi, week in enumerate(WEEKS):
        sub = df_plot[df_plot["week"] == week]["delta_r_r0"]
        if len(sub) >= 4:
            x_centers.append(wi + 1)
            q1_vals.append(sub.quantile(0.25))
            q3_vals.append(sub.quantile(0.75))

    if len(x_centers) < 2:
        return None

    x_arr = np.array(x_centers)
    q1_arr = np.array(q1_vals)
    q3_arr = np.array(q3_vals)

    x_min = x_arr.min() - 0.35
    x_max = x_arr.max() + 0.35
    x_dense = np.linspace(x_min, x_max, N_GRID)

    if len(x_arr) >= 3:
        q1_smooth = interp1d(x_arr, q1_arr, kind=SPLINE_KIND,
                             bounds_error=False, fill_value="extrapolate")(x_dense)
        q3_smooth = interp1d(x_arr, q3_arr, kind=SPLINE_KIND,
                             bounds_error=False, fill_value="extrapolate")(x_dense)
    else:
        q1_smooth = interp1d(x_arr, q1_arr, kind="linear",
                             bounds_error=False, fill_value="extrapolate")(x_dense)
        q3_smooth = interp1d(x_arr, q3_arr, kind="linear",
                             bounds_error=False, fill_value="extrapolate")(x_dense)

    return {"x_dense": x_dense, "q1_smooth": q1_smooth, "q3_smooth": q3_smooth}


def _compute_mean_curve(df_plot: pd.DataFrame) -> tuple | None:
    """计算每周均值并生成平滑均值曲线。"""
    x_centers = []
    mean_vals = []

    for wi, week in enumerate(WEEKS):
        sub = df_plot[df_plot["week"] == week]["delta_r_r0"]
        if len(sub) >= 4:
            x_centers.append(wi + 1)
            mean_vals.append(sub.mean())

    if len(x_centers) < 2:
        return None

    x_arr = np.array(x_centers)
    m_arr = np.array(mean_vals)

    x_min = x_arr.min() - 0.35
    x_max = x_arr.max() + 0.35
    x_dense = np.linspace(x_min, x_max, N_GRID)

    if len(x_arr) >= 3:
        m_smooth = interp1d(x_arr, m_arr, kind=SPLINE_KIND,
                            bounds_error=False, fill_value="extrapolate")(x_dense)
    else:
        m_smooth = interp1d(x_arr, m_arr, kind="linear",
                            bounds_error=False, fill_value="extrapolate")(x_dense)

    return x_dense, m_smooth


def plot_030_standard_iqr_weekly():
    """主绘图函数。"""
    print("=" * 60)
    print("030 标准运动散点图 — week X轴 + IQR带 + 均值曲线")
    print("=" * 60)

    # 1. 提取数据
    print("[1/3] 提取 030 标准运动峰值数据...")
    df_std = _extract_standard_motion_peaks("030")
    print(f"  共 {len(df_std)} 个峰值点，覆盖 {df_std['week'].nunique()} 周")

    if df_std.empty:
        print("  [ERROR] 030 标准运动数据为空，无法绘图")
        return None

    # 2. 构建周数 X 轴
    print("[2/3] 构建 X 轴坐标...")
    df_plot = _build_weekly_positions(df_std)
    print(f"  构建 {len(df_plot)} 个绘图点")

    for week in WEEKS:
        sub = df_plot[df_plot["week"] == week]
        if not sub.empty:
            vals = sub["delta_r_r0"]
            print(f"    {week}: n={len(sub)}, "
                  f"mean={vals.mean():.6f}, "
                  f"Q1={vals.quantile(0.25):.6f}, "
                  f"Q3={vals.quantile(0.75):.6f}")

    # 3. 计算 IQR 带和均值曲线
    iqr_data = _compute_iqr_band(df_plot)
    mean_data = _compute_mean_curve(df_plot)

    # 4. 绘图
    print("[3/3] 绘制图表...")
    fig, ax = plt.subplots(figsize=(10, 6))
    SCALE = 100.0  # 转为百分比

    # ── IQR 阴影带 ──
    if iqr_data is not None:
        ax.fill_between(
            iqr_data["x_dense"],
            iqr_data["q1_smooth"] * SCALE,
            iqr_data["q3_smooth"] * SCALE,
            alpha=0.5,
            color="#99C7CE",
            edgecolor="none",
            zorder=1,
        )

    # ── 均值曲线 ──
    if mean_data is not None:
        x_mean, m_smooth = mean_data
        ax.plot(x_mean, m_smooth * SCALE, color="black", lw=2, zorder=3)

    # ── 散点（统一黑色，50% 透明度，1.5× 大小）──
    ax.scatter(
        df_plot["x"], df_plot["delta_r_r0"] * SCALE,
        c="black", s=45, alpha=0.5,
        edgecolors="none", zorder=2,
    )

    # ── 轴标签 ──
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel("ΔR/R0 (%)", fontsize=12)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"week{i}" for i in range(1, 7)])
    ax.set_xlim(0.5, 6.5)

    # ── Y 轴百分比格式 ──
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%")
    )

    # ── 零线参考 ──
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    fig.tight_layout()

    # 5. 保存
    save_path = OUT_DIR / "delta_r_r0_030_standard_iqr_weekly.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {save_path}")

    return save_path


if __name__ == "__main__":
    plot_030_standard_iqr_weekly()
