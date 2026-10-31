# -*- coding: utf-8 -*-
"""
导出 delta_r_r0_{rabbit}_standard_iqr_weekly.png 的绘图数据点至 Excel。

default rabbit = 030；支持命令行指定任意兔子（如 050）。

数据来源与绘图脚本 analysis/plot_030_standard_iqr_weekly.py 完全一致：
复用其数据提取/构建/统计函数（含 delta_r_r0_plot._extract_standard_motion_peaks），
保证导出的点可以单独完美重绘该图。

Excel（output/tables/delta_r_r0_{rabbit}_standard_iqr_weekly.xlsx）包含：
- scatter_points: 散点（week, time_s, x, delta_r_r0, y_pct，即入选的峰点）
- iqr_band:        IQR 带平滑曲线（x_dense, q1_pct, q3_pct）
- mean_curve:      均值平滑曲线（x_dense, mean_pct）
- README:          绘图参数说明

验证：
1. 从 Excel 读回并与内存数值逐点比对（应 < 1e-12）
2. 用读回数据按原图样式重绘并保存 *_from_excel.png
3. 与原 PNG 做像素级对比（diff 像素数应为 0）

用法：
    python -m analysis.export_030_standard_iqr_weekly        # 默认 030
    python -m analysis.export_030_standard_iqr_weekly 050   # 导出 050
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .config import OUT_TAB, OUT_FIG
from .plot_030_standard_iqr_weekly import (
    _build_weekly_positions,
    _compute_iqr_band,
    _compute_mean_curve,
)
from .delta_r_r0_plot import _extract_standard_motion_peaks

# 原图路径与导出路径
def _paths(rabbit: str = "030") -> tuple[Path, Path, Path]:
    """返回 (原图, Excel, 重绘图) 路径。"""
    base = OUT_FIG / "delta_r_r0" / "optimized_v3" / f"delta_r_r0_{rabbit}_standard_iqr_weekly"
    return (Path(f"{base}.png"),
            OUT_TAB / f"delta_r_r0_{rabbit}_standard_iqr_weekly.xlsx",
            Path(f"{base}_from_excel.png"))

SCALE = 100.0  # 与原脚本一致：转为百分比


def compute_plot_data(rabbit: str = "030") -> dict:
    """完全复用绘图脚本的计算逻辑，返回图中全部数据。"""
    df_std = _extract_standard_motion_peaks(rabbit)
    df_plot = _build_weekly_positions(df_std)
    iqr_data = _compute_iqr_band(df_plot)
    mean_data = _compute_mean_curve(df_plot)

    scatter = pd.DataFrame({
        "week": df_plot["week"].values,
        "time_s": df_plot["time_s"].values,
        "x": df_plot["x"].values,
        "delta_r_r0": df_plot["delta_r_r0"].values,
        "y_pct": df_plot["delta_r_r0"].values * SCALE,
    })

    iqr_df = pd.DataFrame({
        "x_dense": iqr_data["x_dense"],
        "q1_pct": iqr_data["q1_smooth"] * SCALE,
        "q3_pct": iqr_data["q3_smooth"] * SCALE,
    })

    x_mean, m_smooth = mean_data
    mean_df = pd.DataFrame({
        "x_dense": x_mean,
        "mean_pct": m_smooth * SCALE,
    })

    return {"scatter": scatter, "iqr_band": iqr_df, "mean_curve": mean_df}


def export_excel(data: dict, out_path: Path, rabbit: str = "030") -> None:
    """将图中数据点写入 Excel。"""
    readme = pd.DataFrame({
        "item": [
            "source_figure",
            "plot_script",
            "extract_function",
            "figsize",
            "dpi",
            "savefig_kwargs",
            "xlim",
            "xticks",
            "xticklabels",
            "xlabel",
            "ylabel",
            "yaxis_format",
            "zero_line",
            "scatter_style",
            "iqr_band_style",
            "iqr_quantiles",
            "iqr_min_n_per_week",
            "mean_curve_style",
            "mean_min_n_per_week",
            "spline_kind",
            "n_grid",
            "week_x_positions",
            "scatter_sheet_usage",
            "iqr_band_sheet_usage",
            "mean_curve_sheet_usage",
        ],
        "value": [
            "delta_r_r0_{}_standard_iqr_weekly.png".format(rabbit),
            "analysis/plot_030_standard_iqr_weekly.py",
            f"delta_r_r0_plot._extract_standard_motion_peaks('{rabbit}')",
            "(10, 6)",
            "150",
            "bbox_inches='tight'",
            "(0.5, 6.5)",
            "1..6",
            "week1..week6",
            "time",
            "ΔR/R0 (%)",
            "FuncFormatter: f'{v:.1f}%'",
            "axhline(0, color='gray', lw=0.5, ls='--')",
            "c='black', s=45, alpha=0.5, edgecolors='none', zorder=2",
            "fill_between, alpha=0.5, color='#99C7CE', edgecolor='none', zorder=1",
            "Q1=0.25, Q3=0.75",
            "4 (n<4 的周不参与 IQR 带)",
            "plot, color='black', lw=2, zorder=3",
            "4 (n<4 的周不参与均值曲线)",
            "cubic (n>=3 周), 否则 linear; fill_value='extrapolate'",
            "300",
            "周中心 wi+1，周内点按 time_s 排序后均匀排布于 ±0.35",
            "scatter(x, y_pct) 绘制散点",
            "fill_between(x_dense, q1_pct, q3_pct) 绘制 IQR 带",
            "plot(x_dense, mean_pct) 绘制均值曲线",
        ],
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        data["scatter"].to_excel(writer, sheet_name="scatter_points", index=False)
        data["iqr_band"].to_excel(writer, sheet_name="iqr_band", index=False)
        data["mean_curve"].to_excel(writer, sheet_name="mean_curve", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)


def redraw_from_excel(data: dict, out_path: Path) -> None:
    """用读回的 Excel 数据，按原图样式重绘（样式代码与原脚本逐行一致）。"""
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    scatter = data["scatter"]
    iqr_df = data["iqr_band"]
    mean_df = data["mean_curve"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── IQR 阴影带 ──
    ax.fill_between(
        iqr_df["x_dense"], iqr_df["q1_pct"], iqr_df["q3_pct"],
        alpha=0.5, color="#99C7CE", edgecolor="none", zorder=1,
    )

    # ── 均值曲线 ──
    ax.plot(mean_df["x_dense"], mean_df["mean_pct"],
            color="black", lw=2, zorder=3)

    # ── 散点（统一黑色，50% 透明度，1.5× 大小）──
    ax.scatter(
        scatter["x"], scatter["y_pct"],
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify(data_computed: dict, out_xlsx: Path, redraw_png: Path,
           src_png: Path) -> None:
    """从 Excel 读回数据：数值逐点比对 + 重绘像素级对比。"""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("[验证 1/3] 从 Excel 读回并与内存数值逐点比对")
    print("=" * 60)

    sheets = pd.read_excel(out_xlsx, sheet_name=None)
    pairs = [
        ("scatter_points", "scatter",
         ["week", "time_s", "x", "delta_r_r0", "y_pct"]),
        ("iqr_band", "iqr_band", ["x_dense", "q1_pct", "q3_pct"]),
        ("mean_curve", "mean_curve", ["x_dense", "mean_pct"]),
    ]
    ok = True
    for sheet_name, key, cols in pairs:
        df_mem = data_computed[key]
        df_xl = sheets[sheet_name]
        # week 列在 Excel 中是字符串，数值列比对
        num_cols = [c for c in cols if c != "week"]
        max_diff = 0.0
        for c in num_cols:
            a = df_mem[c].to_numpy(dtype=float)
            b = df_xl[c].to_numpy(dtype=float)
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
        status = "PASS" if max_diff < 1e-12 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {sheet_name}: {len(df_xl)} 行, 最大数值差 = {max_diff:.3e}  [{status}]")

    print("\n" + "=" * 60)
    print("[验证 2/3] 用 Excel 读回数据按原样式重绘")
    print("=" * 60)
    redraw_from_excel(
        {"scatter": sheets["scatter_points"],
         "iqr_band": sheets["iqr_band"],
         "mean_curve": sheets["mean_curve"]},
        redraw_png,
    )
    print(f"  重绘图已保存: {redraw_png}")

    print("\n" + "=" * 60)
    print("[验证 3/3] 与原 PNG 像素级对比")
    print("=" * 60)
    img_orig = plt.imread(src_png)
    img_redraw = plt.imread(redraw_png)
    if img_orig.shape != img_redraw.shape:
        print(f"  [WARN] 尺寸不一致: 原图 {img_orig.shape} vs 重绘 {img_redraw.shape}")
        h = min(img_orig.shape[0], img_redraw.shape[0])
        w = min(img_orig.shape[1], img_redraw.shape[1])
        img_orig, img_redraw = img_orig[:h, :w], img_redraw[:h, :w]
    diff = np.abs(img_orig.astype(float) - img_redraw.astype(float))
    n_pixels = diff.shape[0] * diff.shape[1]
    n_diff = int(np.sum(diff.max(axis=-1) > 1e-6))
    print(f"  总像素: {n_pixels}, 差异像素: {n_diff} ({n_diff / n_pixels * 100:.4f}%)")
    if n_diff == 0:
        print("  [PASS] 像素级完全一致，Excel 数据可单独完美重绘原图")
    else:
        print("  [FAIL] 存在像素差异，请检查环境（字体/库版本）")
        ok = False

    print("\n" + ("全部验证通过 [PASS]" if ok else "存在失败项 [FAIL]"))


def main(rabbit: str = "030") -> None:
    """导出指定兔子 IQR 周图的绘图数据（默认 030）。"""
    src_png, out_xlsx, redraw_png = _paths(rabbit)
    print("=" * 60)
    print(f"导出 delta_r_r0_{rabbit}_standard_iqr_weekly.png 绘图数据点")
    print("=" * 60)

    print("[1/3] 复用绘图脚本函数计算图中数据...")
    data = compute_plot_data(rabbit)
    sc = data["scatter"]
    print(f"  散点: {len(sc)} 个")
    for week, sub in sc.groupby("week", sort=False):
        print(f"    {week}: n={len(sub)}, "
              f"mean={sub['delta_r_r0'].mean():.6f}, "
              f"Q1={sub['delta_r_r0'].quantile(0.25):.6f}, "
              f"Q3={sub['delta_r_r0'].quantile(0.75):.6f}")
    print(f"  IQR 带: {len(data['iqr_band'])} 个网格点")
    print(f"  均值曲线: {len(data['mean_curve'])} 个网格点")

    print("[2/3] 写入 Excel...")
    export_excel(data, out_xlsx, rabbit)
    print(f"  Excel 已保存: {out_xlsx}")

    print("[3/3] 验证（读回比对 + 重绘像素对比）...")
    verify(data, out_xlsx, redraw_png, src_png)


if __name__ == "__main__":
    import sys
    _rabbit = sys.argv[1] if len(sys.argv) > 1 else "030"
    main(_rabbit)
