# -*- coding: utf-8 -*-
"""导出 068/094/030 散点趋势图（标准运动 + 自由运动）的采样点坐标为 Excel。

数据来源与 plot_068_094_030_scatter_trend.py 的选点逻辑完全一致：
- 标准运动: _extract_standard_motion_peaks（docs 文档法 ΔR/R₀，峰值点）
- 自由运动: normalize_all_rabbits_v3 双维度趋势引导选点（030 参照兔直放，
  068/094 分位数窗口 + 均值/std 双目标引导）

输出列说明（除原始字段外，附图上散点坐标）：
- x_week_idx: 周索引 (w1-w6 -> 0-5)，图中 x 轴刻度位置
- x_offset:   该兔子在周内的固定偏移（避免三只兔子散点重叠）
- x_plot:     x_week_idx + x_offset，即图中散点的实际 x 坐标
- delta_r_r0: 纵轴值 ΔR/R₀ (%)
- time_s:     采样点/峰在 session 内的时间（秒）
"""
from __future__ import annotations

import pandas as pd

from analysis.config import OUT_TAB, WEEKS
from analysis.delta_r_r0_plot import _extract_standard_motion_peaks
from analysis.delta_r_r0_dual_guided import normalize_all_rabbits_v3

# 与 plot_068_094_030_scatter_trend.py 保持一致
RABBITS = ["068", "094", "030"]
RABBIT_OFFSET = {
    "068": -0.08,
    "094": 0.0,
    "030": 0.08,
}

WEEK_TO_IDX = {w: i for i, w in enumerate(WEEKS)}


def _load_data():
    """选点逻辑与 plot_068_094_030_scatter_trend._load_data 相同。"""
    print("  运行双维度趋势引导选点 (v3) 以获取 optimized 自由运动数据...")
    results = normalize_all_rabbits_v3(verbose=False)
    all_normalized = results["normalized"]

    std_data, free_data = {}, {}
    for rabbit in RABBITS:
        print(f"  {rabbit}...")
        std_data[rabbit] = _extract_standard_motion_peaks(rabbit)
        free_rows = []
        for week, pts in all_normalized.get(rabbit, {}).items():
            for pt in pts:
                if not pt.get("delta_r_r0"):
                    continue
                free_rows.append({
                    "week": week,
                    "time_s": float(pt.get("time_s", 0.0)),
                    "delta_r_r0": float(pt["delta_r_r0"]),
                })
        free_data[rabbit] = pd.DataFrame(free_rows)
    return std_data, free_data


def _to_frame(data: dict) -> pd.DataFrame:
    """{rabbit: DataFrame[week, time_s, delta_r_r0]} -> 长表，附图上 x 坐标列。"""
    frames = []
    for rabbit in RABBITS:
        df = data.get(rabbit)
        if df is None or df.empty:
            continue
        d = df.copy()
        d.insert(0, "rabbit", rabbit)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df["x_week_idx"] = df["week"].map(WEEK_TO_IDX)
    df["x_offset"] = df["rabbit"].map(RABBIT_OFFSET)
    df["x_plot"] = df["x_week_idx"].astype(float) + df["x_offset"]
    return df[["rabbit", "week", "x_week_idx", "x_offset",
               "x_plot", "time_s", "delta_r_r0"]]


def main():
    print("=" * 60)
    print("导出 068/094/030 散点趋势图采样点坐标 -> Excel")
    print("=" * 60)

    print("\n加载数据...")
    std_data, free_data = _load_data()

    free_df = _to_frame(free_data)
    std_df = _to_frame(std_data)

    out_path = OUT_TAB / "068_094_030_scatter_trend_points.xlsx"
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            free_df.to_excel(writer, sheet_name="Free_Scatter_Trend", index=False)
            std_df.to_excel(writer, sheet_name="Standard_Scatter_Trend", index=False)
    except PermissionError:
        # 原文件被 Excel 等程序占用时，自动回退到带后缀的副本
        alt_path = OUT_TAB / "068_094_030_scatter_trend_points_fixed.xlsx"
        with pd.ExcelWriter(alt_path, engine="openpyxl") as writer:
            free_df.to_excel(writer, sheet_name="Free_Scatter_Trend", index=False)
            std_df.to_excel(writer, sheet_name="Standard_Scatter_Trend", index=False)
        print(f"  原文件被占用，已写入备用文件: {alt_path}")
        out_path = alt_path

    print("\n" + "-" * 60)
    print(f"已保存: {out_path}")
    print(f"  Free_Scatter_Trend      : {len(free_df)} 行")
    print(f"  Standard_Scatter_Trend  : {len(std_df)} 行")
    for name, df in [("Free", free_df), ("Standard", std_df)]:
        if df.empty:
            continue
        pivot = df.pivot_table(index="rabbit", columns="week",
                               values="delta_r_r0", aggfunc="count", fill_value=0)
        print(f"\n[{name}] 每周采样点数:")
        print(pivot.to_string())


if __name__ == "__main__":
    main()