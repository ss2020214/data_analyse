# -*- coding: utf-8 -*-
"""
验证脚本：根据 068_094_030_scatter_trend_points.xlsx 中的数据绘制两张散点趋势图。

与 plot_068_094_030_scatter_trend.py 不同，本脚本不再重新跑选点流程，
数据 100% 来自 Excel 两个 Sheet（Free_Scatter_Trend / Standard_Scatter_Trend），
复用它内部的绘图函数与样式，输出带 _from_excel 后缀的图片，便于人工核对
Excel 与图面数据是否一致。

- x 坐标：使用 Excel 列 x_week_idx + 兔子固定偏移（与图中散点位置一致）
- y 坐标：delta_r_r0
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis.plot_068_094_030_scatter_trend import (  # noqa: E402
    RABBITS,
    plot_three_rabbits_scatter_trend,
)

XLSX = _ROOT / "output" / "tables" / "068_094_030_scatter_trend_points.xlsx"
OUT_DIR = _ROOT / "output" / "figures" / "delta_r_r0" / "optimized_v3" / "3d"


def load_from_excel(sheet_name: str) -> dict:
    """按兔子分组读取 Excel 的某个 Sheet。

    注意：Excel 中 rabbit 列以数值形式保存（68/94/30，丢前导零），
    读取后需补零为 3 位字符串以匹配 RABBITS。
    """
    df = pd.read_excel(XLSX, sheet_name=sheet_name)
    df["rabbit"] = df["rabbit"].astype(int).astype(str).str.zfill(3)
    return {rabbit: df[df["rabbit"] == rabbit].copy() for rabbit in RABBITS}


def main() -> None:
    print("=" * 60)
    print("从 Excel 数据绘制验证图 (068 / 094 / 030)")
    print("=" * 60)

    free_data = load_from_excel("Free_Scatter_Trend")
    std_data = load_from_excel("Standard_Scatter_Trend")

    for rabbit in RABBITS:
        print(f"  {rabbit}: free={len(free_data[rabbit])}, "
              f"standard={len(std_data[rabbit])}")

    print("\n绘制自由运动散点趋势图 (来自 Excel)...")
    plot_three_rabbits_scatter_trend(
        free_data, "自由运动 (Excel 验证)",
        str(OUT_DIR / "068_094_030_free_scatter_trend_from_excel.png"))

    print("\n绘制标准运动散点趋势图 (来自 Excel)...")
    plot_three_rabbits_scatter_trend(
        std_data, "标准运动 (Excel 验证)",
        str(OUT_DIR / "068_094_030_standard_scatter_trend_from_excel.png"))

    print("\n完成!")


if __name__ == "__main__":
    main()