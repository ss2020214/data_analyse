# -*- coding: utf-8 -*-
"""
导出 optimized_v3 图 delta_r_r0_088_optimized.png 中第一周 (w1) 与第二周 (w2)
的标准运动采样点到 Excel（顶部子图 (a) "Standard movement"）。

溯源：delta_r_r0_088_optimized.png 由
analysis/delta_r_r0_dual_guided.py 的 normalize_all_rabbits_v3() +
plot_dual_normalized() 绘制。子图 (a) 中每周的点即
all_standard["088"] = delta_r_r0_plot._extract_standard_motion_peaks("088")，
x = time_s - min(time_s)（周内相对时间），y = delta_r_r0（每周最多 10 个峰）。

本脚本不使用 OCR，直接复用原绘图脚本的计算逻辑重新生成完全相同的点。
整个流水线无随机采样，结果确定性可复现。

用法：
    python -m analysis.export_088_w1_w2_standard_points            # 计算并导出
    python -m analysis.export_088_w1_w2_standard_points --verify   # 独立进程重算并逐点比对 Excel
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from .config import OUT_TAB, RABBIT2GROUP

TARGET_RABBIT = "088"
TARGET_WEEKS = ["w1", "w2"]
OUT_XLSX = OUT_TAB / f"{TARGET_RABBIT}_w1_w2_standard_points.xlsx"


def _compute_standard_points() -> dict[str, pd.DataFrame]:
    """复用绘图脚本子图 (a) 的数据逻辑，重算 088 各目标周的标准运动散点。"""
    from .delta_r_r0_plot import _extract_standard_motion_peaks

    df_all_weeks = _extract_standard_motion_peaks(TARGET_RABBIT)
    out = {}
    for week in TARGET_WEEKS:
        df = df_all_weeks[df_all_weeks["week"] == week].copy()
        if df.empty:
            print(f"  [WARN] {TARGET_RABBIT} {week} 无标准运动数据点")
            out[week] = pd.DataFrame()
            continue
        df = df[["time_s", "delta_r_r0"]].reset_index(drop=True)
        df["time_s"] = df["time_s"].astype(float)
        df["delta_r_r0"] = df["delta_r_r0"].astype(float)
        # 与绘图子图 (a) 一致的周内相对时间 x 坐标
        df.insert(1, "time_rel_s", df["time_s"] - df["time_s"].min())
        df.insert(0, "week", week)
        df.insert(0, "group", RABBIT2GROUP[TARGET_RABBIT])
        df.insert(0, "rabbit", TARGET_RABBIT)
        out[week] = df
        print(f"  {week}: {len(df)} 点, "
              f"mean={df['delta_r_r0'].mean():.6f}, std={df['delta_r_r0'].std():.6f}")
    return out


def export():
    print("=" * 60)
    print(f"导出 delta_r_r0_{TARGET_RABBIT}_optimized.png 标准运动子图 "
          f"{TARGET_WEEKS} 散点 -> {OUT_XLSX}")
    print("=" * 60)
    data = _compute_standard_points()

    df_all = pd.concat([d for d in data.values() if not d.empty], ignore_index=True)

    # ---- 汇总：各周及全部的 mean±std（std 为样本标准差 ddof=1）----
    sum_rows = []
    for week, df in data.items():
        if df.empty:
            continue
        vals = df["delta_r_r0"]
        sum_rows.append({"week": week, "n": len(vals),
                         "mean": float(vals.mean()),
                         "std": float(vals.std(ddof=1))})
    all_vals = df_all["delta_r_r0"]
    sum_rows.append({"week": "ALL", "n": len(all_vals),
                     "mean": float(all_vals.mean()),
                     "std": float(all_vals.std(ddof=1))})
    df_summary = pd.DataFrame(sum_rows)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="ALL", index=False)
        # 数据下方空一行追加统计汇总
        df_summary.to_excel(writer, sheet_name="ALL", index=False,
                            startrow=len(df_all) + 2)
        for week, df in data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=week, index=False)
    print("\n汇总 (delta_r_r0):")
    for r in sum_rows:
        print(f"  {r['week']:>4}: n={r['n']:3d}  mean={r['mean']:.6f}  std={r['std']:.6f}")
    print(f"\n已保存: {OUT_XLSX} (共 {len(df_all)} 点)")
    print("可用 --verify 在独立进程中重算并逐点比对本文件。")


def verify():
    print("=" * 60)
    print(f"独立进程重算并与 {OUT_XLSX} 逐点比对")
    print("=" * 60)
    saved = pd.read_excel(OUT_XLSX, sheet_name="ALL", dtype={"rabbit": str})
    # 汇总块与数据块之间有空行，用 week 过滤出数据行
    saved = saved[saved["week"].isin(TARGET_WEEKS)].reset_index(drop=True)
    data = _compute_standard_points()

    ok = True
    for week in TARGET_WEEKS:
        sub_saved = saved[saved["week"] == week].reset_index(drop=True)
        sub_new = data.get(week, pd.DataFrame()).reset_index(drop=True)
        if len(sub_saved) != len(sub_new):
            print(f"  [FAIL] {week}: 点数不一致 保存={len(sub_saved)} 重算={len(sub_new)}")
            ok = False
            continue
        d_time = float(np.max(np.abs(sub_saved["time_s"] - sub_new["time_s"]))) if len(sub_new) else 0.0
        d_val = float(np.max(np.abs(sub_saved["delta_r_r0"] - sub_new["delta_r_r0"]))) if len(sub_new) else 0.0
        flag = "OK" if (d_time < 1e-12 and d_val < 1e-12) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  [{flag}] {week}: {len(sub_new)} 点, "
              f"max|Δtime|={d_time:.2e}, max|Δdelta_r_r0|={d_val:.2e}")
    print("\n验证通过：Excel 中的点与绘图脚本重算结果完全一致。" if ok
          else "\n验证失败：存在不一致，请检查。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        export()
