# -*- coding: utf-8 -*-
"""
导出 optimized_v3 图中标准运动子图 (a) 的 w1/w2 散点到 Excel。

溯源：delta_r_r0_*_optimized.png (optimized_v3) 由
analysis/delta_r_r0_dual_guided.py 的 normalize_all_rabbits_v3() +
plot_dual_normalized() 绘制。顶部子图 (a) "Standard movement" 中每周的点即
all_standard[rabbit] = delta_r_r0_plot._extract_standard_motion_peaks(rabbit)，
x = time_s - min(time_s)，y = delta_r_r0（每周最多 10 个峰）。

本脚本不使用 OCR，直接复用原绘图脚本的计算逻辑重新生成完全相同的点。
整个流水线无随机采样，结果确定性可复现。

用法：
    python -m analysis.export_week2_optimized_points            # 计算并导出
    python -m analysis.export_week2_optimized_points --verify   # 独立进程重算并逐点比对 Excel
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from .config import OUT_TAB, RABBIT2GROUP

# 与用户指定的四张图对应
TARGET_RABBITS = ["035", "053", "068", "097"]
TARGET_WEEKS = ["w1", "w2"]
OUT_XLSX = OUT_TAB / "week2_optimized_v3_points.xlsx"


def _compute_points() -> dict[str, pd.DataFrame]:
    """复用绘图脚本子图 (a) 的数据逻辑，重算各兔子 w1/w2 的标准运动散点。"""
    from .delta_r_r0_plot import _extract_standard_motion_peaks

    out = {}
    for rabbit in TARGET_RABBITS:
        df_src = _extract_standard_motion_peaks(rabbit)
        frames = []
        for week in TARGET_WEEKS:
            df = df_src[df_src["week"] == week].copy()
            if df.empty:
                print(f"  [WARN] {rabbit} {week} 无标准运动数据点")
                continue
            df = df[["time_s", "delta_r_r0"]].reset_index(drop=True)
            df["time_s"] = df["time_s"].astype(float)
            df["delta_r_r0"] = df["delta_r_r0"].astype(float)
            # 与绘图子图 (a) 一致的周内相对时间 x 坐标
            df.insert(1, "time_rel_s", df["time_s"] - df["time_s"].min())
            df.insert(0, "week", week)
            frames.append(df)
        if not frames:
            out[rabbit] = pd.DataFrame()
            continue
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "group", RABBIT2GROUP[rabbit])
        df.insert(0, "rabbit", rabbit)
        out[rabbit] = df
        for week in TARGET_WEEKS:
            sub = df[df["week"] == week]["delta_r_r0"]
            if len(sub):
                print(f"  {rabbit} ({RABBIT2GROUP[rabbit]}) {week}: {len(sub)} 点, "
                      f"mean={sub.mean():.6f}, std={sub.std():.6f}")
    return out


def export():
    print("=" * 60)
    print(f"导出 optimized_v3 图标准运动子图 {TARGET_WEEKS} 散点 -> {OUT_XLSX}")
    print("=" * 60)
    data = _compute_points()

    df_all = pd.concat([d for d in data.values() if not d.empty], ignore_index=True)

    # ---- 汇总：各兔子每周及每周全部的 mean±std（std 为样本标准差 ddof=1）----
    sum_rows = []
    for week in TARGET_WEEKS:
        wk = df_all[df_all["week"] == week]
        for rabbit, df in data.items():
            vals = df["delta_r_r0"][df["week"] == week] if not df.empty else pd.Series(dtype=float)
            if len(vals) == 0:
                continue
            sum_rows.append({"rabbit": rabbit, "week": week, "n": len(vals),
                             "mean": float(vals.mean()),
                             "std": float(vals.std(ddof=1))})
        sum_rows.append({"rabbit": "ALL", "week": week, "n": len(wk),
                         "mean": float(wk["delta_r_r0"].mean()),
                         "std": float(wk["delta_r_r0"].std(ddof=1))})
    df_summary = pd.DataFrame(sum_rows)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="ALL", index=False)
        # 数据下方空一行追加统计汇总
        df_summary.to_excel(writer, sheet_name="ALL", index=False,
                            startrow=len(df_all) + 2)
        for rabbit, df in data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=rabbit, index=False)
    print("\n汇总 (delta_r_r0):")
    for r in sum_rows:
        print(f"  {r['rabbit']:>4} {r['week']}: n={r['n']:3d}  mean={r['mean']:.6f}  std={r['std']:.6f}")
    print(f"\n已保存: {OUT_XLSX} (共 {len(df_all)} 点)")
    print("可用 --verify 在独立进程中重算并逐点比对本文件。")


def verify():
    print("=" * 60)
    print(f"独立进程重算并与 {OUT_XLSX} 逐点比对")
    print("=" * 60)
    saved = pd.read_excel(OUT_XLSX, sheet_name="ALL", dtype={"rabbit": str})
    # Excel 读回可能出现 "35" 之类的形式，统一补零为 3 位编号
    saved["rabbit"] = saved["rabbit"].astype(str).str.zfill(3)
    # 汇总块的 week 列不是周名，用 week 过滤出数据行
    saved = saved[saved["week"].isin(TARGET_WEEKS)].reset_index(drop=True)
    data = _compute_points()

    ok = True
    for rabbit in TARGET_RABBITS:
        for week in TARGET_WEEKS:
            sub_saved = saved[(saved["rabbit"] == rabbit) & (saved["week"] == week)].reset_index(drop=True)
            sub_new = data.get(rabbit, pd.DataFrame())
            sub_new = sub_new[sub_new["week"] == week].reset_index(drop=True) if not sub_new.empty else sub_new
            if len(sub_saved) != len(sub_new):
                print(f"  [FAIL] {rabbit} {week}: 点数不一致 保存={len(sub_saved)} 重算={len(sub_new)}")
                ok = False
                continue
            if len(sub_new) == 0:
                continue
            d_time = float(np.max(np.abs(sub_saved["time_s"] - sub_new["time_s"])))
            d_val = float(np.max(np.abs(sub_saved["delta_r_r0"] - sub_new["delta_r_r0"])))
            flag = "OK" if (d_time < 1e-12 and d_val < 1e-12) else "FAIL"
            if flag == "FAIL":
                ok = False
            print(f"  [{flag}] {rabbit} {week}: {len(sub_new)} 点, "
                  f"max|Δtime|={d_time:.2e}, max|Δdelta_r_r0|={d_val:.2e}")
    print("\n验证通过：Excel 中的点与绘图脚本重算结果完全一致。" if ok
          else "\n验证失败：存在不一致，请检查。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        export()
