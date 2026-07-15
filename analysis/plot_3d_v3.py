# -*- coding: utf-8 -*-
"""
optimized_v3 数据 3D 可视化。

基于 v3 归一化后的数据，绘制三个三维散点图：
  1. 自由运动（原始采样点，归一化前）
  2. 自由运动（v3 双维度归一化后）
  3. 自由运动峰值（活跃窗口内生理学峰值点）

X 轴：兔子编号
Y 轴：周数（1-6）
Z 轴：ΔR/R₀
每个兔子一种颜色，散点图带有组内 IQR（四分位距）带。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
# Windows 兼容中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from .config import GROUPS, WEEKS, OUT_FIG, RABBIT2GROUP, ALL_RABBITS

# ── 输出目录 ─────────────────────────────────────────────
OPT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
REPORT_DIR = OPT_DIR.parent / "reports"
OUT_3D_DIR = OPT_DIR / "3d"
OUT_3D_DIR.mkdir(parents=True, exist_ok=True)

# ── 配色：每个兔子独立颜色 ──────────────────────────────
# 用 tab20 为 15 只兔子分配颜色
RABBIT_COLORS = {}
cmap = plt.get_cmap("tab20", len(ALL_RABBITS))
for i, rabbit in enumerate(ALL_RABBITS):
    RABBIT_COLORS[rabbit] = cmap(i)

# 组级别柔和底色（用于 IQR 带）
GROUP_ALPHA_COLORS = {
    "normal":       (0.12, 0.47, 0.71, 0.18),   # 蓝
    "pathological": (1.00, 0.50, 0.05, 0.18),   # 橙
    "infection":    (0.17, 0.63, 0.17, 0.18),   # 绿
    "sham":         (0.84, 0.15, 0.16, 0.18),   # 红
}


# ══════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════

def _load_free_motion_raw() -> pd.DataFrame:
    """加载自由运动原始数据（归一化前的 original 列）。

    返回 DataFrame: rabbit, group, week_num, delta_r_r0
    """
    path = REPORT_DIR / "normalization_details.csv"
    if not path.exists():
        print(f"  [WARN] 找不到 {path}")
        return pd.DataFrame(columns=["rabbit", "group", "week_num", "delta_r_r0"])

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["week_num"] = df["week"].str.extract(r"(\d+)").astype(int)

    result = df[["rabbit", "group", "week_num", "original"]].copy()
    result.rename(columns={"original": "delta_r_r0"}, inplace=True)
    result = result[result["delta_r_r0"].notna() & (result["delta_r_r0"] > 0)]
    return result.reset_index(drop=True)


def _load_free_motion_normalized() -> pd.DataFrame:
    """加载自由运动 v3 归一化后数据。

    规则：
    - action == "replaced" → 用 closest
    - action 含 "kept"      → 用 original（保持不变）
    """
    path = REPORT_DIR / "normalization_details.csv"
    if not path.exists():
        print(f"  [WARN] 找不到 {path}")
        return pd.DataFrame(columns=["rabbit", "group", "week_num", "delta_r_r0"])

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["week_num"] = df["week"].str.extract(r"(\d+)").astype(int)

    # 根据 action 决定有效值
    is_replaced = df["action"].str.lower().str.startswith("replaced")
    df["delta_r_r0"] = np.where(is_replaced, df["closest"], df["original"])

    result = df[["rabbit", "group", "week_num", "delta_r_r0"]].copy()
    result = result[result["delta_r_r0"].notna() & (result["delta_r_r0"] > 0)]
    return result.reset_index(drop=True)


def _load_free_motion_peaks() -> pd.DataFrame:
    """加载自由运动峰值数据（peaks_per_session.csv）。

    返回 DataFrame: rabbit, group, week_num, delta_r_r0
    """
    from .config import OUT_TAB
    path = OUT_TAB / "peaks_per_session.csv"
    if not path.exists():
        print(f"  [WARN] 找不到 {path}")
        return pd.DataFrame(columns=["rabbit", "group", "week_num", "delta_r_r0"])

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["week_num"] = df["week"].str.extract(r"(\d+)").astype(int)

    result = df[["rabbit", "group", "week_num", "amp_rel_baseline"]].copy()
    result.rename(columns={"amp_rel_baseline": "delta_r_r0"}, inplace=True)
    result = result[result["delta_r_r0"].notna() & (result["delta_r_r0"] > 0)]
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# 2. 3D 绘图辅助
# ══════════════════════════════════════════════════════════

def _compute_group_iqr(df: pd.DataFrame) -> dict:
    """计算每组每周的 IQR 统计量。

    返回: {(group, week_num): {"q1": ..., "q3": ..., "median": ..., "mean": ...}}
    """
    stats = {}
    for (group, week_num), grp in df.groupby(["group", "week_num"]):
        vals = grp["delta_r_r0"].dropna().to_numpy()
        if len(vals) < 4:
            continue
        stats[(group, week_num)] = {
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "n": len(vals),
        }
    return stats


def _draw_iqr_band(ax, df: pd.DataFrame, group_order: list[str],
                    rabbit_to_x: dict, group_rabbit_x_ranges: dict):
    """在 3D 轴上绘制组内 IQR 半透明带。

    为每组每周绘制一个矩形面：
    - X 范围：该组兔子在 X 轴的起止
    - Y：周数
    - Z：Q1 到 Q3
    """
    iqr_stats = _compute_group_iqr(df)

    for group_name in group_order:
        x_min, x_max = group_rabbit_x_ranges.get(group_name, (0, 0))
        if x_min >= x_max:
            continue

        color = GROUP_ALPHA_COLORS.get(group_name, (0.5, 0.5, 0.5, 0.15))

        for week_num in range(1, 7):
            key = (group_name, week_num)
            if key not in iqr_stats:
                continue
            s = iqr_stats[key]
            y = week_num
            z1, z2 = s["q1"], s["q3"]

            # 绘制 IQR 矩形（半透明面）
            verts = [
                [(x_min, y, z1), (x_max, y, z1),
                 (x_max, y, z2), (x_min, y, z2)],
            ]
            band = Poly3DCollection(verts, alpha=color[3], facecolor=color[:3],
                                     edgecolor=color[:3], linewidths=0.5)
            ax.add_collection3d(band)

            # 绘制中位线
            ax.plot([x_min, x_max], [y, y], [s["median"], s["median"]],
                    color=color[:3], lw=1.2, alpha=0.7, zorder=5)


# ══════════════════════════════════════════════════════════
# 3. 单图绘制入口
# ══════════════════════════════════════════════════════════

def plot_3d_scatter(df: pd.DataFrame, title: str, save_name: str,
                     group_order: list[str] | None = None):
    """绘制单个 3D 散点图。

    参数：
        df: 含 rabbit, group, week_num, delta_r_r0
        title: 图表标题
        save_name: 保存文件名（不含扩展名）
        group_order: 组顺序（控制 X 轴排列）
    """
    if df.empty:
        print(f"  [SKIP] {title}: 无数据")
        return

    if group_order is None:
        group_order = ["normal", "pathological", "infection", "sham"]

    # ── 建立 X 轴映射 ──
    # X 轴上按组顺序排列，组间留空隙
    rabbit_to_x = {}
    group_rabbit_x_ranges = {}
    x_current = 0
    x_labels = []
    x_positions = []

    for group_name in group_order:
        rabbits = [r for r in ALL_RABBITS if RABBIT2GROUP.get(r) == group_name]
        if not rabbits:
            continue
        g_start = x_current
        for rabbit in rabbits:
            rabbit_to_x[rabbit] = x_current
            x_labels.append(rabbit)
            x_positions.append(x_current)
            x_current += 1
        g_end = x_current - 1
        group_rabbit_x_ranges[group_name] = (g_start, g_end)
        x_current += 0.8  # 组间间距

    # ── 过滤有效行 ──
    df_plot = df[df["rabbit"].isin(rabbit_to_x)].copy()
    df_plot["x"] = df_plot["rabbit"].map(rabbit_to_x)

    # 添加微小抖动避免完全重叠
    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.22, 0.22, size=len(df_plot))
    jitter_y = rng.uniform(-0.15, 0.15, size=len(df_plot))
    df_plot["x_jit"] = df_plot["x"] + jitter_x
    df_plot["y_jit"] = df_plot["week_num"] + jitter_y

    # ── 创建 3D 图 ──
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")

    # ── 绘制组内 IQR 带 ──
    _draw_iqr_band(ax, df_plot, group_order, rabbit_to_x, group_rabbit_x_ranges)

    # ── 绘制散点 ──
    for rabbit in ALL_RABBITS:
        sub = df_plot[df_plot["rabbit"] == rabbit]
        if sub.empty:
            continue
        color = RABBIT_COLORS.get(rabbit, "gray")
        ax.scatter(sub["x_jit"], sub["y_jit"], sub["delta_r_r0"],
                   c=[color], s=18, alpha=0.55, edgecolors="none",
                   label=rabbit, zorder=10)

    # ── 轴标签 ──
    ax.set_xlabel("Rabbit", fontsize=11, labelpad=10)
    ax.set_ylabel("Week", fontsize=11, labelpad=10)
    ax.set_zlabel("Delta R / R0", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # X 轴刻度
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=30)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"W{i}" for i in range(1, 7)])

    # 图例 — mplot3d 需手动整理，排除重复标签
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=8, ncol=4, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), framealpha=0.7)

    # 视角
    ax.view_init(elev=22, azim=-48)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08)
    save_path = OUT_3D_DIR / f"{save_name}.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════
# 4. 主入口
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("optimized_v3 数据 3D 可视化")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[1/4] 加载自由运动原始数据...")
    df_raw = _load_free_motion_raw()
    print(f"      共 {len(df_raw)} 个点")

    print("\n[2/4] 加载自由运动 v3 归一化数据...")
    df_norm = _load_free_motion_normalized()
    print(f"      共 {len(df_norm)} 个点")

    print("\n[3/4] 加载自由运动峰值数据...")
    df_peaks = _load_free_motion_peaks()
    print(f"      共 {len(df_peaks)} 个点")

    # ── 绘制 3 张 3D 图 ──
    print("\n[4/4] 绘制 3D 散点图...")

    plot_3d_scatter(
        df_raw,
        title="自由运动 — 原始采样点 (normalization 前)",
        save_name="3d_free_motion_raw",
    )

    plot_3d_scatter(
        df_norm,
        title="自由运动 — v3 双维度归一化后",
        save_name="3d_free_motion_v3_normalized",
    )

    plot_3d_scatter(
        df_peaks,
        title="自由运动峰值 — 活跃窗口内生理学峰值点",
        save_name="3d_free_motion_peaks",
    )

    print("\n" + "=" * 60)
    print(f"完成! 图表已保存到: {OUT_3D_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
