# -*- coding: utf-8 -*-
"""
3D 替代形式可视化:094 / 068 / 030 三兔的 surface / wireframe / violin_box 三模式。

复用 plot_3d_three_rabbits 的数据加载,每兔独立趋势面,不做跨兔插值。
设计依据: .qoder/plans/2026-07-18-3d-alt-forms-design.md
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline

from .config import OUT_FIG
from .plot_3d_three_rabbits import (
    _load_data, _to_df, _setup_chinese_font, _desaturate_hex,
)

# ── 常量 ─────────────────────────────────────────────────
RABBITS = ["094", "068", "030"]  # X 轴顺序
RABBIT_COLORS = {
    "094": "#8c564b",  # 病理 - 棕
    "068": "#cd853f",  # 假手术 - 金
    "030": "#1f77b4",  # 正常 - 蓝
}
TYPE_LABELS = {
    "standard": "标准运动",
    "free":     "自由运动",
    "peaks":    "自由运动峰值",
}
MODE_LABELS = {
    "surface":    "趋势面",
    "wireframe":  "线框",
    "violin_box": "箱体",
}
OUT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3" / "3d" / "alt_forms"

X_SPACING = 1.8
Z_MAX = 105
VIEW_ELEV = 18
VIEW_AZIM = -30


# ══════════════════════════════════════════════════════════
# 1. 预处理
# ══════════════════════════════════════════════════════════

def _normalize_per_rabbit(df: pd.DataFrame) -> pd.DataFrame:
    """每兔按自身 max 归一化 delta_r_r0 到 0-100%。"""
    df = df.copy()
    for rabbit in RABBITS:
        mask = df["rabbit"] == rabbit
        vals = df.loc[mask, "delta_r_r0"]
        if vals.empty:
            continue
        vmax = float(vals.max())
        if vmax > 0:
            df.loc[mask, "delta_r_r0"] = vals / vmax * 100
    return df


def _compute_cell_stats(df: pd.DataFrame) -> dict:
    """每个 (rabbit, week_num) 单元的 P25/P50/P75 与须线。

    返回: {(rabbit, week_num): {q1, p50, q3, whisker_lo, whisker_hi, n}}
    """
    stats: dict[tuple[str, int], dict] = {}
    for (rabbit, week), grp in df.groupby(["rabbit", "week_num"]):
        vals = grp["delta_r_r0"].dropna().to_numpy()
        vals = vals[vals > 0]
        if len(vals) < 2:
            continue
        q1 = float(np.percentile(vals, 25))
        med = float(np.percentile(vals, 50))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - q1
        # 标准 Tukey 箱线图须线:以铰链 Q1/Q3 为基准
        lo = max(float(vals.min()), q1 - 1.5 * iqr)
        hi = min(float(vals.max()), q3 + 1.5 * iqr)
        stats[(rabbit, int(week))] = {
            "q1": q1, "p50": med, "q3": q3,
            "whisker_lo": lo, "whisker_hi": hi, "n": len(vals),
        }
    return stats


def _dense_curve(weeks: list[int], zs: list[float], n_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """用 CubicSpline 在周方向加密曲线。weeks 须严格递增且 >=2 个。"""
    xs = np.asarray(weeks, dtype=float)
    zs = np.asarray(zs, dtype=float)
    if len(xs) < 2:
        return xs, zs
    cs = CubicSpline(xs, zs)
    x_dense = np.linspace(xs.min(), xs.max(), n_points)
    return x_dense, cs(x_dense)


# ══════════════════════════════════════════════════════════
# 2. 绘制函数
# ══════════════════════════════════════════════════════════

def _draw_surface(ax, stats: dict, rabbit_to_x: dict, env_colors: dict):
    """每兔独立趋势面:顶面 plot_surface(P75) + 前后侧面 Poly3DCollection 分段梯形 + P50 中线。"""
    half_w = 0.35
    for rabbit, x_pos in rabbit_to_x.items():
        weeks = sorted([w for (r, w) in stats if r == rabbit])
        if len(weeks) < 2:
            continue
        z25 = [stats[(rabbit, w)]["q1"] for w in weeks]
        z50 = [stats[(rabbit, w)]["p50"] for w in weeks]
        z75 = [stats[(rabbit, w)]["q3"] for w in weeks]
        yd, z25d = _dense_curve(weeks, z25)
        _, z50d = _dense_curve(weeks, z50)
        _, z75d = _dense_curve(weeks, z75)
        env_color = env_colors[rabbit]
        n = len(yd)

        # 顶面(P75)光滑曲面
        X_grid, Y_grid = np.meshgrid([x_pos - half_w, x_pos + half_w], yd)
        Z_top = np.tile(z75d[:, None], (1, 2))
        ax.plot_surface(X_grid, Y_grid, Z_top, alpha=0.35, color=env_color,
                        linewidth=0, antialiased=True, shade=False)

        # 前后侧面:沿 Y 分段梯形(P25 底 → P75 顶)
        side_verts = []
        for i in range(n - 1):
            y1, y2 = yd[i], yd[i + 1]
            for x_face in (x_pos - half_w, x_pos + half_w):
                side_verts.append([
                    (x_face, y1, z25d[i]),
                    (x_face, y2, z25d[i + 1]),
                    (x_face, y2, z75d[i + 1]),
                    (x_face, y1, z75d[i]),
                ])
        sides = Poly3DCollection(side_verts, alpha=0.45, facecolor=env_color,
                                 edgecolor=env_color, linewidths=0.4)
        ax.add_collection3d(sides)

        # P50 中线
        ax.plot(np.full(n, x_pos), yd, z50d, color=env_color, lw=1.8, zorder=8)


def _draw_wireframe(ax, stats: dict, rabbit_to_x: dict, env_colors: dict):
    """每兔独立线框:P75 顶面 + P25 底面 plot_wireframe + 竖向连接线 + P50 中线。"""
    half_w = 0.35
    for rabbit, x_pos in rabbit_to_x.items():
        weeks = sorted([w for (r, w) in stats if r == rabbit])
        if len(weeks) < 2:
            continue
        z25 = [stats[(rabbit, w)]["q1"] for w in weeks]
        z50 = [stats[(rabbit, w)]["p50"] for w in weeks]
        z75 = [stats[(rabbit, w)]["q3"] for w in weeks]
        yd, z25d = _dense_curve(weeks, z25)
        _, z50d = _dense_curve(weeks, z50)
        _, z75d = _dense_curve(weeks, z75)
        env_color = env_colors[rabbit]
        n = len(yd)

        X_grid, Y_grid = np.meshgrid([x_pos - half_w, x_pos + half_w], yd)
        # P75 顶面线框
        Z_top = np.tile(z75d[:, None], (1, 2))
        ax.plot_wireframe(X_grid, Y_grid, Z_top, color=env_color,
                          linewidth=0.8, alpha=0.8)
        # P25 底面线框
        Z_bot = np.tile(z25d[:, None], (1, 2))
        ax.plot_wireframe(X_grid, Y_grid, Z_bot, color=env_color,
                          linewidth=0.6, alpha=0.5)
        # 前后侧面竖向连接线(间隔取样,从 P25 到 P75)
        step = max(1, n // 12)
        for i in range(0, n, step):
            for x_face in (x_pos - half_w, x_pos + half_w):
                ax.plot([x_face, x_face], [yd[i], yd[i]],
                        [z25d[i], z75d[i]], color=env_color, lw=0.6, alpha=0.6)
        # P50 中线(加粗)
        ax.plot(np.full(n, x_pos), yd, z50d, color=env_color, lw=1.6, zorder=8)


def _box_faces(x0, x1, y0, y1, z0, z1):
    """返回长方体的 6 个矩形面(顶点顺序逆时针,法线朝外)。"""
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # 底 z0
        [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],  # 顶 z1
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # 左 x0
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],  # 右 x1
        [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],  # 前 y0
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # 后 y1
    ]


def _draw_violin_box(ax, stats: dict, rabbit_to_x: dict, env_colors: dict):
    """每 (rabbit×week) 单元:Q1-Q3 长方体 + 中位线 + 须线 + 须端横帽。"""
    x_half = 0.25
    y_half = 0.35
    cap_half = 0.18
    box_alpha = 0.6

    for rabbit, x_pos in rabbit_to_x.items():
        weeks = sorted([w for (r, w) in stats if r == rabbit])
        if not weeks:
            continue
        env_color = env_colors[rabbit]
        face_verts = []
        for w in weeks:
            s = stats[(rabbit, w)]
            q1, q3 = s["q1"], s["q3"]
            if q3 <= q1:
                continue
            x0, x1 = x_pos - x_half, x_pos + x_half
            y0, y1 = w - y_half, w + y_half
            face_verts.extend(_box_faces(x0, x1, y0, y1, q1, q3))
            # 中位线(沿 X 方向,在箱体中央 Y)
            ax.plot([x0, x1], [w, w], [s["p50"], s["p50"]],
                    color="#222222", lw=2.0, zorder=9)
            # 须线(从箱体顶/底到 whisker_hi/lo,在 X 中心)
            ax.plot([x_pos, x_pos], [w, w], [q3, s["whisker_hi"]],
                    color=env_color, lw=1.2, zorder=7)
            ax.plot([x_pos, x_pos], [w, w], [q1, s["whisker_lo"]],
                    color=env_color, lw=1.2, zorder=7)
            # 须端横帽
            ax.plot([x_pos - cap_half, x_pos + cap_half], [w, w],
                    [s["whisker_hi"], s["whisker_hi"]], color=env_color, lw=1.2)
            ax.plot([x_pos - cap_half, x_pos + cap_half], [w, w],
                    [s["whisker_lo"], s["whisker_lo"]], color=env_color, lw=1.2)
        if face_verts:
            boxes = Poly3DCollection(face_verts, alpha=box_alpha,
                                     facecolor=env_color, edgecolor=env_color,
                                     linewidths=0.6)
            ax.add_collection3d(boxes)


# ══════════════════════════════════════════════════════════
# 3. 统一绘图框架
# ══════════════════════════════════════════════════════════

def _setup_ax_common(ax, rabbit_to_x: dict, z_label: str, title: str):
    ax.set_proj_type("persp")
    ax.set_box_aspect([1, 4, 0.6])
    ax.set_xlabel("Rabbit ID", fontsize=12, labelpad=10)
    ax.set_ylabel("Time", fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=18)
    ax.set_xticks([rabbit_to_x[r] for r in RABBITS])
    ax.set_xticklabels(RABBITS, fontsize=10, rotation=25)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Week{i}" for i in range(1, 7)], fontsize=10)
    ax.set_zlim(0, Z_MAX)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def plot_one(df_all: pd.DataFrame, chart_type: str, mode: str):
    """绘制单张 3D 替代形式图。

    参数:
        df_all: 含 rabbit, week_num, delta_r_r0
        chart_type: standard / free / peaks
        mode: surface / wireframe / violin_box
    """
    _setup_chinese_font()
    if df_all.empty:
        print(f"  [SKIP] {chart_type}/{mode}: 无数据")
        return None

    rabbit_to_x = {r: i * X_SPACING for i, r in enumerate(RABBITS)}
    df_plot = _normalize_per_rabbit(df_all)
    # 防御性过滤:仅保留目标三兔(防止外部传入多兔数据污染)
    df_plot = df_plot[df_plot["rabbit"].isin(rabbit_to_x)]
    stats = _compute_cell_stats(df_plot)
    env_colors = {r: _desaturate_hex(RABBIT_COLORS[r]) for r in RABBITS}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    z_label = "Normalized ΔR/R0 (%)"
    cn_type = TYPE_LABELS[chart_type]
    cn_mode = MODE_LABELS[mode]
    title = f"094 / 068 / 030 — {cn_type} [{cn_mode}]"

    if mode == "surface":
        _draw_surface(ax, stats, rabbit_to_x, env_colors)
    elif mode == "wireframe":
        _draw_wireframe(ax, stats, rabbit_to_x, env_colors)
    elif mode == "violin_box":
        _draw_violin_box(ax, stats, rabbit_to_x, env_colors)
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 图例
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="w",
                      markerfacecolor=RABBIT_COLORS[r], markersize=10, label=r)
               for r in RABBITS]
    ax.legend(handles=handles, fontsize=9, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), framealpha=0.7)

    _setup_ax_common(ax, rabbit_to_x, z_label, title)

    out_dir = OUT_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"3d_094_068_030_{chart_type}_{mode}.png"
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════
# 4. 主入口
# ══════════════════════════════════════════════════════════

CHART_SEQUENCE = [("standard", "std"), ("free", "free"), ("peaks", "peaks")]
ALL_MODES = ["surface", "wireframe", "violin_box"]


def main():
    _setup_chinese_font()
    parser = argparse.ArgumentParser(description="3D 替代形式可视化")
    parser.add_argument("--mode", choices=ALL_MODES, default=None,
                        help="仅生成单一模式(默认全部)")
    args = parser.parse_args()
    modes = [args.mode] if args.mode else ALL_MODES

    print("=" * 60)
    print("094 / 068 / 030 三兔 3D 替代形式可视化")
    print("=" * 60)
    print("\n加载数据...")
    data = _load_data()  # (std, free, peaks) dict 字典三元组
    data_map = dict(zip(["std", "free", "peaks"], data))

    for mode in modes:
        print(f"\n--- {MODE_LABELS[mode]} ({mode}) ---")
        for chart_type, key in CHART_SEQUENCE:
            print(f"  [{chart_type}]")
            plot_one(_to_df(data_map[key]), chart_type, mode)

    print("\n" + "=" * 60)
    print(f"完成! 输出目录: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
