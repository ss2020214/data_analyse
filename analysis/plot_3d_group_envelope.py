# -*- coding: utf-8 -*-
"""
分组 3D 散点图包络线可视化。

为 normal（排除071）/pathological/infection/sham 四个实验组
分别生成三张 3D 散点图：
  1. 标准运动峰值
  2. 自由运动采样点
  3. 自由运动窗口内峰值点

X 轴：兔子编号（组内间距 >= 1.5）
Y 轴：周数（1-6）
Z 轴：ΔR/R₀（P99 截断）

每只兔子每周的数据点带有从 Z=0 到 P95 值的半透明包络填充。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib

# Windows 兼容中文字体
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC',
    'Arial Unicode MS', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from .config import GROUPS, WEEKS, OUT_FIG, RABBIT2GROUP, ALL_RABBITS

# ── 配置 ─────────────────────────────────────────────────
EXCLUDED_RABBITS = {"071"}
GROUP_ORDER = ["normal", "pathological", "infection", "sham"]

# 组名中文映射
GROUP_CN = {
    "normal":       "正常愈合组",
    "pathological": "未愈合组",
    "infection":    "感染组",
    "sham":         "空白对照组",
}

# 每只兔子独立颜色（组内色相大幅差异化，肉眼可区分）
RABBIT_COLORS: dict[str, str] = {
    # normal（蓝、橙、绿 — 色相差异 ~120°）
    "030": "#1f77b4",
    "075": "#ff7f0e",
    "020": "#2ca02c",
    # pathological（红、紫、棕、青 — 跨冷暖系）
    "043": "#d62728",
    "086": "#9467bd",
    "094": "#8c564b",
    "015": "#17becf",
    # infection（粉、橄榄、灰 — 明显差异）
    "016": "#e377c2",
    "065": "#bcbd22",
    "088": "#7f7f7f",
    # sham（深蓝、热粉、秘鲁金、深绿）
    "035": "#1a5276",
    "053": "#ff69b4",
    "068": "#cd853f",
    "097": "#006400",
}


def _desaturate_hex(hex_color: str, blend: float = 0.70) -> tuple[float, float, float]:
    """将 hex 颜色向白色插值，得到低饱和度版本用于包络填充。

    blend 越接近 1.0 颜色越淡。
    返回 0-1 范围的 RGB tuple。
    """
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = r + (255 - r) * blend
    g2 = g + (255 - g) * blend
    b2 = b + (255 - b) * blend
    return (r2 / 255.0, g2 / 255.0, b2 / 255.0)

# 图表类型中文标签
TYPE_LABELS = {
    "standard": "标准运动",
    "free":     "自由运动",
    "peaks":    "自由运动峰值",
}

# 输出目录
OPT_DIR = OUT_FIG / "delta_r_r0" / "optimized_v3"
OUT_3D_DIR = OPT_DIR / "3d" / "group"


# ══════════════════════════════════════════════════════════
# 0. 字体设置（必须在所有导入完成后调用，防止被其他模块覆盖）
# ══════════════════════════════════════════════════════════

def _setup_chinese_font():
    """确保中文字体正确渲染，抵御其他模块的 font 覆盖。

    使用 Windows 系统预装的中文字体，优先级：SimHei > Microsoft YaHei > KaiTi。
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*Font family.*not found")
    # 重建字体缓存，确保能找到系统字体
    from matplotlib.font_manager import _load_fontmanager
    try:
        _load_fontmanager(try_read_cache=False)
    except Exception:
        pass
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'DejaVu Sans',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════
# 1. 数据加载（复用已有选点函数）
# ══════════════════════════════════════════════════════════

def _load_standard_data() -> dict[str, pd.DataFrame]:
    """加载所有兔子的标准运动峰值数据。"""
    from .delta_r_r0_plot import _extract_standard_motion_peaks
    result = {}
    for rabbit in ALL_RABBITS:
        if rabbit in EXCLUDED_RABBITS:
            continue
        try:
            df = _extract_standard_motion_peaks(rabbit)
            if not df.empty:
                result[rabbit] = df
        except Exception as e:
            print(f"    [WARN] {rabbit} 标准运动提取失败: {e}")
    return result


def _load_free_motion_data() -> dict[str, pd.DataFrame]:
    """加载所有兔子的自由运动采样点数据。"""
    from .delta_r_r0_plot import _extract_free_motion_points
    result = {}
    for rabbit in ALL_RABBITS:
        if rabbit in EXCLUDED_RABBITS:
            continue
        try:
            df = _extract_free_motion_points(rabbit, n_per_week=30)
            if not df.empty:
                result[rabbit] = df
        except Exception as e:
            print(f"    [WARN] {rabbit} 自由运动提取失败: {e}")
    return result


def _load_free_peaks_data() -> dict[str, pd.DataFrame]:
    """加载所有兔子的自由运动峰值点数据。"""
    from .delta_r_r0_plot import _extract_free_motion_peaks_in_window
    result = {}
    for rabbit in ALL_RABBITS:
        if rabbit in EXCLUDED_RABBITS:
            continue
        try:
            df = _extract_free_motion_peaks_in_window(rabbit)
            if not df.empty:
                result[rabbit] = df
        except Exception as e:
            print(f"    [WARN] {rabbit} 自由运动峰值提取失败: {e}")
    return result


def _build_group_df(
    data_dict: dict[str, pd.DataFrame],
    group_name: str,
) -> pd.DataFrame:
    """将 {rabbit: DataFrame(week, time_s, delta_r_r0)} 转为统一格式。

    返回 DataFrame: rabbit, group, week_num, delta_r_r0
    """
    rabbits_in_group = [
        r for r in ALL_RABBITS
        if RABBIT2GROUP.get(r) == group_name and r not in EXCLUDED_RABBITS
    ]
    rows = []
    for rabbit in rabbits_in_group:
        df = data_dict.get(rabbit)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            week_str = str(row["week"])
            week_num = int(week_str[1:])  # "w1" → 1
            rows.append({
                "rabbit": rabbit,
                "group": group_name,
                "week_num": week_num,
                "delta_r_r0": float(row["delta_r_r0"]),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 2. 包络线绘制
# ══════════════════════════════════════════════════════════

def _draw_rabbit_envelope(
    ax,
    df_group: pd.DataFrame,
    rabbit_to_x: dict[str, float],
    rabbit_env_colors: dict[str, tuple],
):
    """为每只兔子绘制连续跨周包络填充（Z=0 → 最大值）。

    包络线是 Y-Z 平面内的 2D 填充区域：
    每只兔子在自身 X 中心位置绘制一个垂直面片，
    底部 Z=0，顶部跟随每周最大值，逐周连接形成连续幕墙。
    """
    fill_alpha = 0.55

    for rabbit in df_group["rabbit"].unique():
        x_pos = rabbit_to_x.get(rabbit)
        if x_pos is None:
            continue
        env_color = rabbit_env_colors.get(rabbit, (0.5, 0.5, 0.5))
        sub_r = df_group[df_group["rabbit"] == rabbit]

        # 收集每周最大值
        week_z: dict[int, float] = {}
        for week_num in range(1, 7):
            sub = sub_r[sub_r["week_num"] == week_num]
            if sub.empty:
                continue
            vals = sub["delta_r_r0"].dropna().to_numpy()
            vals_pos = vals[vals > 0]
            if len(vals_pos) < 2:
                continue
            z = float(np.max(vals_pos))
            if z > 0:
                week_z[week_num] = z

        if len(week_z) < 2:
            continue

        sorted_ws = sorted(week_z.keys())

        # 逐段连接相邻周，在兔子 X 中心位置绘制 Y-Z 垂直面片
        for i in range(len(sorted_ws) - 1):
            w1, w2 = sorted_ws[i], sorted_ws[i + 1]
            z1, z2 = week_z[w1], week_z[w2]
            y1, y2 = w1, w2

            face = Poly3DCollection(
                [[(x_pos, y1, 0), (x_pos, y2, 0),
                  (x_pos, y2, z2), (x_pos, y1, z1)]],
                alpha=fill_alpha, facecolor=env_color,
                edgecolor=env_color, linewidths=0.5,
            )
            ax.add_collection3d(face)


# ══════════════════════════════════════════════════════════
# 3. 单组单图绘制
# ══════════════════════════════════════════════════════════

def plot_group_3d(
    df_group: pd.DataFrame,
    group_name: str,
    chart_type: str,  # "standard" | "free" | "peaks"
):
    """为单个组绘制一张 3D 散点图。

    参数：
        df_group: 含 rabbit, group, week_num, delta_r_r0
        group_name: 组标识
        chart_type: 图表类型
    """
    _setup_chinese_font()

    if df_group.empty:
        print(f"  [SKIP] {group_name}/{chart_type}: 无数据")
        return

    rabbits_in_group = [
        r for r in ALL_RABBITS
        if RABBIT2GROUP.get(r) == group_name and r not in EXCLUDED_RABBITS
    ]
    # normal 组：020 最前，030 最后
    if group_name == "normal":
        rabbits_in_group = ["020"] + [r for r in rabbits_in_group if r not in ("020", "030")] + ["030"]
    if not rabbits_in_group:
        print(f"  [SKIP] {group_name}/{chart_type}: 组内无兔子")
        return

    # ── X 轴位置映射 ──
    rabbit_to_x: dict[str, float] = {}
    x_spacing = 1.8
    for i, rabbit in enumerate(rabbits_in_group):
        rabbit_to_x[rabbit] = i * x_spacing

    # ── 添加抖动 ──
    rng = np.random.default_rng(42)
    df_plot = df_group[df_group["rabbit"].isin(rabbit_to_x)].copy()
    df_plot["x"] = df_plot["rabbit"].map(rabbit_to_x)
    df_plot["x_jit"] = df_plot["x"] + rng.uniform(-0.30, 0.30, size=len(df_plot))
    df_plot["y_jit"] = df_plot["week_num"] + rng.uniform(-0.25, 0.25, size=len(df_plot))

    # ── Z 轴处理 ──
    df_env = df_group[df_group["rabbit"].isin(rabbit_to_x)].copy()
    if group_name == "normal":
        # normal 组：每只兔子按自身最大值归一化到 0-100%，展示相对趋势
        for rabbit in rabbits_in_group:
            mask = df_plot["rabbit"] == rabbit
            vals = df_plot.loc[mask, "delta_r_r0"]
            vmax = float(vals.max())
            if vmax > 0:
                df_plot.loc[mask, "delta_r_r0"] = vals / vmax * 100
            mask_env = df_env["rabbit"] == rabbit
            vals_env = df_env.loc[mask_env, "delta_r_r0"]
            vmax_env = float(vals_env.max())
            if vmax_env > 0:
                df_env.loc[mask_env, "delta_r_r0"] = vals_env / vmax_env * 100
        z_label = "Normalized ΔR/R0 (%)"
        z_max = 105  # 留 5% 余量
    else:
        # 其他组：原始百分比
        df_plot["delta_r_r0"] = df_plot["delta_r_r0"] * 100
        df_env["delta_r_r0"] = df_env["delta_r_r0"] * 100
        all_vals = df_plot["delta_r_r0"].dropna().to_numpy()
        all_vals_pos = all_vals[all_vals > 0]
        if len(all_vals_pos) > 0:
            z_max = float(np.percentile(all_vals_pos, 99))
            z_max = max(z_max, 0.5)
        else:
            z_max = 10.0
        z_label = "ΔR/R0 (%)"

    # 裁剪超出上限的点
    df_plot["delta_r_r0_clipped"] = df_plot["delta_r_r0"].clip(upper=z_max)

    # ── 创建 3D 图 ──
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")
    # 强制三轴比例，Y 轴拉长为 X 轴的 5 倍
    ax.set_box_aspect([1, 4, 0.6])

    # ── 包络填充（每只兔子独立颜色，低饱和度）──
    rabbit_env_colors = {
        r: _desaturate_hex(RABBIT_COLORS.get(r, "#808080"))
        for r in rabbits_in_group
    }
    _draw_rabbit_envelope(ax, df_env, rabbit_to_x, rabbit_env_colors)

    # ── 散点 ──
    for rabbit in rabbits_in_group:
        sub = df_plot[df_plot["rabbit"] == rabbit]
        if sub.empty:
            continue
        color = RABBIT_COLORS.get(rabbit, "gray")
        ax.scatter(
            sub["x_jit"], sub["y_jit"], sub["delta_r_r0_clipped"],
            c=[color], s=28, alpha=0.10, edgecolors="none",
            label=rabbit, zorder=10,
        )

    # ── 轴标签 ──
    ax.set_xlabel("Rabbit ID", fontsize=12, labelpad=10)
    ax.set_ylabel("Time", fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)

    cn_type = TYPE_LABELS.get(chart_type, chart_type)
    cn_group = GROUP_CN.get(group_name, group_name)
    ax.set_title(
        f"{cn_group} — {cn_type}",
        fontsize=14, pad=18,
    )

    # ── 轴刻度 ──
    x_ticks = [rabbit_to_x[r] for r in rabbits_in_group]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(rabbits_in_group, fontsize=10, rotation=25)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"Week{i}" for i in range(1, 7)], fontsize=10)
    ax.set_zlim(0, z_max)

    # ── 网格线 ──
    ax.grid(False)
    # 仅保留 XY 底面淡色参考线
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ── 图例 ──
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=9, ncol=len(rabbits_in_group),
        loc="upper left", bbox_to_anchor=(0.0, 1.0),
        framealpha=0.7,
    )

    # ── 视角 ──
    ax.view_init(elev=18, azim=-30)

    # ── 保存 ──
    group_dir = OUT_3D_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    save_path = group_dir / f"3d_{group_name}_{chart_type}.png"
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════
# 4. 主入口
# ══════════════════════════════════════════════════════════

def main():
    _setup_chinese_font()  # 在所有操作之前先设置字体
    print("=" * 60)
    print("分组 3D 散点图包络线可视化")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[1/3] 加载标准运动数据...")
    standard_data = _load_standard_data()
    n_std = sum(len(df) for df in standard_data.values())
    print(f"      共 {len(standard_data)} 只兔子, {n_std} 个点")

    print("\n[2/3] 加载自由运动数据...")
    free_data = _load_free_motion_data()
    n_free = sum(len(df) for df in free_data.values())
    print(f"      共 {len(free_data)} 只兔子, {n_free} 个点")

    print("\n[3/3] 加载自由运动峰值数据...")
    peaks_data = _load_free_peaks_data()
    n_peaks = sum(len(df) for df in peaks_data.values())
    print(f"      共 {len(peaks_data)} 只兔子, {n_peaks} 个点")

    # ── 为每组生成 3 张图 ──
    print("\n" + "=" * 60)
    print("生成 3D 散点图...")
    print("=" * 60)

    for group_name in GROUP_ORDER:
        print(f"\n--- {GROUP_CN.get(group_name, group_name)} ---")

        # 1) 标准运动
        df_std = _build_group_df(standard_data, group_name)
        plot_group_3d(df_std, group_name, "standard")

        # 2) 自由运动
        df_free = _build_group_df(free_data, group_name)
        plot_group_3d(df_free, group_name, "free")

        # 3) 自由运动峰值
        df_peaks = _build_group_df(peaks_data, group_name)
        plot_group_3d(df_peaks, group_name, "peaks")

    print("\n" + "=" * 60)
    print(f"完成! 图表已保存到: {OUT_3D_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
