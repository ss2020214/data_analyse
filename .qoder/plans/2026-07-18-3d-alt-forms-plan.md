# 3D 替代形式实现计划

> **For agentic workers:** 本计划可逐任务执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 新建 `analysis/plot_3d_alt_forms.py`,为 094/068/030 三兔的三类数据(standard/free/peaks)生成 surface / wireframe / violin_box 三种 3D 形式共 9 张图。

**Architecture:** 单文件脚本,复用 `plot_3d_three_rabbits.py` 的数据加载(`_load_data`/`_to_df`)与字体/配色工具,新增统计预处理与三个绘制函数。每兔独立趋势面,不做跨兔插值。`scipy.interpolate.CubicSpline` 做周方向加密。

**Tech Stack:** matplotlib(mplot3d)、numpy、pandas、scipy.interpolate.CubicSpline

**验证策略:** 项目无 pytest 框架,采用"运行脚本 + `python -c` 内联断言 + 检查输出文件"作为验证。

**设计依据:** [2026-07-18-3d-alt-forms-design.md](./2026-07-18-3d-alt-forms-design.md)

---

## 文件结构

- **Create:** `analysis/plot_3d_alt_forms.py` — 单文件,含常量、复用导入、预处理、三模式绘制、主入口

## 复用来源(plot_3d_three_rabbits.py)

- `_load_data()` — 返回 `(std, free, peaks)` 三个 `{rabbit: DataFrame}` 字典
- `_to_df(data_dict)` — 转为 DataFrame(rabbit, week_num, delta_r_r0)
- `_setup_chinese_font()` — 中文字体
- `_desaturate_hex(hex, blend)` — 低饱和色

注意:`plot_3d_three_rabbits.RABBITS = ["068","094","030"]`,但本脚本按 094→068→030 排列 X 轴(数据加载顺序不影响内容,仅排列顺序由本脚本控制)。

---

## Task 1: 脚本骨架 + 常量 + 归一化

**Files:**
- Create: `analysis/plot_3d_alt_forms.py`

- [ ] **Step 1: 创建脚本骨架与常量**

写入 `analysis/plot_3d_alt_forms.py`:

```python
# -*- coding: utf-8 -*-
"""
3D 替代形式可视化:094 / 068 / 030 三兔的 surface / wireframe / violin_box 三模式。
复用 plot_3d_three_rabbits 的数据加载,每兔独立趋势面,不做跨兔插值。
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
```

- [ ] **Step 2: 实现每兔归一化**

在同一文件追加:

```python
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
```

- [ ] **Step 3: 验证归一化**

运行:
```
cd d:\desktop\data_analyse\p1; python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from analysis.plot_3d_alt_forms import _load_data,_to_df,_normalize_per_rabbit,RABBITS; std,free,peaks=_load_data(); df=_normalize_per_rabbit(_to_df(free)); import numpy as np; [print(r, float(df[df.rabbit==r].delta_r_r0.max())) for r in RABBITS]"
```
Expected: 每只兔子最大值均为 `100.0`

- [ ] **Step 4: Commit**

```
git add analysis/plot_3d_alt_forms.py
git commit -m "新增 3D 替代形式脚本骨架与每兔归一化

- 新建 analysis/plot_3d_alt_forms.py
- 复用 plot_3d_three_rabbits 的 _load_data/_to_df/_setup_chinese_font/_desaturate_hex
- 定义常量 RABBITS(094/068/030)、RABBIT_COLORS、TYPE_LABELS、MODE_LABELS、OUT_DIR
- 实现 _normalize_per_rabbit:每兔按自身 max 归一化到 0-100%"
```

---

## Task 2: 单元统计 _compute_cell_stats + 加密曲线

**Files:**
- Modify: `analysis/plot_3d_alt_forms.py`

- [ ] **Step 1: 实现单元统计**

追加:

```python
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
        lo = max(float(vals.min()), med - 1.5 * iqr)
        hi = min(float(vals.max()), med + 1.5 * iqr)
        stats[(rabbit, int(week))] = {
            "q1": q1, "p50": med, "q3": q3,
            "whisker_lo": lo, "whisker_hi": hi, "n": len(vals),
        }
    return stats
```

- [ ] **Step 2: 实现周方向加密曲线**

追加:

```python
def _dense_curve(weeks: list[int], zs: list[float], n_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """用 CubicSpline 在周方向加密曲线。weeks 须严格递增且 >=2 个。"""
    xs = np.asarray(weeks, dtype=float)
    zs = np.asarray(zs, dtype=float)
    if len(xs) < 2:
        return xs, zs
    cs = CubicSpline(xs, zs)
    x_dense = np.linspace(xs.min(), xs.max(), n_points)
    return x_dense, cs(x_dense)
```

- [ ] **Step 3: 验证统计**

运行:
```
cd d:\desktop\data_analyse\p1; python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from analysis.plot_3d_alt_forms import _load_data,_to_df,_normalize_per_rabbit,_compute_cell_stats; std,free,peaks=_load_data(); df=_normalize_per_rabbit(_to_df(free)); st=_compute_cell_stats(df); k=('094',1); print(k, st.get(k))"
```
Expected: 打印 094 第1周的 q1/p50/q3/whisker_lo/whisker_hi/n,数值在 0-105 范围

- [ ] **Step 4: Commit**

```
git add analysis/plot_3d_alt_forms.py
git commit -m "实现单元统计与周方向加密

- 新增 _compute_cell_stats:按 (rabbit,week_num) 计算 P25/P50/P75 与 1.5×IQR 须线
- 新增 _dense_curve:用 scipy CubicSpline 在周方向加密为 60 点光滑曲线"
```

---

## Task 3: surface 模式(每兔独立趋势面 + IQR 包络)

**Files:**
- Modify: `analysis/plot_3d_alt_forms.py`

- [ ] **Step 1: 实现 _draw_surface**

追加。每兔:顶面 plot_surface(P75)半透明 + 前后侧面 Poly3DCollection 分段梯形(P25→P75)+ P50 中线:

```python
def _draw_surface(ax, stats: dict, rabbit_to_x: dict, env_colors: dict):
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

        # 后侧面:沿 Y 分段梯形(P25 底 → P75 顶)
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
```

- [ ] **Step 2: 实现统一绘图框架 plot_one 的 surface 分支**

追加:

```python
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
    _setup_chinese_font()
    if df_all.empty:
        print(f"  [SKIP] {chart_type}/{mode}: 无数据")
        return None
    rabbit_to_x = {r: i * X_SPACING for i, r in enumerate(RABBITS)}
    df_plot = _normalize_per_rabbit(df_all)
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
```

注意:`_draw_wireframe` 与 `_draw_violin_box` 在 Task 4/5 实现,本步运行 surface 时会因未定义报错——所以本步仅运行 surface 临时验证(见 Step 3,先用标准运动单图测试)。

- [ ] **Step 3: 临时验证 surface**

运行(标准运动 surface):
```
cd d:\desktop\data_analyse\p1; python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from analysis.plot_3d_alt_forms import _load_data,_to_df,plot_one; std,free,peaks=_load_data(); plot_one(_to_df(std),'standard','surface')"
```
Expected: 无报错,打印"已保存: ...alt_forms/surface/3d_094_068_030_standard_surface.png";打开图片中文正常、三道趋势墙可见

- [ ] **Step 4: Commit**

```
git add analysis/plot_3d_alt_forms.py
git commit -m "实现 surface 模式:每兔独立趋势面+IQR 包络

- 新增 _draw_surface:每兔用 CubicSpline 加密 P25/P50/P75
  顶面 plot_surface(P75)半透明光滑曲面 + 前后侧面 Poly3DCollection 分段梯形(P25→P75)
  + P50 中线,不做跨兔插值
- 新增 _setup_ax_common 统一轴/视角/盒子比例/标签配置
- 新增 plot_one 统一绘图框架,按 mode 分派 surface/wireframe/violin_box
  含图例(三兔色块)、标题格式 094 / 068 / 030 — 类型 [模式]
- 输出路径 OUT_DIR/{mode}/3d_094_068_030_{type}_{mode}.png"
```

---

## Task 4: wireframe 模式(每兔独立线框)

**Files:**
- Modify: `analysis/plot_3d_alt_forms.py`

- [ ] **Step 1: 实现 _draw_wireframe**

追加。每兔:P75 顶面 + P25 底面 plot_wireframe + P50 加粗中线:

```python
def _draw_wireframe(ax, stats: dict, rabbit_to_x: dict, env_colors: dict):
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
        # 前后侧面竖向连接线(每周一根,从 P25 到 P75)
        for i in range(0, n, max(1, n // 12)):
            for x_face in (x_pos - half_w, x_pos + half_w):
                ax.plot([x_face, x_face], [yd[i], yd[i]],
                        [z25d[i], z75d[i]], color=env_color, lw=0.6, alpha=0.6)
        # P50 中线(加粗)
        ax.plot(np.full(n, x_pos), yd, z50d, color=env_color, lw=1.6, zorder=8)
```

- [ ] **Step 2: 验证 wireframe**

运行(标准运动 wireframe):
```
cd d:\desktop\data_analyse\p1; python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from analysis.plot_3d_alt_forms import _load_data,_to_df,plot_one; std,free,peaks=_load_data(); plot_one(_to_df(std),'standard','wireframe')"
```
Expected: 无报错,生成 `.../wireframe/3d_094_068_030_standard_wireframe.png`;线框骨架清晰不遮挡后方

- [ ] **Step 3: Commit**

```
git add analysis/plot_3d_alt_forms.py
git commit -m "实现 wireframe 模式:每兔独立线框骨架

- 新增 _draw_wireframe:每兔 P75 顶面 + P25 底面 plot_wireframe
  + 前后侧面竖向连接线(间隔取样)+ P50 加粗中线
- 线框结构清晰,不遮挡后方数据"
```

---

## Task 5: violin_box 模式(每单元 3D 箱体)

**Files:**
- Modify: `analysis/plot_3d_alt_forms.py`

- [ ] **Step 1: 实现 _draw_violin_box**

追加。每 (rabbit×week) 单元:Q1-Q3 长方体(6 面 Poly3DCollection)+ 中位线 + 须线 + 须端横帽:

```python
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
    x_half = 0.25  # 箱体 X 半宽
    y_half = 0.35  # 箱体 Y 半宽
    cap_half = 0.18  # 须端横帽半宽
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
```

- [ ] **Step 2: 验证 violin_box**

运行(标准运动 violin_box):
```
cd d:\desktop\data_analyse\p1; python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from analysis.plot_3d_alt_forms import _load_data,_to_df,plot_one; std,free,peaks=_load_data(); plot_one(_to_df(std),'standard','violin_box')"
```
Expected: 无报错,生成 `.../violin_box/3d_094_068_030_standard_violin_box.png`;每兔每周可见箱体+中位线+须

- [ ] **Step 3: Commit**

```
git add analysis/plot_3d_alt_forms.py
git commit -m "实现 violin_box 模式:每单元 3D 箱体

- 新增 _box_faces:生成长方体 6 个矩形面顶点
- 新增 _draw_violin_box:每 (rabbit×week) 单元绘制 Q1-Q3 长方体(Poly3DCollection 6 面体)
  + 中位线(深色)+ 须线(1.5×IQR)+ 须端横帽
- 最忠实于每单元真实分布"
```

---

## Task 6: 主入口 + 命令行 --mode + 全量生成

**Files:**
- Modify: `analysis/plot_3d_alt_forms.py`

- [ ] **Step 1: 实现 main 与命令行参数**

追加:

```python
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
```

- [ ] **Step 2: 验证单模式生成**

运行(仅 surface,3 张图):
```
cd d:\desktop\data_analyse\p1; python -m analysis.plot_3d_alt_forms --mode surface
```
Expected: 打印 3 张 surface 图保存路径,无报错

- [ ] **Step 3: 验证全量生成**

运行(9 张图):
```
cd d:\desktop\data_analyse\p1; python -m analysis.plot_3d_alt_forms
```
Expected: 依次打印 surface(3)→wireframe(3)→violin_box(3)共 9 条保存路径,无报错

- [ ] **Step 4: 检查输出文件齐全**

运行:
```
cd d:\desktop\data_analyse\p1; powershell -c "Get-ChildItem -Recurse output\figures\delta_r_r0\optimized_v3\3d\alt_forms -Filter *.png | Measure-Object | Select-Object -ExpandProperty Count"
```
Expected: `9`

- [ ] **Step 5: Commit**

```
git add analysis/plot_3d_alt_forms.py
git commit -m "实现主入口与命令行参数

- 新增 main():默认生成全部 9 张图,支持 --mode surface|wireframe|violin_box 单模式
- CHART_SEQUENCE 控制 standard→free→peaks 顺序
- 加载 _load_data 三元组并映射到 std/free/peaks
- 运行后打印每张图路径便于挑选"
```

---

## Task 7: 视觉一致性校验 + 最终验证

**Files:**
- 无修改,仅验证

- [ ] **Step 1: 验证三模式内三图风格统一**

打开下列 9 张图人工核对配色/视角/标题格式一致:
- `alt_forms/surface/3d_094_068_030_{standard,free,peaks}_surface.png`
- `alt_forms/wireframe/3d_094_068_030_{standard,free,peaks}_wireframe.png`
- `alt_forms/violin_box/3d_094_068_030_{standard,free,peaks}_violin_box.png`

核对清单:
- [ ] 094 棕 / 068 金 / 030 蓝配色一致
- [ ] 标题格式 `094 / 068 / 030 — {中文类型} [{中文模式}]` 一致
- [ ] 视角 elev=18 / azim=-30 一致
- [ ] Z 轴 0-105、Week1-6 刻度一致
- [ ] 中文(标题/坐标)无乱码、无方块

- [ ] **Step 2: 验证不影响现有输出**

运行现有脚本确认原 3d_094_068_030_*.png 仍可正常生成:
```
cd d:\desktop\data_analyse\p1; python -m analysis.plot_3d_three_rabbits
```
Expected: 原脚本正常生成 `3d_068_094_030_*.png`(注意原脚本保存名),无报错;新脚本的 alt_forms 子目录与原文件并存,无覆盖

- [ ] **Step 3: 最终 Commit(如有微调)**

如 Step 1 发现配色/视角需微调,修改后提交:
```
git add analysis/plot_3d_alt_forms.py
git commit -m "微调 3D 替代形式视觉参数

- [具体调整项]"
```

---

## 自审清单

- [x] Spec 覆盖:设计文档 §3 三模式 → Task 3/4/5;§4 视觉规范 → _setup_ax_common 与各函数;§5 输出9图 → Task 6;§9 验证标准 → Task 7
- [x] 无占位符:所有步骤含完整可运行代码
- [x] 类型一致:_compute_cell_stats 返回 key (rabbit,week_num),三绘制函数签名统一 (ax, stats, rabbit_to_x, env_colors)
- [x] 复用一致:_load_data/_to_df/_setup_chinese_font/_desaturate_hex 均来自 plot_3d_three_rabbits
