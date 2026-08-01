# 3D 可视化替代形式设计文档

## 1. 背景与目标

项目当前为三只代表兔子(094 病理 / 068 假手术 / 030 正常)生成三张 3D 图:

- `3d_094_068_030_standard.png`(标准运动)
- `3d_094_068_030_free.png`(自由运动)
- `3d_094_068_030_peaks.png`(自由运动峰值)

现有形式为"散点云 + Poly3DCollection 垂直包络幕墙",由 [plot_3d_three_rabbits.py](../../analysis/plot_3d_three_rabbits.py) 生成。当前视觉效果对数据特征的呈现不够直观,需要探索其他 3D 展示方式。

本设计的目标:**新建独立脚本,为上述三类数据提供三种可选 3D 表现形式,三种形式内部三张图统一规范以保证跨图可比,且不破坏现有输出。**

## 2. 设计决策

### 2.1 关键决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| 实现位置 | 新建独立脚本 `analysis/plot_3d_alt_forms.py` | 不污染现有脚本,复用其数据加载函数 |
| 可视化形式 | surface / wireframe / violin_box 三模式 | 用户指定探索的三种技术 |
| 跨兔插值 | **不做** | 094/068/030 是三种不同生理状态的离散个体,跨兔插值会虚构不存在的中间态,科学上不严谨 |
| 单元统计量 | P25 / P50 / P75 / 1.5×IQR 须线 | 中位数趋势 + IQR 包络,与项目现有 IQR 包络带规范一致 |
| 归一化 | 每兔按自身 max 归一化到 0-100% | 与 plot_3d_three_rabbits.py 现有规范一致 |
| 图间关系 | 同一模式内三张图统一形式 | 保证 free/peaks/standard 跨图直接可比 |
| 输出路径 | 子目录 `alt_forms/{mode}/` | 不覆盖原 `3d_094_068_030_*.png`,便于对比挑选 |

### 2.2 数据流

```
plot_3d_alt_forms.py
   ├─ 复用 plot_3d_three_rabbits._load_data()   → {rabbit: DataFrame} × 3 类(standard/free/peaks)
   ├─ 复用 plot_3d_three_rabbits._to_df()        → 统一 DataFrame(rabbit, week_num, delta_r_r0)
   ├─ _normalize_per_rabbit(df)                 → 每兔按自身 max 归一化到 0-100%
   ├─ _compute_cell_stats(df)                    → 每个 (rabbit×week) 单元 P25/P50/P75/whisker_lo/whisker_hi
   └─ plot_one(df, chart_type, mode)            → 按 mode 分派
        ├─ _draw_surface(ax, stats, rabbit_to_x, env_colors)    # 每兔独立趋势面+IQR 包络
        ├─ _draw_wireframe(ax, stats, rabbit_to_x, env_colors)  # 每兔独立线框骨架
        └─ _draw_violin_box(ax, df, stats, rabbit_to_x, env_colors)  # 每单元 3D 箱体
```

## 3. 三种模式实现规格

### 3.1 统一预处理

- 兔子顺序:`094 → 068 → 030`
- X 间距:1.8(与现有脚本一致)
- 每兔归一化:自身 max → 100%
- Z 轴范围:0-105(留 5% 余量)
- cell 统计:按 (rabbit, week_num) groupby,计算 P25/P50/P75 与须线(标准 Tukey:lo=max(P0, Q1-1.5×IQR), hi=min(P100, Q3+1.5×IQR))

### 3.2 surface 模式(每兔独立趋势面)

对每只兔子:
- 收集 6 周的 P50 序列 → 用 `scipy.interpolate.CubicSpline` 在周方向(1~6)加密为 ~60 点光滑曲线
- 同法加密 P25、P75 曲线
- 在该兔 X 中心位置(±窄宽 0.35)用 `Poly3DCollection` 画带状面片:顶=P75,底=P25,形成 IQR 包络带(半透明 alpha=0.45)
- 中间叠加 P50 实线(线宽 1.8)
- 不在兔子之间做任何插值

视觉:三道并排的"趋势墙",呈现单兔随周的变化地貌。

### 3.3 wireframe 模式(每兔独立线框)

对每只兔子:
- 用加密后的 P50/P25/P75 三条曲线,构造 3×N 的网格(X=兔子中心±窄宽,Y=加密周,Z=统计量)
- `ax.plot_wireframe(X, Y, Z, ...)` 绘制骨架,线宽 0.8,颜色为该兔低饱和色
- P50 主线加粗(linewidth 1.6)突出中位趋势

视觉:线框骨架,不遮挡后方,结构清晰。

### 3.4 violin_box 模式(每单元 3D 箱体)

对每个 (兔子×周) 单元:
- 在该兔 X 中心位置画一个 3D 长方体(X 宽 0.5,Y 宽 0.7,Z 从 Q1 到 Q3),用 `Poly3DCollection` 6 面体,半透明 alpha=0.6,面色为该兔色
- 中位线:在长方体中央画 P50 横线(线宽 2.0,深色)
- 须线:从 Q3 向上到 whisker_hi,从 Q1 向下到 whisker_lo,细线(线宽 1.2)
- 须端横帽:whisker_hi/lo 处画短横线

视觉:最忠实于每单元真实分布,可比性强。

## 4. 统一视觉规范

| 项 | 规格 |
|---|---|
| 兔子配色 | 094=`#8c564b`(棕)、068=`#cd853f`(金)、030=`#1f77b4`(蓝) |
| 包络低饱和色 | 复用 `_desaturate_hex(color, blend=0.70)` |
| 字体 | SimHei 优先(复用 `_setup_chinese_font`) |
| 视角 | `view_init(elev=18, azim=-30)` 统一(与现有可比) |
| 盒子比例 | `set_box_aspect([1, 4, 0.6])`(Y 轴拉伸,与现有一致) |
| 投影 | `set_proj_type("persp")` |
| 标题 | `094 / 068 / 030 — {中文类型} [{模式}]` |
| DPI | 180 |
| 网格 | `ax.grid(False)`,三轴 pane.fill=False |

## 5. 输出产物

9 张图(3 模式 × 3 类型),路径:

```
output/figures/delta_r_r0/optimized_v3/3d/alt_forms/
├─ surface/
│   ├─ 3d_094_068_030_standard_surface.png
│   ├─ 3d_094_068_030_free_surface.png
│   └─ 3d_094_068_030_peaks_surface.png
├─ wireframe/
│   ├─ 3d_094_068_030_standard_wireframe.png
│   ├─ 3d_094_068_030_free_wireframe.png
│   └─ 3d_094_068_030_peaks_wireframe.png
└─ violin_box/
    ├─ 3d_094_068_030_standard_violin_box.png
    ├─ 3d_094_068_030_free_violin_box.png
    └─ 3d_094_068_030_peaks_violin_box.png
```

## 6. 主入口

```python
def main():
    # 默认生成全部 9 张
    # 支持 --mode surface|wireframe|violin_box 单一模式
```

运行后打印每张图路径,便于用户挑选。最终选定某模式后,可再决定是否将其作为默认覆盖原文件。

## 7. 复用清单

从 [plot_3d_three_rabbits.py](../../analysis/plot_3d_three_rabbits.py) 复用:
- `_load_data()` — 加载 std/free/peaks 三类数据
- `_to_df(data_dict)` — 转 DataFrame
- `_setup_chinese_font()` — 中文字体
- `_desaturate_hex()` — 低饱和色
- 常量 `RABBITS`(改为 094/068/030 顺序)、`RABBIT_COLORS`、`TYPE_LABELS`、`OUT_DIR`

## 8. 不做的事(YAGNI)

- 不修改现有 `plot_3d_three_rabbits.py`,原 `3d_094_068_030_*.png` 保留
- 不做跨兔插值(已规避科学风险)
- 不引入交互式 3D(plotly 等),保持 matplotlib 静态输出与项目一致
- 不新增数据加载逻辑,完全复用现有函数

## 9. 验证标准

- 脚本可独立运行 `python -m analysis.plot_3d_alt_forms`,无报错
- 生成 9 张 PNG,均能正常打开渲染中文
- 三种模式的 free/peaks/standard 三张图视觉风格统一(配色/视角/标题格式一致)
- 不影响现有 `plot_3d_three_rabbits.py` 输出
