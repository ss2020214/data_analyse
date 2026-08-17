# -*- coding: utf-8 -*-
"""
动态基线 ΔR/R₀ 算法总体流程图（论文方法章节 Figure 1）。

主流程自上而下：数据加载 → 重采样 → 硬截断 → MAD 离群修复 →
伪迹屏蔽 → 动态基线 → 去趋势 → Stage 2 寻峰 → ΔR/R₀ → 周汇总。
左侧为标准运动参考特征输入，右侧为视频标注时间对齐输入。

运行: python -m analysis.plot_methodology_flowchart
输出: output/figures/methodology/algorithm_flowchart.png
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from .config import OUT_FIG

OUT_DIR = OUT_FIG / "methodology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 颜色方案 ----------
C_MAIN = "#DCE9F7"      # 主流程浅蓝
C_MAIN_EDGE = "#2F5B8F"
C_KEY = "#FDEBD0"       # 关键步骤（基线/结果）浅橙
C_KEY_EDGE = "#B9770E"
C_MASK = "#FADBD8"      # 伪迹屏蔽浅红
C_MASK_EDGE = "#B03A2E"
C_SIDE = "#E8F6EF"      # 侧向输入浅绿
C_SIDE_EDGE = "#1E8449"
C_NOTE = "#F4F6F7"      # 备注浅灰
C_NOTE_EDGE = "#7F8C8D"


def _box(ax, x, y, w, h, text, fc, ec, fs=10, fw="normal", lw=1.5,
         ls="-", radius=0.02):
    """绘制圆角矩形框 + 居中文字。"""
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.6,rounding_size={radius*100}",
                       mutation_aspect=1.0,
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, zorder=3, linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1, ls="-", color="#34495E", lw=1.6,
           style="-|>", label=None, label_offset=(0, 0), fs=8.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1),
                zorder=1)
    if label:
        ax.text((x0 + x1) / 2 + label_offset[0],
                (y0 + y1) / 2 + label_offset[1],
                label, fontsize=fs, color=color, ha="center", va="center")


def plot_flowchart():
    fig, ax = plt.subplots(figsize=(11.5, 14.5))
    ax.set_xlim(0, 115)
    ax.set_ylim(0, 154)
    ax.axis("off")

    # ---------- 主流程 ----------
    bx, bw, bh = 34.0, 47.0, 9.0     # 主框位置/宽/高
    gap = 4.2                        # 框间距
    y_top = 138.0
    ys = [y_top - i * (bh + gap) for i in range(10)]

    steps = [
        ("① 原始数据加载与整序\n时间(s)/阻值(Ω) 双列 · 去污 · 排序去重", C_MAIN, C_MAIN_EDGE),
        ("② 均匀网格重采样\n线性插值至 fs = 5.0 Hz", C_MAIN, C_MAIN_EDGE),
        ("③ 硬截断伪迹剔除\n超出 [median/1.5, 1.5×median] 区间的点 → 插值修复", C_MAIN, C_MAIN_EDGE),
        ("④ 滚动 MAD 离群修复\n5 s 窗 · 偏离局部中位数 > 6×MAD → 插值修复", C_MAIN, C_MAIN_EDGE),
        ("⑤ 伪迹区域屏蔽（禁止寻峰）\n基线台阶 >10×MAD/s · 快速跃变 >80×MAD/s", C_MASK, C_MASK_EDGE),
        ("⑥ 动态基线 R0(t)\n25 s 居中滚动 P30 分位数（边缘 bfill/ffill）", C_KEY, C_KEY_EDGE),
        ("⑦ 去趋势\nD(t) = R_clean(t) − R0(t) · 全局 MAD 噪底", C_MAIN, C_MAIN_EDGE),
        ("⑧ Stage 2 寻峰（三重约束）\n0.5×MAD 阈值 · 上限过滤 · 上升<下降形态过滤", C_MAIN, C_MAIN_EDGE),
        ("⑨ ΔR/R0 计算\n(R_clean − R0)/R0 · 有限值 · R0>1e-6 · 取正值", C_KEY, C_KEY_EDGE),
        ("⑩ 周水平汇总\n每周均值 ± 标准差 → 跨周趋势与组间比较", C_MAIN, C_MAIN_EDGE),
    ]

    for (txt, fc, ec), y in zip(steps, ys):
        fw = "bold" if fc == C_KEY else "normal"
        _box(ax, bx, y, bw, bh, txt, fc, ec, fs=10.5, fw=fw)

    # 主流程箭头
    for i in range(len(ys) - 1):
        _arrow(ax, bx + bw / 2, ys[i], bx + bw / 2, ys[i + 1] + bh)

    # ---------- 左侧：标准运动参考特征 ----------
    lx, lw_ = 2.0, 27.0
    ly = ys[5] - 3.0
    _box(ax, lx, ly, lw_, 13.0,
         "标准运动参考特征\n5 s P30 局部基线寻峰\n相对幅值中位数 / P90\nSNR · FWHM",
         C_SIDE, C_SIDE_EDGE, fs=9.5, fw="bold")
    # → ④ SNR 自适应离群阈值
    _arrow(ax, lx + lw_, ly + 11.0, bx, ys[3] + bh / 2,
           ls="--", color=C_SIDE_EDGE, label="SNR 自适应\n离群阈值",
           label_offset=(-3.5, 2.5))
    # → ⑧ 上限过滤与阈值自适应
    _arrow(ax, lx + lw_, ly + 6.5, bx, ys[7] + bh / 2,
           ls="--", color=C_SIDE_EDGE, label="幅值上限 ×1.2\n阈值自适应",
           label_offset=(-3.5, -3.0))
    # → ⑨ 相对幅值 P90 上限
    _arrow(ax, lx + lw_, ly + 2.0, bx, ys[8] + bh / 2,
           ls="--", color=C_SIDE_EDGE, label="P90 上限过滤",
           label_offset=(-3.5, -5.5))

    # ---------- 右侧：视频标注时间对齐 ----------
    rx, rw_ = 86.0, 27.0
    ry = ys[6] - 2.0
    _box(ax, rx, ry, rw_, 13.0,
         "视频标注时间对齐\n标注窗口 start/end/count\n±10 s 网格搜索 δ*\n逐窗口峰数匹配",
         C_SIDE, C_SIDE_EDGE, fs=9.5, fw="bold")
    # → ⑧ 活跃窗口内峰归属
    _arrow(ax, rx, ry + 9.5, bx + bw, ys[7] + bh / 2,
           ls="--", color=C_SIDE_EDGE, label="活跃窗口\n峰归属",
           label_offset=(-4.5, 2.5))
    # → ⑨ 采样点窗口约束
    _arrow(ax, rx, ry + 3.5, bx + bw, ys[8] + bh / 2,
           ls="--", color=C_SIDE_EDGE, label="窗口内\n采样点约束",
           label_offset=(-4.5, -3.0))

    # ---------- 备注：Stage 1 变体 ----------
    _box(ax, 86.0, ys[5] + 1.0, 27.0, 7.5,
         "Stage 1 变体（时域特征）\n75 s 窗 · P15 · 宽松寻峰\n→ IPI / 跳跃频率",
         C_NOTE, C_NOTE_EDGE, fs=9, ls="--")
    _arrow(ax, 86.0, ys[5] + 4.5, bx + bw, ys[5] + bh / 2,
           ls=":", color=C_NOTE_EDGE)

    # ---------- 标题 ----------
    ax.text(57.5, 152.0, "动态基线 ΔR/R0 量化算法 — 总体处理流程",
            ha="center", va="center", fontsize=14, fontweight="bold")

    fig.tight_layout()
    out_png = OUT_DIR / "algorithm_flowchart.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存: {out_png}")
    return out_png


if __name__ == "__main__":
    plot_flowchart()
