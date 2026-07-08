# -*- coding: utf-8 -*-
"""
项目级配置：兔子样本元信息、文件路径、关键算法参数。

说明
----
- 兔子按背景文档分为4组：normal（正常愈合）、pathological（未愈合）、infection（感染）、sham（空白对照）。
- ORIG/LABEL 文件命名各只兔子有差异，这里通过 SUFFIX_ORIG 显式指定。
- 处理参数集中在 PARAMS 中，便于后续调参溯源。
"""

from pathlib import Path

# 项目根目录（脚本所在的上级目录）
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT_INTER = OUT / "intermediate"
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables"
for p in (OUT_INTER, OUT_FIG, OUT_TAB):
    p.mkdir(parents=True, exist_ok=True)

# 分组定义（4组）
# - normal:       骨折后正常愈合（6周内愈合）
# - pathological: 骨折后未愈合（至第6周未完全愈合）
# - infection:    做完手术发现感染
# - sham:         空白对照，仅植入传感器未骨折
GROUPS = {
    "normal":       ["030", "075", "071", "020"],
    "pathological": ["043", "086", "094", "015"],
    "infection":    ["016", "065", "088"],
    "sham":         ["035", "053", "068", "097"],
}
RABBIT2GROUP = {r: g for g, rs in GROUPS.items() for r in rs}
ALL_RABBITS = (GROUPS["normal"] + GROUPS["pathological"]
               + GROUPS["infection"] + GROUPS["sham"])

# 文件夹与文件名映射
FOLDERS = {
    "016": "016-new-run", "020": "020-new-run", "030": "030-new",
    "043": "043-new-run", "065": "065-new", "068": "068-new",
    "071": "071-new", "075": "075-new",
    "086": "086-new-run", "088": "088-new",
    "094": "094-new-run", "015": "015-new-run",
    "035": "035-new-run", "053": "053-new-run", "097": "097-new-run",
}
# 030 兔子原始文件命名为 _originl.xlsx（拼写差异），其它为 _original.xlsx
SUFFIX_ORIG = {r: ("_originl.xlsx" if r == "030" else "_original.xlsx")
               for r in ALL_RABBITS}

WEEKS = ["w1", "w2", "w3", "w4", "w5", "w6"]


def path_orig(rabbit: str) -> Path:
    return ROOT / FOLDERS[rabbit] / f"{rabbit}{SUFFIX_ORIG[rabbit]}"


def path_label(rabbit: str) -> Path:
    return ROOT / FOLDERS[rabbit] / f"{rabbit}_label.xlsx"


# 关键算法参数（每一项都会写入 report.md 说明）
PARAMS = {
    # 硬截断：原始 R 超出全局 median 的 [low, high] 倍区间即认为伪迹
    # 根据全体 58 session 统计，正常生物信号最大仅达 median 的 1.15×，
    # 所有 >1.5× 的点都是传感器接触伪迹。
    "hard_clip_high": 1.5,               # R > 1.5× median 划为伪迹
    "hard_clip_low": 1.0 / 1.5,          # R < median/1.5 同上

    # baseline 不稳定区域检测：
    # |dR_baseline/dt| > k× MAD（每秒变化超过 k 个噪底）的区域视为台阶事件
    # 这些区域不参与寻峰（baseline 被伪迹拉走、导致 detrended 似是而非）
    # 用绝对量级不用 robust 量级，避免 baseline 极稳时 median(db)=0 导致阈值坍缩
    "baseline_unstable_k": 10.0,         # |db/dt| > 10×MAD 即估为不稳定
    "baseline_unstable_dilate_s": 1.0,   # 不稳定区域边界补丁，避免过渡区误判

    # 重采样到统一时间网格的频率（原采样 ≈ 4.8 Hz）
    "resample_fs": 5.0,                  # Hz

    # 离群值剔除：rolling 窗 + MAD 倍数；超出阈值的点视为污染并插值
    "outlier_window_s": 5.0,             # 秒
    "outlier_mad_k": 6.0,                # 阈值倍数

    # 动态基线：滚动中位数窗口（足够长以滤掉跳跃，足够短以跟踪漂移）
    # v3：15→25 s ——burst 密集区段中位数会被峰体本身抬高，拉长窗口让 baseline
    # 更贴近静息位、detrended 峰高不被吞掉
    "baseline_window_s": 25.0,           # 秒
    # v4：baseline 从 P50（median）改为低分位数 quantile。
    # 信号大部分时间处于静息基线、偶尔向上跳起 → median 会被峰本身抬高。
    # P30 更接近"真静息位"，detrended 幅度不被吞掉，等价于再放松一档阈值。
    "baseline_quantile": 0.30,           # 0.5=中位数，0.3=偏静息位
    
    # 基线估计方法：
    #   "rolling_quantile" - 滚动分位数（当前默认，快速但可能被密集峰抬高）
    #   "asls" - 非对称最小二乘 (AsLS)，迭代拟合不敏感于峰密度的光滑基线
    # AsLS 优势：对峰密度不敏感，自动适应信号局部特征，无需调窗口大小
    # AsLS 劣势：计算量更大（~10次迭代 × 稀疏线性求解），单次 ~0.1s
    "baseline_method": "rolling_quantile",  # "rolling_quantile" | "asls"
    "asls_lambda": 1e6,                   # AsLS 平滑度（越大基线越光滑）
    "asls_p": 0.01,                       # AsLS 非对称性（越小基线越低，0.001~0.1）
    "asls_niter": 10,                     # AsLS 最大迭代次数

    # 峰值检测（detrended 信号上）
    # v3 放宽：用户反馈"相对基线提升不高但也是峰"被漏掉。
    # 原 peak_height=2.0 、prominence=3.0 、min_distance=0.30 偏保守。
    "peak_min_distance_s": 0.18,         # 0.30→0.18：burst 内连击可独立识别
    "peak_prominence_mad_k": 0.0,         # 禁用 prominence 阈值（不加 prominence 约束）
    "peak_height_mad_k": 0.5,            # 1.2→0.5：降低 height 阈值，多检出峰

    # 峰值增长速率限制（独立于 baseline_unstable）：
    # |d(detrended)/dt| > k × mad_global 即视为传感器瞬态/接触跳变伪迹。
    # baseline_unstable 抓 baseline 已跟上的台阶，本项抓 baseline 还未来得及
    # 跟上的瞬时跃变；两者合并屏蔽，候选峰落入区域则剔除。
    # 正常兔子跳跃峰上升速率约 30-50× mad/s 量级，因此阈值取 80 留出余量。
    "peak_rise_rate_mad_k": 80.0,        # |d(detrended)/dt| > k×MAD/s 视为伪迹
    "peak_rise_dilate_s": 0.3,           # 快速跃变区域边界向外补丁（秒）

    # 时间对齐：搜索的延迟范围（label 相对 signal 平移的秒数）
    # background.md: "视频最多比传感器早或者晚 10 秒"
    "lag_search_min_s": -10.0,
    "lag_search_max_s": 10.0,
    "lag_search_step_s": 0.1,

    # label 窗口边界扩展（自适应）
    # 短窗口标注误差占比大，需要更多容忍；长窗口标注精度高，pad可以小
    # 实际 pad = min(label_window_pad_max_s, label_window_pad_base_s / max(1, 窗口时长/2))
    # 短窗口(1s): pad≈min(1.0, 1.0/0.5)=1.0s
    # 中窗口(3s): pad≈min(1.0, 1.0/1.5)=0.67s
    # 长窗口(6s): pad≈min(1.0, 1.0/3.0)=0.33s
    "label_window_pad_max_s": 1.0,            # 最大扩展量
    "label_window_pad_base_s": 1.0,           # 基准扩展量（用于反比计算）
}
