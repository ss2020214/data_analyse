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

# 新增空白对照兔子：仅提供标准运动数据（"标准运动数据汇总"目录），
# 无原始连续信号/人工标注文件，因此不并入 ALL_RABBITS，
# 只参与标准运动相关分析与图表（std_peaks / IQR 周趋势 / 标准运动散点趋势）。
STD_ONLY_RABBITS = {"050": "sham", "070": "sham"}

RABBIT2GROUP = {r: g for g, rs in GROUPS.items() for r in rs}
RABBIT2GROUP.update(STD_ONLY_RABBITS)
ALL_RABBITS = (GROUPS["normal"] + GROUPS["pathological"]
               + GROUPS["infection"] + GROUPS["sham"])
# 标准运动全量兔子（含仅标准运动数据的兔子），供标准运动加载/绘图使用
ALL_RABBITS_STD = ALL_RABBITS + sorted(STD_ONLY_RABBITS)

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
    # P30 更接近“真静息位”，detrended 幅度不被吞掉，等价于再放松一档阈值。
    "baseline_quantile": 0.30,           # 0.5=中位数，0.3=偏静息位

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

    # ------ 文档法 ΔR/R₀（峰值−谷值 / 全局 R₀）参数组 ------
    # 口径: ΔR/R₀ = (R_peak − R_valley) / R₀
    #  - R_peak: 峰处真实阻值（清洗后）
    #  - R_valley: 该峰局部结构谷（find_peaks left_base 处阻值）
    #  - R₀: 全局常量（session 静息段 P20 分位数，受 r_quantile 配置）
    # use_doc_method: 主开关，默认 False（现有图不受影响）；对比验证通过后翻转启用。
    # 自由运动按周单侧离群剔除（用于主图 delta_r_r0_*.png）：
    # 每周对 ΔR/R₀ 计算 med + mad_k × robust_std，只剔除偏大的单侧离群点。
    # robust_std = 1.4826×MAD，但设定绝对地板 floor_std，防止 med≈0 的周
    # MAD 塌缩导致阈值过低而过度剔除（如 Sham 静息周）。
    "free_outlier_mad_k": 3.0,     # 稳健 std 倍数（>med+mad_k×robust_std 判为离群）
    "free_outlier_floor_std": 0.005,  # robust_std 绝对下限（0.5% ΔR/R₀）

    "use_doc_method": False,   # 已切回原动态基线算法（文档法关闭）
    # 结构谷检测的滚动基线窗口（秒），与现标准法 P30 基线 C30 一致
    "doc_detrend_window_s": 5.0,
    # 谷回退阈值：峰脚 left_base 与峰间距 < 该值（秒）时，改取峰±0.5s 局部最小值
    "doc_valley_min_gap_s": 0.2,
    # 峰左侧谷搜索窗口（秒）：谷只在与峰的该距离内寻找，且取窗口内距峰最近的谷
    "doc_valley_seek_s": 1.5,
    # 静息段 R₀ 的分位数（P20 = 0.20）
    "doc_r0_quantile": 0.20,
    # 静息段最小占比，低于该值记录告警并回退整段低分位数，避免 R₀ 偏肥
    "doc_rest_min_ratio": 0.1,
    # 标准运动上限容差（沿用现标准运动上限过滤允许值）
    "doc_upper_limit_k": 1.2,
}

# 标准运动滚动基线分位数的按兔子/周覆盖（默认取 PARAMS["baseline_quantile"]=0.30）
# 050 的 w3/w4 中 P30 基线被密集峰体抬高、偏离静息位，按用户要求改用 P20
STD_BASELINE_QUANTILE_OVERRIDE = {
    "050": {"w3": 0.20, "w4": 0.20},
}

# 标准运动基线模式覆盖（优先于分位数覆盖）：
#   None（默认）→ 滚动分位数基线；"global_min" → 整段最小值常数基线
STD_BASELINE_MODE_OVERRIDE = {
    "050": {"w3": "global_min", "w4": "global_min"},
}


def std_baseline_quantile(rabbit: str, week: str) -> float:
    """标准运动滚动基线的分位数：优先按兔子/周覆盖，否则用全局默认值。"""
    return float(STD_BASELINE_QUANTILE_OVERRIDE.get(rabbit, {})
                 .get(week, PARAMS["baseline_quantile"]))


def std_baseline_mode(rabbit: str, week: str) -> str | None:
    """标准运动基线模式：None=滚动分位数（默认），"global_min"=整段最小值常数基线。"""
    return STD_BASELINE_MODE_OVERRIDE.get(rabbit, {}).get(week)


# 每周选点模式覆盖（峰数超过 max_per_week 时的选点策略）：
#   "max"（默认）→ 时间分 max_per_week 段，段内取 ΔR/R₀ 最大
#   "min"         → 时间分段，段内取 ΔR/R₀ 最小
#   "seg_median"  → 时间分段，段内取 ΔR/R₀ 中位数对应的峰
#   "global_min"  → 不按时间分段，直接取全周 ΔR/R₀ 最小的 max_per_week 个
# 050：w1/w2 取全局最小 10 个；w5 段内取最小；w6 段内取中位
STD_SELECT_MODE_OVERRIDE = {
    "050": {"w1": "global_min", "w2": "global_min",
            "w5": "min", "w6": "seg_median"},
}


def std_select_mode(rabbit: str, week: str) -> str:
    """每周选点模式：默认 "max"，支持按兔子/周覆盖 min / seg_median / global_min。"""
    return STD_SELECT_MODE_OVERRIDE.get(rabbit, {}).get(week, "max")


# 手动峰级覆盖：按峰时间（秒）强制剔除/补入指定峰，用于个别峰的人工精确调整。
# 应用时机：在该周常规选点完成后；容差 0.05s（峰间距≥1s）；
# 补入峰必须存在于该周已检出的正值峰池中。
STD_SELECT_FORCE_EXCLUDE = {
    "050": {"w5": [43.63]},   # w5 原第 1 个入选（绿#1）峰
}
STD_SELECT_FORCE_INCLUDE = {
    "050": {"w5": [63.81]},   # 原绿#5 与绿#6 之间的峰
}
