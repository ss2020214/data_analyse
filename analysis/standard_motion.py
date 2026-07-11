# -*- coding: utf-8 -*-
"""
标准运动数据清洗与处理模块。

处理「标准运动数据汇总」文件夹中的数据：
- 人工对兔子腿部进行标准化摆动产生的传感器数据
- 每个Excel文件代表一只兔子，每个sheet代表不同周
- 每个sheet可能包含多组3列数据：[时间(s), 阻值(Ω), 运动次数]
- 多组之间用NaN列分隔

输出：标准化的CSV文件，格式为：
    rabbit, week, group, time_s, resistance_ohm, movement_count
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import json

from .config import ROOT, OUT

# ─── 配置 ──────────────────────────────────────────────────────────────────────
STANDARD_MOTION_DIR = ROOT / "标准运动数据汇总"
OUT_STD = OUT / "standard_motion"
OUT_STD.mkdir(parents=True, exist_ok=True)

# 清洗参数
CLEAN_PARAMS = {
    "time_min_s": 0.0,           # 时间最小值（排除负时间）
    "time_max_s": 900.0,         # 时间最大值（15分钟，兼容超10分钟的实验）
    "resistance_min_ohm": 1000,  # 电阻最小值（低于此不合理）
    "resistance_max_ohm": 1500000, # 电阻最大值（不同传感器基阻差异大，留足余量）
    "max_time_gap_s": 5.0,       # 最大允许时间间隔（秒）
    "min_points_per_group": 5,   # 每组最少数据点
    "hard_clip_high": 1.5,       # 硬截断上界（相对中位数倍数）
    "hard_clip_low": 1.0 / 1.5,  # 硬截断下界
    "outlier_mad_k": 6.0,        # MAD离群判别阈值倍数
    "outlier_window_s": 5.0,     # 滚动窗口（秒）
    "resample_fs": 5.0,          # 重采样频率（Hz）
}


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def _normalize_week(name: str) -> str:
    """统一周编号：'1w'→'w1', 'W1'→'w1'"""
    n = name.strip().lower()
    if n.startswith("w"):
        return n
    if n.endswith("w"):
        return "w" + n[:-1]
    return n


def _parse_rabbit_id(filename: str) -> str:
    """从文件名提取兔子ID：'Y016标准运动.xlsx' → 'Y016'"""
    stem = Path(filename).stem
    # 兔子ID是ASCII字母+数字（不含中文）
    rabbit_id = ""
    for ch in stem:
        if ch.isascii() and ch.isalnum():
            rabbit_id += ch
        else:
            break
    return rabbit_id


def _extract_groups(df: pd.DataFrame) -> List[Dict]:
    """
    从一个sheet的DataFrame中提取所有运动组。
    
    数据格式：
    - 3列 = 1组 [time, resistance, count]
    - 7列 = 2组 [t,r,c, NaN, t,r,c]
    - 11列 = 3组 [t,r,c, NaN, t,r,c, NaN, t,r,c]
    - 以此类推：cols = 4*n - 1（n为组数）
    - 特殊：2列 = [time, resistance]（无count列）
    
    返回：
        list of dict, 每个dict包含 'time', 'resistance', 'count'
    """
    ncols = df.shape[1]
    groups = []

    if ncols == 0 or df.shape[0] == 0:
        return groups

    # 特殊情况：只有2列（无count列）
    if ncols == 2:
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        r = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        groups.append({
            "time": t,
            "resistance": r,
            "count": np.nan,  # 未提供
        })
        return groups

    # 通用解析：每4列一组（3数据列 + 1分隔列），最后一组没有分隔列
    # 列布局: [0,1,2], [4,5,6], [8,9,10], ...
    col_idx = 0
    group_num = 0
    while col_idx + 2 < ncols:
        t_col = col_idx
        r_col = col_idx + 1
        c_col = col_idx + 2

        t = pd.to_numeric(df.iloc[:, t_col], errors="coerce").to_numpy()
        r = pd.to_numeric(df.iloc[:, r_col], errors="coerce").to_numpy()
        c = pd.to_numeric(df.iloc[:, c_col], errors="coerce").to_numpy()

        # 提取count值（通常仅第一行有值）
        count_vals = c[np.isfinite(c)]
        count = float(count_vals[0]) if len(count_vals) > 0 else np.nan

        groups.append({
            "time": t,
            "resistance": r,
            "count": count,
        })

        group_num += 1
        col_idx += 4  # 跳过3列数据 + 1列NaN分隔符

    # 如果 ncols == col_idx + 2（最后一组只有2列，如2列的特殊处理已覆盖上面）
    # 额外检查：如果剩余恰好2列
    if col_idx + 1 < ncols and col_idx + 2 >= ncols:
        t = pd.to_numeric(df.iloc[:, col_idx], errors="coerce").to_numpy()
        r = pd.to_numeric(df.iloc[:, col_idx + 1], errors="coerce").to_numpy()
        groups.append({
            "time": t,
            "resistance": r,
            "count": np.nan,
        })

    return groups


def load_all_standard_motion() -> pd.DataFrame:
    """
    加载标准运动数据汇总文件夹中的所有数据。
    
    返回：
        DataFrame with columns: rabbit, week, group_idx, time_s, resistance_ohm, movement_count
    """
    records = []
    folder = STANDARD_MOTION_DIR

    if not folder.exists():
        raise FileNotFoundError(f"标准运动数据文件夹不存在: {folder}")

    for fpath in sorted(folder.glob("*.xlsx")):
        rabbit_id = _parse_rabbit_id(fpath.name)
        xf = pd.ExcelFile(fpath)

        for sheet in xf.sheet_names:
            week = _normalize_week(sheet)
            df = xf.parse(sheet, header=None)

            if df.shape[0] == 0:
                continue

            groups = _extract_groups(df)

            for g_idx, grp in enumerate(groups):
                t = grp["time"]
                r = grp["resistance"]
                count = grp["count"]

                # 基本过滤：去除NaN行
                mask = np.isfinite(t) & np.isfinite(r)
                t_valid = t[mask]
                r_valid = r[mask]

                if len(t_valid) < 3:
                    continue

                for i in range(len(t_valid)):
                    records.append({
                        "rabbit": rabbit_id,
                        "week": week,
                        "group_idx": g_idx,
                        "time_s": t_valid[i],
                        "resistance_ohm": r_valid[i],
                        "movement_count": count,
                    })

    df_all = pd.DataFrame(records)
    return df_all


# ─── 数据清洗 ──────────────────────────────────────────────────────────────────

def clean_group(t: np.ndarray, r: np.ndarray,
                params: dict = CLEAN_PARAMS) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    对单组数据进行清洗。
    
    步骤：
    1. 范围过滤：时间和电阻值必须在合理范围内
    2. 时间排序和去重
    3. 硬截断：电阻偏离中位数过远的点
    4. MAD离群值检测与插值修复
    
    返回：
        (t_clean, r_clean, info_dict)
    """
    info = {
        "n_raw": len(t),
        "n_range_removed": 0,
        "n_hard_clip": 0,
        "n_mad_outlier": 0,
        "n_final": 0,
    }

    # 1) 范围过滤
    valid = (
        (t >= params["time_min_s"]) &
        (t <= params["time_max_s"]) &
        (r >= params["resistance_min_ohm"]) &
        (r <= params["resistance_max_ohm"])
    )
    info["n_range_removed"] = int((~valid).sum())
    t, r = t[valid], r[valid]

    if len(t) < params["min_points_per_group"]:
        info["n_final"] = len(t)
        return t, r, info

    # 2) 时间排序和去重
    idx = np.argsort(t)
    t, r = t[idx], r[idx]
    _, uniq_idx = np.unique(t, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    t, r = t[uniq_idx], r[uniq_idx]

    # 3) 硬截断：偏离中位数过远
    r_med = float(np.nanmedian(r))
    hi_thresh = r_med * params["hard_clip_high"]
    lo_thresh = r_med * params["hard_clip_low"]
    hard_bad = (r > hi_thresh) | (r < lo_thresh)
    info["n_hard_clip"] = int(hard_bad.sum())

    if hard_bad.any():
        good_idx = np.where(~hard_bad)[0]
        if good_idx.size > 3:
            r[hard_bad] = np.interp(
                np.where(hard_bad)[0], good_idx, r[good_idx]
            )

    # 4) MAD离群值检测
    fs_est = _estimate_fs(t)
    win = max(5, int(params["outlier_window_s"] * fs_est))
    s = pd.Series(r)
    rmed = s.rolling(win, center=True, min_periods=3).median()
    rmad = (s - rmed).abs().rolling(win, center=True, min_periods=3).median()
    global_mad = float(np.nanmedian(np.abs(r - np.nanmedian(r)))) or 1.0
    thr = params["outlier_mad_k"] * np.maximum(rmad.to_numpy(), global_mad * 0.5)
    bad = np.abs(r - rmed.to_numpy()) > thr
    info["n_mad_outlier"] = int(bad.sum())

    if bad.any():
        good_idx = np.where(~bad)[0]
        if good_idx.size > 3:
            r[bad] = np.interp(np.where(bad)[0], good_idx, r[good_idx])

    info["n_final"] = len(t)
    return t, r, info


def _estimate_fs(t: np.ndarray) -> float:
    """估算采样频率"""
    if len(t) < 2:
        return 5.0
    dt = np.median(np.diff(t))
    if dt <= 0:
        return 5.0
    return 1.0 / dt


# ─── 数据结构化 ────────────────────────────────────────────────────────────────

def structure_data(df_raw: pd.DataFrame, params: dict = CLEAN_PARAMS) -> pd.DataFrame:
    """
    对加载的原始数据进行清洗和结构化。
    
    按 (rabbit, week, group_idx) 分组，逐组执行清洗。
    
    返回：
        清洗后的 DataFrame，附加 'r_clean' 列
    """
    results = []

    grouped = df_raw.groupby(["rabbit", "week", "group_idx"])

    for (rabbit, week, g_idx), grp_df in grouped:
        t = grp_df["time_s"].to_numpy()
        r = grp_df["resistance_ohm"].to_numpy()
        count = grp_df["movement_count"].iloc[0]

        t_clean, r_clean, info = clean_group(t, r, params)

        if len(t_clean) < params["min_points_per_group"]:
            continue

        for i in range(len(t_clean)):
            results.append({
                "rabbit": rabbit,
                "week": week,
                "group_idx": g_idx,
                "time_s": round(t_clean[i], 4),
                "resistance_ohm": round(r_clean[i], 2),
                "movement_count": count,
            })

    df_out = pd.DataFrame(results)
    # 确保时间排序
    df_out = df_out.sort_values(
        ["rabbit", "week", "group_idx", "time_s"]
    ).reset_index(drop=True)
    return df_out


# ─── 数据验证 ──────────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> Dict:
    """
    对清洗后的数据集执行完整性和合理性验证。
    
    返回：
        验证报告字典
    """
    report = {
        "total_records": len(df),
        "rabbits": sorted(df["rabbit"].unique().tolist()),
        "n_rabbits": df["rabbit"].nunique(),
        "weeks_per_rabbit": {},
        "groups_per_session": {},
        "time_stats": {},
        "resistance_stats": {},
        "issues": [],
    }

    # 每只兔子的周数统计
    for rabbit in df["rabbit"].unique():
        weeks = sorted(df[df["rabbit"] == rabbit]["week"].unique().tolist())
        report["weeks_per_rabbit"][rabbit] = weeks

    # 每个session的组数
    session_groups = df.groupby(["rabbit", "week"])["group_idx"].nunique()
    report["groups_per_session"] = {
        "min": int(session_groups.min()),
        "max": int(session_groups.max()),
        "mean": round(float(session_groups.mean()), 1),
    }

    # 时间统计
    time_durations = []
    for (rabbit, week, g_idx), grp in df.groupby(["rabbit", "week", "group_idx"]):
        duration = grp["time_s"].max() - grp["time_s"].min()
        time_durations.append(duration)
        # 检查时间单调性
        if not np.all(np.diff(grp["time_s"].values) > 0):
            report["issues"].append(
                f"{rabbit}/{week}/g{g_idx}: 时间序列非严格递增"
            )

    report["time_stats"] = {
        "duration_min_s": round(min(time_durations), 2),
        "duration_max_s": round(max(time_durations), 2),
        "duration_mean_s": round(np.mean(time_durations), 2),
    }

    # 电阻统计
    report["resistance_stats"] = {
        "min_ohm": round(float(df["resistance_ohm"].min()), 1),
        "max_ohm": round(float(df["resistance_ohm"].max()), 1),
        "median_ohm": round(float(df["resistance_ohm"].median()), 1),
        "std_ohm": round(float(df["resistance_ohm"].std()), 1),
    }

    # 验证movement_count合理性
    counts = df.groupby(["rabbit", "week", "group_idx"])["movement_count"].first()
    valid_counts = counts.dropna()
    if len(valid_counts) > 0:
        report["movement_count_stats"] = {
            "min": float(valid_counts.min()),
            "max": float(valid_counts.max()),
            "mean": round(float(valid_counts.mean()), 1),
            "n_missing": int(counts.isna().sum()),
        }
        # 异常count值检查
        if valid_counts.max() > 100:
            report["issues"].append(
                f"存在异常大的运动次数: max={valid_counts.max()}"
            )

    # 空session检查
    for rabbit in report["rabbits"]:
        rabbit_weeks = report["weeks_per_rabbit"][rabbit]
        if len(rabbit_weeks) == 0:
            report["issues"].append(f"{rabbit}: 无有效周数据")

    return report


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def process_standard_motion(save: bool = True, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    完整处理流程：加载 → 清洗 → 结构化 → 验证 → 输出。
    
    参数：
        save: 是否保存输出文件
        verbose: 是否打印处理信息
    
    返回：
        (df_clean, validation_report)
    """
    if verbose:
        print("=" * 60)
        print("标准运动数据处理")
        print("=" * 60)

    # 1) 加载
    if verbose:
        print("\n[1/5] 数据加载...")
    df_raw = load_all_standard_motion()
    if verbose:
        print(f"  原始记录数: {len(df_raw):,}")
        print(f"  兔子数: {df_raw['rabbit'].nunique()}")
        print(f"  总sessions: {df_raw.groupby(['rabbit', 'week']).ngroups}")
        print(f"  总groups: {df_raw.groupby(['rabbit', 'week', 'group_idx']).ngroups}")

    # 2) 清洗与结构化
    if verbose:
        print("\n[2/5] 数据清洗...")
    df_clean = structure_data(df_raw)
    n_removed = len(df_raw) - len(df_clean)
    if verbose:
        print(f"  清洗后记录数: {len(df_clean):,}")
        print(f"  剔除点数: {n_removed:,} ({100*n_removed/len(df_raw):.1f}%)")

    # 3) 时间序列检查
    if verbose:
        print("\n[3/5] 时间序列结构化验证...")
    n_groups = df_clean.groupby(["rabbit", "week", "group_idx"]).ngroups
    if verbose:
        print(f"  有效运动组数: {n_groups}")

    # 4) 数据验证
    if verbose:
        print("\n[4/5] 数据完整性验证...")
    report = validate_data(df_clean)
    if verbose:
        print(f"  兔子: {report['rabbits']}")
        print(f"  每只兔子周数: ", end="")
        for rb, wks in report["weeks_per_rabbit"].items():
            print(f"{rb}({len(wks)}w) ", end="")
        print()
        print(f"  电阻范围: {report['resistance_stats']['min_ohm']:.0f} "
              f"~ {report['resistance_stats']['max_ohm']:.0f} Ω")
        print(f"  时长范围: {report['time_stats']['duration_min_s']:.1f} "
              f"~ {report['time_stats']['duration_max_s']:.1f} s")
        if "movement_count_stats" in report:
            mc = report["movement_count_stats"]
            print(f"  运动次数: {mc['min']:.0f}~{mc['max']:.0f} "
                  f"(均值{mc['mean']:.1f}, {mc['n_missing']}组缺失)")
        if report["issues"]:
            print(f"\n  ⚠ 发现 {len(report['issues'])} 个问题:")
            for issue in report["issues"][:10]:
                print(f"    - {issue}")

    # 5) 保存输出
    if save:
        if verbose:
            print("\n[5/5] 输出标准化数据...")

        # 主CSV
        csv_path = OUT_STD / "standard_motion_cleaned.csv"
        df_clean.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  CSV: {csv_path}")

        # 按兔子拆分
        per_rabbit_dir = OUT_STD / "per_rabbit"
        per_rabbit_dir.mkdir(exist_ok=True)
        for rabbit in df_clean["rabbit"].unique():
            rb_df = df_clean[df_clean["rabbit"] == rabbit]
            rb_path = per_rabbit_dir / f"{rabbit}_standard_motion.csv"
            rb_df.to_csv(rb_path, index=False, encoding="utf-8-sig")

        # 验证报告
        report_path = OUT_STD / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  报告: {report_path}")

        # 摘要表：每个session的统计信息
        summary_records = []
        for (rabbit, week, g_idx), grp in df_clean.groupby(
            ["rabbit", "week", "group_idx"]
        ):
            summary_records.append({
                "rabbit": rabbit,
                "week": week,
                "group_idx": g_idx,
                "n_points": len(grp),
                "duration_s": round(grp["time_s"].max() - grp["time_s"].min(), 2),
                "fs_hz": round(
                    (len(grp) - 1) / max(1e-6, grp["time_s"].max() - grp["time_s"].min()),
                    2,
                ),
                "resistance_median": round(grp["resistance_ohm"].median(), 1),
                "resistance_std": round(grp["resistance_ohm"].std(), 1),
                "movement_count": grp["movement_count"].iloc[0],
            })
        summary_df = pd.DataFrame(summary_records)
        summary_path = OUT_STD / "session_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  摘要: {summary_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)

    return df_clean, report


# ─── 便捷接口 ──────────────────────────────────────────────────────────────────

def load_standard_session(rabbit: str, week: str) -> List[Dict]:
    """
    加载某只兔子某周的标准运动数据（清洗后）。
    
    返回：
        list of dict, 每个dict包含:
            't': np.ndarray (时间),
            'r': np.ndarray (电阻),
            'count': float (运动次数),
            'info': dict (清洗信息)
    """
    folder = STANDARD_MOTION_DIR
    # 查找对应文件（排除 ~$ 临时文件）
    rabbit_upper = rabbit.upper()
    target = None
    for fpath in folder.glob("*.xlsx"):
        if fpath.name.startswith("~$"):
            continue
        if rabbit_upper in fpath.stem.upper():
            target = fpath
            break

    if target is None:
        raise FileNotFoundError(f"未找到兔子 {rabbit} 的标准运动文件")

    xf = pd.ExcelFile(target)
    week_norm = _normalize_week(week)
    sheet_map = {_normalize_week(s): s for s in xf.sheet_names}

    if week_norm not in sheet_map:
        raise KeyError(f"{rabbit} 没有 {week_norm} 的标准运动数据")

    df = xf.parse(sheet_map[week_norm], header=None)
    groups = _extract_groups(df)

    results = []
    for grp in groups:
        t = grp["time"]
        r = grp["resistance"]
        mask = np.isfinite(t) & np.isfinite(r)
        t, r = t[mask], r[mask]

        if len(t) < CLEAN_PARAMS["min_points_per_group"]:
            continue

        t_clean, r_clean, info = clean_group(t, r)
        results.append({
            "t": t_clean,
            "r": r_clean,
            "count": grp["count"],
            "info": info,
        })

    return results


# ─── 参考特征提取（供双阶段寻峰使用）──────────────────────────────

def _find_standard_file(rabbit: str) -> Optional[Path]:
    """根据兔子数字ID查找标准运动文件。"""
    folder = STANDARD_MOTION_DIR
    if not folder.exists():
        return None
    # 匹配规则：文件名中含有兔子的数字ID（排除 ~$ 临时文件）
    for fpath in folder.glob("*.xlsx"):
        if fpath.name.startswith("~$"):
            continue
        parsed = _parse_rabbit_id(fpath.name)
        numeric = ''.join(c for c in parsed if c.isdigit())
        if numeric == rabbit or parsed.upper().endswith(rabbit):
            return fpath
    return None


def get_reference_profile(rabbit: str, week: str) -> Optional[Dict]:
    """
    从标准运动数据提取传感器参考特征，用于指导自由运动数据的清洗和寻峰。

    返回 dict 包含：
        baseline_median       基线电阻中位数 (Ω)
        peak_amp_median       峰值幅度中位数 (Ω)
        peak_amp_mad          峰值幅度MAD (Ω)
        peak_rel_amp_median   相对幅度 (peak_amp / baseline)
        peak_fwhm_median      峰宽中位数 (s)
        noise_mad             静息段噪声水平 MAD (Ω)
        snr                   信噪比 (peak_amp_median / noise_mad)
        n_peaks_total         检出的标准峰总数
        movement_count_total  标注的运动次数总和
    """
    target = _find_standard_file(rabbit)
    if target is None:
        return None

    xf = pd.ExcelFile(target)
    week_norm = _normalize_week(week)
    sheet_map = {_normalize_week(s): s for s in xf.sheet_names}

    if week_norm not in sheet_map:
        # 回退：尝试用最近的可用周
        available = sorted(sheet_map.keys())
        if not available:
            return None
        # 找最近的周
        target_num = int(week_norm[1:]) if week_norm[1:].isdigit() else 1
        best = min(available, key=lambda w: abs(int(w[1:]) - target_num)
                   if w[1:].isdigit() else 999)
        week_norm = best

    df = xf.parse(sheet_map[week_norm], header=None)
    groups = _extract_groups(df)

    if not groups:
        return None

    # 对每组标准运动数据做简单预处理并提取峰特征
    all_baselines = []
    all_peak_amps = []
    all_peak_fwhms = []
    all_noise_vals = []
    total_count = 0

    from scipy.signal import find_peaks, peak_widths

    for grp in groups:
        t = grp["time"]
        r = grp["resistance"]
        count = grp["count"]
        mask = np.isfinite(t) & np.isfinite(r)
        t, r = t[mask], r[mask]
        if len(t) < 10:
            continue

        # 排序去重
        idx_sort = np.argsort(t)
        t, r = t[idx_sort], r[idx_sort]

        # P30 滚动基线（5s窗口，与 _extract_standard_motion_peaks 一致）
        fs_est = _estimate_fs(t)
        win_b = max(11, int(5.0 * fs_est))
        if win_b % 2 == 0:
            win_b += 1
        # 边界处理：点数不足以支撑完整窗口时，缩小至点数的 2/3
        if len(t) < win_b:
            win_b = max(5, (2 * len(t)) // 3)
            if win_b % 2 == 0:
                win_b += 1
        baseline = (pd.Series(r)
                    .rolling(win_b, center=True, min_periods=max(3, win_b // 2))
                    .quantile(0.30)
                    .bfill().ffill().to_numpy())

        detrended = r - baseline
        bl_median = float(np.nanmedian(baseline))
        all_baselines.append(bl_median)

        # 峰检测：与 _extract_standard_motion_peaks 使用相同参数
        mad = float(np.nanmedian(np.abs(detrended - np.nanmedian(detrended))))
        if mad <= 0:
            mad = float(np.nanstd(detrended)) or 1.0

        min_dist_s = 1.0
        distance = max(1, int(min_dist_s * fs_est))

        # 自适应阈值衰减：默认 2×MAD，检出不足时逐级降低
        height = 2.0 * mad
        peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)
        if count > 0 and peaks_idx.size < count * 0.5:
            height = 1.5 * mad
            peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)
        if count > 0 and peaks_idx.size < count * 0.5:
            height = 1.0 * mad
            peaks_idx, _ = find_peaks(detrended, distance=distance, height=height)

        if peaks_idx.size > 0:
            peak_amps = detrended[peaks_idx]
            all_peak_amps.extend(peak_amps.tolist())

            # FWHM
            try:
                widths, _, _, _ = peak_widths(detrended, peaks_idx, rel_height=0.5)
                fwhms = widths / fs_est
                all_peak_fwhms.extend(fwhms.tolist())
            except Exception as e:
                warnings.warn(f"get_reference_profile({rabbit},{week}): "
                              f"peak_widths failed ({e})")

        # 噪声估计：不在峰附近的段
        peak_mask = np.zeros(len(detrended), dtype=bool)
        for pi in peaks_idx:
            lo = max(0, pi - int(0.5 * fs_est))
            hi = min(len(detrended), pi + int(0.5 * fs_est))
            peak_mask[lo:hi] = True
        noise_seg = detrended[~peak_mask]
        if noise_seg.size > 5:
            all_noise_vals.extend(noise_seg.tolist())

        # 累加运动次数
        if np.isfinite(grp["count"]):
            total_count += int(grp["count"])

    # 汇总
    if not all_peak_amps:
        return None

    peak_amps_arr = np.array(all_peak_amps)
    noise_arr = np.array(all_noise_vals) if all_noise_vals else np.array([0.0])
    noise_mad = float(np.nanmedian(np.abs(noise_arr - np.nanmedian(noise_arr))))
    if noise_mad <= 0:
        noise_mad = float(np.nanstd(noise_arr)) or 1.0

    bl_median = float(np.nanmedian(all_baselines))
    peak_amp_median = float(np.nanmedian(peak_amps_arr))
    peak_amp_mad = float(np.nanmedian(np.abs(peak_amps_arr - peak_amp_median)))

    profile = {
        "baseline_median": bl_median,
        "peak_amp_median": peak_amp_median,
        "peak_amp_mad": peak_amp_mad,
        "peak_rel_amp_median": peak_amp_median / max(abs(bl_median), 1e-6),
        "peak_fwhm_median": float(np.nanmedian(all_peak_fwhms))
                             if all_peak_fwhms else 0.4,
        "noise_mad": noise_mad,
        "snr": peak_amp_median / max(noise_mad, 1e-6),
        "n_peaks_total": len(all_peak_amps),
        "movement_count_total": total_count,
    }
    return profile


def get_all_reference_profiles() -> Dict[str, Dict[str, Dict]]:
    """
    获取所有兔子所有周的标准运动参考特征。

    返回： {rabbit_id: {week: profile_dict}}
    其中 rabbit_id 是纯数字（如 "016"），与 config.py 中一致。
    """
    from .config import ALL_RABBITS, WEEKS

    profiles = {}
    for rabbit in ALL_RABBITS:
        profiles[rabbit] = {}
        for week in WEEKS:
            try:
                prof = get_reference_profile(rabbit, week)
                if prof is not None:
                    profiles[rabbit][week] = prof
            except Exception:
                continue
    return profiles


if __name__ == "__main__":
    process_standard_motion()
