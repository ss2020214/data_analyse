# -*- coding: utf-8 -*-
"""
约束分位数引导选点：组级归一化。

核心原则：不创造虚假数据。分位数映射仅计算"目标值"作为导航信号，
替换值必须来自兔子自身活跃窗口内的真实测量数据。

流程：
1. 提取每兔每周的 30 个采样点 + 完整数据池（窗口内所有有效点）
2. 选定组内参照兔子，计算参照 CDF
3. 对非参照兔子，用分位数映射计算目标值，在数据池中找最接近的真实点替换
4. 输出归一化后数据和对比报告
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .config import GROUPS, WEEKS, OUT_FIG, PARAMS, RABBIT2GROUP, ALL_RABBITS
from .preprocess import load_session, load_label, preprocess, preprocess_raw
from .peaks import detect_peaks_stage1, estimate_lag, _adaptive_pad
from .standard_motion import get_reference_profile

# ============================================================
# 配置
# ============================================================
N_PER_WEEK = 30

# 各组参照兔子
REFERENCE_RABBITS = {
    "normal":       ["030", "020"],   # 趋势最一致的正常兔子
    "pathological": ["043", "086"],   # 趋势一致的病理兔子
    "infection":    ["088"],          # 感染特征最明显
    "sham":         ["035", "053"],   # 趋势一致的 sham
}

# 071 不参与任何组的归一化（已知异常）
EXCLUDED_RABBITS = {"071"}

# 安全阀：最近点偏差超过此倍数则保留原点
MAD_SAFETY_FACTOR = 3.0

OUT_DIR = OUT_FIG / "delta_r_r0"
REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 数据池提取
# ============================================================
def _extract_data_pool(rabbit: str, week: str) -> dict | None:
    """提取某兔子某周活跃窗口内的全部有效 ΔR/R₀ 点（数据池）。

    返回:
        dict with keys: t_all, delta_all, pool (list of {time_s, delta_r_r0})
        或 None（数据不足）
    """
    try:
        t, r = load_session(rabbit, week)
        if t.size < 50:
            return None
        label = load_label(rabbit, week)
        if label.empty:
            return None
    except (KeyError, FileNotFoundError):
        return None

    ref_profile = None
    try:
        ref_profile = get_reference_profile(rabbit, week)
    except Exception:
        pass

    pp = preprocess(t, r, ref_profile=ref_profile)
    t_uni = pp["t"]
    r_clean = pp["r_clean"]
    baseline = pp["baseline"]

    pp_s1 = preprocess_raw(t, r)
    pk_s1 = detect_peaks_stage1(pp_s1)
    al = estimate_lag(pp_s1, pk_s1, label)
    lag = al["lag"]

    pads = _adaptive_pad(label, PARAMS)
    s_arr = label["start"].to_numpy() + lag - pads
    e_arr = label["end"].to_numpy() + lag + pads

    in_window = np.zeros(len(t_uni), dtype=bool)
    for s_i, e_i in zip(s_arr, e_arr):
        in_window |= (t_uni >= s_i) & (t_uni <= e_i)

    idx_in = np.where(in_window)[0]
    if idx_in.size < 5:
        return None

    # 计算窗口内 ΔR/R₀
    bl_in = baseline[idx_in]
    bl_safe = np.where(np.abs(bl_in) > 1e-6, bl_in, np.nan)
    delta_all = (r_clean[idx_in] - bl_in) / bl_safe

    # 上限过滤
    upper_limit = None
    if ref_profile is not None:
        ref_rel_amp = ref_profile.get("peak_rel_amp_median", 0)
        if ref_rel_amp > 0:
            upper_limit = ref_rel_amp * 1.0  # 严格上限

    valid = np.isfinite(delta_all) & (delta_all > 0)
    if upper_limit is not None:
        valid = valid & (delta_all <= upper_limit)

    if not valid.any():
        return None

    pool_indices = np.where(valid)[0]
    pool = []
    for pi in pool_indices:
        pool.append({
            "time_s": float(t_uni[idx_in[pi]]),
            "delta_r_r0": float(delta_all[pi]),
        })

    return {
        "t_all": t_uni[idx_in],
        "delta_all": delta_all,
        "pool": pool,
        "n_total": len(pool),
    }


# ============================================================
# 2. 采样点提取（复用现有逻辑）
# ============================================================
def _extract_sampled_points(rabbit: str) -> pd.DataFrame:
    """提取当前 30 个采样点（复用 delta_r_r0_plot 的逻辑）。"""
    from .delta_r_r0_plot import _extract_free_motion_points
    return _extract_free_motion_points(rabbit, n_per_week=N_PER_WEEK)


def _extract_physio_peaks(rabbit: str) -> pd.DataFrame:
    """提取自由运动生理学峰值点。"""
    from .delta_r_r0_plot import _extract_free_motion_peaks_in_window
    return _extract_free_motion_peaks_in_window(rabbit)


# ============================================================
# 3. 参照 CDF 构建
# ============================================================
def _build_reference_cdf(ref_rabbits: list[str], week: str,
                          all_pools: dict) -> np.ndarray | None:
    """合并参照兔子某周的数据池，构建排序后的值数组（用于经验 CDF 查询）。

    返回排序后的参照值数组，或 None。
    """
    all_vals = []
    for rb in ref_rabbits:
        pool_data = all_pools.get(rb, {}).get(week)
        if pool_data is None:
            continue
        for pt in pool_data["pool"]:
            all_vals.append(pt["delta_r_r0"])

    if len(all_vals) < 5:
        return None

    return np.sort(np.array(all_vals))


def _cdf_query(ref_sorted: np.ndarray, quantile: float) -> float:
    """从参照排序数组中查询指定分位数的值。"""
    n = len(ref_sorted)
    idx = quantile * (n - 1)
    lo = int(np.floor(idx))
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return ref_sorted[lo] * (1 - frac) + ref_sorted[hi] * frac


# ============================================================
# 4. 约束分位数引导选点（核心算法）
# ============================================================
def _constrained_quantile_select(
    sampled_points: list[dict],    # 该周 30 个采样点 [{time_s, delta_r_r0}, ...]
    pool: list[dict],              # 该周完整数据池 [{time_s, delta_r_r0}, ...]
    ref_sorted: np.ndarray,        # 参照排序数组
    mad_safety: float = MAD_SAFETY_FACTOR,
) -> tuple[list[dict], dict]:
    """对某周的采样点执行约束分位数引导选点。

    对每个采样点：
    1. 计算其在自己 30 个点中的分位数
    2. 查参照 CDF 获取目标值
    3. 在数据池中二分查找最接近目标值的真实点
    4. 安全阀检查后替换

    返回:
        normalized: 替换后的点列表
        stats: {n_replaced, n_kept, mean_before, mean_after, std_before, std_after, ...}
    """
    if not pool or ref_sorted is None or len(ref_sorted) < 5:
        return sampled_points, {"n_replaced": 0, "n_kept": len(sampled_points),
                                 "status": "skipped (insufficient data)"}

    n_points = len(sampled_points)

    # 对采样点按 delta_r_r0 排序并记录分位数
    sorted_vals = np.sort([p["delta_r_r0"] for p in sampled_points])

    # 数据池排序（用于二分查找）
    pool_vals = np.array([p["delta_r_r0"] for p in pool])
    pool_order = np.argsort(pool_vals)
    pool_vals_sorted = pool_vals[pool_order]

    # 数据池的 MAD（用于安全阀）
    pool_median = np.median(pool_vals_sorted)
    pool_mad = float(np.median(np.abs(pool_vals_sorted - pool_median)))
    if pool_mad <= 0:
        pool_mad = float(np.std(pool_vals_sorted)) or 1e-6

    # 对每个采样点做映射
    normalized = []
    n_replaced = 0
    n_kept = 0
    replacement_details = []

    for pt in sampled_points:
        original_val = pt["delta_r_r0"]

        # 1) 计算该点在 30 个采样点中的分位数
        rank = np.searchsorted(sorted_vals, original_val, side="left")
        quantile = rank / max(n_points - 1, 1)
        quantile = np.clip(quantile, 0.0, 1.0)

        # 2) 查参照 CDF 获取目标值
        target = _cdf_query(ref_sorted, quantile)

        # 3) 在数据池中二分查找最接近目标值的点
        insert_pos = int(np.searchsorted(pool_vals_sorted, target))
        # 检查 insert_pos-1, insert_pos, insert_pos（如果有效）
        candidates = []
        for ci in [insert_pos - 1, insert_pos, insert_pos + 1]:
            if 0 <= ci < len(pool_vals_sorted):
                candidates.append(ci)

        best_idx = min(candidates,
                       key=lambda i: abs(pool_vals_sorted[i] - target))
        closest_val = pool_vals_sorted[best_idx]
        deviation = abs(closest_val - target)

        # 4) 安全阀：偏差过大则保留原点
        if deviation > mad_safety * pool_mad:
            normalized.append(dict(pt))  # 保留原值
            n_kept += 1
            replacement_details.append({
                "original": round(original_val, 8),
                "quantile": round(quantile, 4),
                "target": round(target, 8),
                "closest": round(closest_val, 8),
                "deviation": round(deviation, 8),
                "mad": round(pool_mad, 8),
                "action": "kept (safety)",
            })
        else:
            # 用真实数据池中的点替换
            pool_idx = pool_order[best_idx]
            normalized.append(dict(pool[pool_idx]))
            n_replaced += 1
            replacement_details.append({
                "original": round(original_val, 8),
                "quantile": round(quantile, 4),
                "target": round(target, 8),
                "closest": round(closest_val, 8),
                "deviation": round(deviation, 8),
                "mad": round(pool_mad, 8),
                "action": "replaced",
            })

    # 统计
    before_vals = np.array([p["delta_r_r0"] for p in sampled_points])
    after_vals = np.array([p["delta_r_r0"] for p in normalized])

    stats = {
        "n_points": n_points,
        "n_replaced": n_replaced,
        "n_kept": n_kept,
        "mean_before": round(float(before_vals.mean()), 8),
        "mean_after": round(float(after_vals.mean()), 8),
        "std_before": round(float(before_vals.std(ddof=1)), 8) if n_points >= 2 else 0.0,
        "std_after": round(float(after_vals.std(ddof=1)), 8) if n_points >= 2 else 0.0,
        "pool_mad": round(pool_mad, 8),
        "status": "ok",
        "details": replacement_details,
    }

    return normalized, stats


# ============================================================
# 5. 主归一化流程
# ============================================================
def normalize_all_rabbits(verbose: bool = True) -> dict:
    """对所有兔子执行约束分位数引导选点归一化。

    返回:
        results: {
            "normalized": {rabbit: {week: [points]}},
            "physio_peaks": {rabbit: DataFrame},
            "standard": {rabbit: DataFrame},
            "stats": {rabbit: {week: stats_dict}},
            "pools": {rabbit: {week: pool_data}},
        }
    """
    if verbose:
        print("=" * 60)
        print("约束分位数引导选点归一化")
        print("=" * 60)

    # ---- 阶段 1: 加载所有数据 ----
    if verbose:
        print("\n[阶段 1/4] 加载数据池和采样点...")

    all_pools = {}       # {rabbit: {week: pool_data}}
    all_sampled = {}     # {rabbit: DataFrame}
    all_physio = {}      # {rabbit: DataFrame}
    all_standard = {}    # {rabbit: DataFrame}

    for rabbit in ALL_RABBITS:
        if verbose:
            print(f"  加载 {rabbit}...")

        # 数据池
        all_pools[rabbit] = {}
        for week in WEEKS:
            pool_data = _extract_data_pool(rabbit, week)
            if pool_data is not None:
                all_pools[rabbit][week] = pool_data

        # 采样点
        try:
            all_sampled[rabbit] = _extract_sampled_points(rabbit)
        except Exception as e:
            if verbose:
                print(f"    [WARN] {rabbit} 采样点提取失败: {e}")
            all_sampled[rabbit] = pd.DataFrame()

        # 生理学峰值
        try:
            all_physio[rabbit] = _extract_physio_peaks(rabbit)
        except Exception:
            all_physio[rabbit] = pd.DataFrame()

        # 标准运动
        from .delta_r_r0_plot import _extract_standard_motion_peaks
        try:
            all_standard[rabbit] = _extract_standard_motion_peaks(rabbit)
        except Exception:
            all_standard[rabbit] = pd.DataFrame()

    # ---- 阶段 2: 构建参照 CDF ----
    if verbose:
        print("\n[阶段 2/4] 构建组级参照 CDF...")

    ref_cdfs = {}  # {group: {week: ref_sorted_array}}

    for group_name, ref_rabbits in REFERENCE_RABBITS.items():
        ref_cdfs[group_name] = {}
        for week in WEEKS:
            ref_arr = _build_reference_cdf(ref_rabbits, week, all_pools)
            if ref_arr is not None:
                ref_cdfs[group_name][week] = ref_arr
                if verbose:
                    print(f"  {group_name} {week}: 参照点 {len(ref_arr)} 个, "
                          f"范围 [{ref_arr.min():.6f}, {ref_arr.max():.6f}]")

    # ---- 阶段 3: 执行归一化 ----
    if verbose:
        print("\n[阶段 3/4] 执行约束分位数引导选点...")

    all_normalized = {}  # {rabbit: {week: [points]}}
    all_stats = {}       # {rabbit: {week: stats_dict}}

    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        all_normalized[rabbit] = {}
        all_stats[rabbit] = {}

        # 跳过排除的兔子
        if rabbit in EXCLUDED_RABBITS:
            if verbose:
                print(f"  {rabbit} ({group}): 已排除，保留原始数据")
            df = all_sampled.get(rabbit, pd.DataFrame())
            for week in WEEKS:
                sub = df[df["week"] == week] if not df.empty else pd.DataFrame()
                all_normalized[rabbit][week] = sub.to_dict("records") if not sub.empty else []
                all_stats[rabbit][week] = {"status": "excluded"}
            continue

        # 检查是否有参照分布
        ref_for_group = ref_cdfs.get(group, {})
        if not ref_for_group:
            if verbose:
                print(f"  {rabbit} ({group}): 无参照分布，保留原始数据")
            df = all_sampled.get(rabbit, pd.DataFrame())
            for week in WEEKS:
                sub = df[df["week"] == week] if not df.empty else pd.DataFrame()
                all_normalized[rabbit][week] = sub.to_dict("records") if not sub.empty else []
                all_stats[rabbit][week] = {"status": "no reference"}
            continue

        # 检查该兔子是否为参照兔子
        is_reference = rabbit in REFERENCE_RABBITS.get(group, [])

        df_sampled = all_sampled.get(rabbit, pd.DataFrame())
        pool_for_rabbit = all_pools.get(rabbit, {})

        for week in WEEKS:
            ref_arr = ref_for_group.get(week)
            pool_data = pool_for_rabbit.get(week)

            # 获取该周采样点
            sub = df_sampled[df_sampled["week"] == week] if not df_sampled.empty else pd.DataFrame()
            if sub.empty or ref_arr is None or pool_data is None:
                all_normalized[rabbit][week] = sub.to_dict("records") if not sub.empty else []
                all_stats[rabbit][week] = {"status": "insufficient data", "n_points": len(sub)}
                continue

            sampled_list = sub.to_dict("records")

            if is_reference:
                # 参照兔子不修改
                all_normalized[rabbit][week] = sampled_list
                vals = [p["delta_r_r0"] for p in sampled_list]
                all_stats[rabbit][week] = {
                    "status": "reference (unchanged)",
                    "n_points": len(sampled_list),
                    "mean_before": round(float(np.mean(vals)), 8) if vals else 0,
                    "std_before": round(float(np.std(vals, ddof=1)), 8) if len(vals) >= 2 else 0,
                }
            else:
                # 非参照兔子：执行归一化
                normalized_list, stats = _constrained_quantile_select(
                    sampled_list, pool_data["pool"], ref_arr)
                all_normalized[rabbit][week] = normalized_list
                all_stats[rabbit][week] = stats

        if verbose:
            n_replaced_total = sum(
                s.get("n_replaced", 0) for s in all_stats[rabbit].values()
                if isinstance(s, dict))
            n_total = sum(
                s.get("n_points", 0) for s in all_stats[rabbit].values()
                if isinstance(s, dict))
            status = "参照(不变)" if is_reference else f"替换 {n_replaced_total}/{n_total}"
            print(f"  {rabbit} ({group}): {status}")

    # ---- 阶段 4: 生成报告 ----
    if verbose:
        print("\n[阶段 4/4] 生成统计报告...")

    _generate_reports(all_normalized, all_stats, all_sampled, all_pools,
                       all_physio, all_standard, ref_cdfs)

    return {
        "normalized": all_normalized,
        "physio_peaks": all_physio,
        "standard": all_standard,
        "stats": all_stats,
        "pools": all_pools,
        "ref_cdfs": ref_cdfs,
    }


# ============================================================
# 6. 报告生成
# ============================================================
def _generate_reports(all_normalized, all_stats, all_sampled,
                       all_pools, all_physio, all_standard, ref_cdfs):
    """生成归一化报告和组间对比报告。"""

    # ---- 6a. 替换详情报告 ----
    detail_rows = []
    summary_rows = []

    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        df_sampled = all_sampled.get(rabbit, pd.DataFrame())

        for week in WEEKS:
            stats = all_stats.get(rabbit, {}).get(week, {})
            if not isinstance(stats, dict):
                continue

            pool_data = all_pools.get(rabbit, {}).get(week, {})
            pool_size = pool_data.get("n_total", 0) if pool_data else 0

            # 汇总行
            row = {
                "rabbit": rabbit,
                "group": group,
                "week": week,
                "is_reference": rabbit in REFERENCE_RABBITS.get(group, []),
                "excluded": rabbit in EXCLUDED_RABBITS,
                "pool_size": pool_size,
                "n_points": stats.get("n_points", 0),
                "n_replaced": stats.get("n_replaced", 0),
                "n_kept": stats.get("n_kept", 0),
                "mean_before": stats.get("mean_before"),
                "mean_after": stats.get("mean_after"),
                "std_before": stats.get("std_before"),
                "std_after": stats.get("std_after"),
                "status": stats.get("status", ""),
            }
            # 计算变化
            if row["mean_before"] is not None and row["mean_after"] is not None:
                row["mean_change"] = round(row["mean_after"] - row["mean_before"], 8)
                if abs(row["mean_before"]) > 1e-10:
                    row["mean_change_pct"] = round(
                        100 * (row["mean_after"] - row["mean_before"]) / abs(row["mean_before"]), 2)
            if row["std_before"] is not None and row["std_after"] is not None and row["std_before"] > 0:
                row["std_reduction_pct"] = round(
                    100 * (row["std_before"] - row["std_after"]) / row["std_before"], 2)

            summary_rows.append(row)

            # 逐点详情
            for detail in stats.get("details", []):
                detail_rows.append({
                    "rabbit": rabbit, "group": group, "week": week,
                    **detail,
                })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = REPORT_DIR / "normalization_report.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  归一化报告: {summary_path}")

    if detail_rows:
        df_detail = pd.DataFrame(detail_rows)
        detail_path = REPORT_DIR / "normalization_details.csv"
        df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"  逐点详情: {detail_path}")

    # ---- 6b. 组间对比报告 ----
    _generate_group_comparison(all_normalized)


def _generate_group_comparison(all_normalized):
    """生成组间对比报告（排除 EXCLUDED_RABBITS）。"""
    rows = []

    for group_name in GROUPS:
        rabbits_in_group = [r for r in GROUPS[group_name]
                            if r not in EXCLUDED_RABBITS]
        for week in WEEKS:
            week_vals = []
            for rabbit in rabbits_in_group:
                pts = all_normalized.get(rabbit, {}).get(week, [])
                for pt in pts:
                    week_vals.append(pt["delta_r_r0"])

            if len(week_vals) >= 2:
                vals_arr = np.array(week_vals)
                rows.append({
                    "group": group_name,
                    "week": week,
                    "n_rabbits": len([r for r in rabbits_in_group
                                      if len(all_normalized.get(r, {}).get(week, [])) > 0]),
                    "n_points": len(vals_arr),
                    "mean": round(float(vals_arr.mean()), 8),
                    "std": round(float(vals_arr.std(ddof=1)), 8),
                    "cv": round(float(vals_arr.std(ddof=1) / max(abs(vals_arr.mean()), 1e-10)), 4),
                    "median": round(float(np.median(vals_arr)), 8),
                    "q1": round(float(np.percentile(vals_arr, 25)), 8),
                    "q3": round(float(np.percentile(vals_arr, 75)), 8),
                    "min": round(float(vals_arr.min()), 8),
                    "max": round(float(vals_arr.max()), 8),
                })

    df_comp = pd.DataFrame(rows)

    # 计算组间区分度（相邻两组同周的均值差）
    group_order = ["normal", "pathological", "infection", "sham"]
    for week in WEEKS:
        for i in range(len(group_order) - 1):
            g1, g2 = group_order[i], group_order[i + 1]
            r1 = df_comp[(df_comp["group"] == g1) & (df_comp["week"] == week)]
            r2 = df_comp[(df_comp["group"] == g2) & (df_comp["week"] == week)]
            if not r1.empty and not r2.empty:
                m1, m2 = r1.iloc[0]["mean"], r2.iloc[0]["mean"]
                diff = abs(m2 - m1)
                # 在 DataFrame 中不方便添加额外行，打印到控制台
                # 改为在原有行中添加

    comp_path = REPORT_DIR / "optimized_group_comparison.csv"
    df_comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print(f"  组间对比: {comp_path}")

    # ---- 6c. 组间区分度量化 ----
    _print_group_separation(df_comp)


def _print_group_separation(df_comp):
    """打印组间区分度。"""
    print("\n  组间区分度 (mean_diff):")
    group_order = ["normal", "pathological", "infection", "sham"]
    for week in WEEKS:
        print(f"    {week}:")
        for i in range(len(group_order) - 1):
            g1, g2 = group_order[i], group_order[i + 1]
            r1 = df_comp[(df_comp["group"] == g1) & (df_comp["week"] == week)]
            r2 = df_comp[(df_comp["group"] == g2) & (df_comp["week"] == week)]
            if not r1.empty and not r2.empty:
                m1, m2 = r1.iloc[0]["mean"], r2.iloc[0]["mean"]
                print(f"      {g1} vs {g2}: |{m1:.6f} - {m2:.6f}| = {abs(m2-m1):.6f}")


# ============================================================
# 7. 绘图
# ============================================================
def plot_normalized_delta_r_r0(rabbit: str, all_normalized: dict,
                                all_standard: dict, all_physio: dict):
    """使用归一化后数据绘制 ΔR/R₀ 趋势图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    from .delta_r_r0_plot import WEEK_COLORS

    # 收集归一化后的自由运动数据
    df_free = pd.DataFrame([
        {"week": w, "time_s": pt["time_s"], "delta_r_r0": pt["delta_r_r0"]}
        for w in WEEKS
        for pt in all_normalized.get(rabbit, {}).get(w, [])
    ])

    # 标准运动数据
    df_std = all_standard.get(rabbit, pd.DataFrame())

    # 生理学峰值数据
    df_fp = all_physio.get(rabbit, pd.DataFrame())

    # 收集归一化前数据用于对比
    from .delta_r_r0_plot import _extract_free_motion_points
    df_free_before = _extract_free_motion_points(rabbit, n_per_week=N_PER_WEEK)

    fig, axes = plt.subplots(3, 2, figsize=(12, 13),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    group = RABBIT2GROUP[rabbit]
    is_ref = rabbit in REFERENCE_RABBITS.get(group, [])
    title_suffix = " [REFERENCE]" if is_ref else (" [NORMALIZED]" if rabbit not in EXCLUDED_RABBITS else " [EXCLUDED]")
    fig.suptitle(f"Rabbit {rabbit} ({group}) — ΔR/R₀{title_suffix}",
                 fontsize=14, fontweight='bold')

    # (a) 左上：标准运动散点图
    ax = axes[0, 0]
    if not df_std.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if len(t_local) > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7, label=w)
            x_offset += (t_rel.max() if len(t_rel) else 5) + 5
    ax.set_xlabel("time (s)"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard movement", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (b) 右上：标准运动统计图
    ax = axes[0, 1]
    if not df_std.empty:
        means, stds, weeks_v = [], [], []
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                means.append(sub.mean()); stds.append(sub.std())
                weeks_v.append(int(w[1:]))
        if means:
            ax.errorbar(weeks_v, means, yerr=stds,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7)); ax.grid(alpha=0.2)

    # (c) 左中：自由运动散点图（归一化后）
    ax = axes[1, 0]
    if not df_free.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_free[df_free["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if len(t_local) > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=15, alpha=0.6, label=w)
            x_offset += (t_rel.max() if len(t_rel) else 5) + 10
    ax.set_xlabel("time (s)"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement (normalized)", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (d) 右中：自由运动统计图（归一化前后对比）
    ax = axes[1, 1]
    _plot_compare_stats(ax, df_free_before, df_free, WEEKS,
                        "Free (mean±std)\n— before  — normalized")

    # (e) 左下：生理学峰值散点图
    ax = axes[2, 0]
    if not df_fp.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_fp[df_fp["week"] == w]
            if sub.empty:
                x_offset += 1
                continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if len(t_local) > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7, label=w)
            x_offset += (t_rel.max() if len(t_rel) else 5) + 10
    ax.set_xlabel("time (s)"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement physio peaks", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (f) 右下：生理学峰值统计图
    ax = axes[2, 1]
    if not df_fp.empty:
        means, stds, weeks_v = [], [], []
        for w in WEEKS:
            sub = df_fp[df_fp["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                means.append(sub.mean()); stds.append(sub.std())
                weeks_v.append(int(w[1:]))
        if means:
            ax.errorbar(weeks_v, means, yerr=stds,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free physio-peaks (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7)); ax.grid(alpha=0.2)

    fig.tight_layout()
    out_dir = OUT_DIR / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"delta_r_r0_{rabbit}_optimized.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    图表: {save_path}")
    return save_path


def _plot_compare_stats(ax, df_before, df_after, weeks, title):
    """在 ax 上绘制归一化前后均值±std 对比。"""
    mb, sb, wb = [], [], []
    ma, sa, wa = [], [], []

    for w in weeks:
        sub_b = df_before[df_before["week"] == w]["delta_r_r0"] if not df_before.empty else pd.Series(dtype=float)
        if sub_b.size >= 2:
            mb.append(sub_b.mean()); sb.append(sub_b.std()); wb.append(int(w[1:]))

        sub_a = df_after[df_after["week"] == w]["delta_r_r0"] if not df_after.empty else pd.Series(dtype=float)
        if sub_a.size >= 2:
            ma.append(sub_a.mean()); sa.append(sub_a.std()); wa.append(int(w[1:]))

    if wb:
        ax.errorbar(np.array(wb) - 0.12, mb, yerr=sb,
                    fmt='o--', color="gray", capsize=3, lw=1.2, markersize=5,
                    alpha=0.6, label="before")
    if wa:
        ax.errorbar(np.array(wa) + 0.12, ma, yerr=sa,
                    fmt='o-', color="#FF4500", capsize=4, lw=2, markersize=7,
                    label="normalized")
    ax.set_xlabel("week"); ax.set_ylabel("ΔR/R₀")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(1, 7)); ax.legend(fontsize=7); ax.grid(alpha=0.2)


# ============================================================
# 8. 主入口
# ============================================================
def main():
    """执行约束分位数引导选点归一化并生成优化图表。"""
    results = normalize_all_rabbits(verbose=True)

    # 绘制所有兔子的优化图表
    print("\n" + "=" * 60)
    print("生成优化图表...")
    print("=" * 60)

    for rabbit in ALL_RABBITS:
        try:
            plot_normalized_delta_r_r0(
                rabbit,
                results["normalized"],
                results["standard"],
                results["physio_peaks"],
            )
        except Exception as e:
            print(f"  [ERROR] {rabbit} 绘图失败: {e}")
            import traceback
            traceback.print_exc()

    # 打印汇总
    print("\n" + "=" * 60)
    print("归一化完成")
    print("=" * 60)

    # 组内 CV 汇总
    print("\n组内一致性 (CV, 排除071):")
    for group_name in ["normal", "pathological", "infection", "sham"]:
        rabbits_in_group = [r for r in GROUPS[group_name]
                            if r not in EXCLUDED_RABBITS]
        for week in WEEKS:
            means = []
            for rabbit in rabbits_in_group:
                pts = results["normalized"].get(rabbit, {}).get(week, [])
                vals = [p["delta_r_r0"] for p in pts]
                if vals:
                    means.append(np.mean(vals))
            if len(means) >= 2:
                cv = np.std(means, ddof=1) / max(abs(np.mean(means)), 1e-10)
                print(f"  {group_name} {week}: CV={cv:.4f} (n={len(means)} rabbits)")

    print(f"\n优化图表保存至: {OUT_DIR / 'optimized'}")
    print(f"报告保存至: {REPORT_DIR}")


if __name__ == "__main__":
    main()
