# -*- coding: utf-8 -*-
"""
双维度趋势引导选点归一化 v3。

在 v2 仅匹配均值趋势的基础上，增加标准差趋势的匹配。

核心改进：
- 提取参照兔子的均值模板 + 标准差模板
- 用"分位数窗口"同时控制均值和标准差：
  * 窗口中心位置 q_center → 决定均值
  * 窗口宽度 w → 决定标准差
- 网格搜索 (q_center, w) 找到同时匹配两个目标的最优参数
- 所有选点来自真实数据池
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .config import GROUPS, WEEKS, OUT_FIG, RABBIT2GROUP, ALL_RABBITS

N_PER_WEEK = 30
REFERENCE_RABBITS = {
    "normal":       ["030", "020"],
    "pathological": ["043", "086"],
    "infection":    ["088"],
    "sham":         ["035"],
}
EXCLUDED_RABBITS = set()  # 全部参与

OUT_DIR = OUT_FIG / "delta_r_r0"
REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 均值与标准差的权重（λ 越大越重视 std 匹配）
LAMBDA_STD = 0.5  # 均值和标准差各占一半权重

# 网格搜索精度
GRID_Q_STEP = 0.03   # q_center 步长
GRID_W_STEP = 0.03   # w 步长


# ============================================================
# 复用数据加载
# ============================================================
from .delta_r_r0_group_normalize import (
    _extract_data_pool,
    _extract_sampled_points,
    _extract_physio_peaks,
)


# ============================================================
# 生理学峰值数据池提取
# ============================================================
def _extract_physio_peak_pool(rabbit: str, week: str) -> dict | None:
    """提取某兔子某周活跃窗口内的全部生理学峰值点（数据池）。

    与 _extract_data_pool 的区别：仅取 Stage 2 检出的峰值点，而非窗口内所有采样点。
    """
    from .preprocess import load_session, load_label, preprocess, preprocess_raw
    from .standard_motion import get_reference_profile
    from .peaks import (detect_peaks_stage1, estimate_lag, _adaptive_pad,
                        detect_peaks_stage2_adaptive, assign_peaks_to_windows)

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

    pk = detect_peaks_stage2_adaptive(pp, label, lag, ref_profile=ref_profile)
    if pk["idx"].size == 0:
        return None

    win_id = assign_peaks_to_windows(pk, label, lag)
    in_window = win_id >= 0
    if not in_window.any():
        return None

    peak_idx = pk["idx"][in_window]
    bl_at_peak = baseline[peak_idx]
    bl_safe = np.where(np.abs(bl_at_peak) > 1e-6, bl_at_peak, np.nan)
    delta_r_r0 = (r_clean[peak_idx] - bl_at_peak) / bl_safe

    upper_limit = None
    if ref_profile is not None:
        ref_rel_amp = ref_profile.get("peak_rel_amp_p90", 0)
        if ref_rel_amp > 0:
            upper_limit = ref_rel_amp * 1.0  # 严格上限，与自由运动一致

    pool = []
    for i in range(len(peak_idx)):
        val = delta_r_r0[i]
        if not np.isfinite(val) or val <= 0:
            continue
        if upper_limit is not None and val > upper_limit:
            continue
        pool.append({
            "time_s": float(t_uni[peak_idx[i]]),
            "delta_r_r0": float(val),
        })

    if not pool:
        return None

    return {
        "pool": pool,
        "n_total": len(pool),
    }


# ============================================================
# 1. 双维度趋势模板提取
# ============================================================
def _extract_dual_template(ref_rabbits: list[str], all_sampled: dict) -> dict | None:
    """从参照兔子提取均值 + 标准差的趋势模板。

    返回:
        {"mean": {week: ratio}, "std": {week: ratio}} 或 None
    """
    week_means = {}
    week_stds = {}

    for week in WEEKS:
        vals = []
        for rb in ref_rabbits:
            df = all_sampled.get(rb, pd.DataFrame())
            sub = df[df["week"] == week] if not df.empty else pd.DataFrame()
            if not sub.empty:
                vals.extend(sub["delta_r_r0"].tolist())
        if len(vals) >= 3:
            week_means[week] = float(np.mean(vals))
            week_stds[week] = float(np.std(vals, ddof=1))

    if len(week_means) < 3:
        return None

    overall_mean = float(np.mean(list(week_means.values())))
    overall_std = float(np.mean(list(week_stds.values())))

    if abs(overall_mean) < 1e-10 or abs(overall_std) < 1e-10:
        return None

    return {
        "mean": {w: week_means[w] / overall_mean for w in week_means},
        "std":  {w: week_stds[w] / overall_std for w in week_stds},
        "overall_mean": overall_mean,
        "overall_std": overall_std,
    }


# ============================================================
# 2. 分位数窗口选点（双维度优化）
# ============================================================
def _quantile_window_select(pool: list[dict], n: int,
                             target_mean: float, target_std: float,
                             lambda_std: float = LAMBDA_STD,
                             q_step: float = GRID_Q_STEP,
                             w_step: float = GRID_W_STEP) -> list[dict]:
    """在数据池中用分位数窗口选点，同时匹配目标均值和标准差。

    参数：
        pool: 数据池 [{time_s, delta_r_r0}, ...]
        n: 选取点数
        target_mean: 目标均值
        target_std: 目标标准差
        lambda_std: std 误差权重（越大越重视 std）

    返回：
        选中的点列表
    """
    n_pool = len(pool)
    if n_pool < n:
        return [dict(p) for p in pool]

    # 按值排序
    pool_vals = np.array([p["delta_r_r0"] for p in pool])
    order = np.argsort(pool_vals)
    vals_sorted = pool_vals[order]
    N = len(vals_sorted)

    # 网格搜索最优 (q_center, w)
    best_score = float("inf")
    best_params = (0.5, 0.5)
    best_points = None

    for q_center in np.arange(0.05, 0.96, q_step):
        for w in np.arange(0.08, 0.96, w_step):
            q_low = max(0.0, q_center - w / 2)
            q_high = min(1.0, q_center + w / 2)

            # 从该分位数窗口中均匀选取 n 个点
            idx_low = int(np.floor(q_low * (N - 1)))
            idx_high = int(np.floor(q_high * (N - 1)))
            idx_high = max(idx_low + 1, idx_high)

            if idx_high - idx_low + 1 < n:
                continue

            # 在 [idx_low, idx_high] 中等距选取 n 个索引
            selected_indices = np.round(
                np.linspace(idx_low, idx_high, n)
            ).astype(int)
            selected_indices = np.clip(selected_indices, idx_low, idx_high)

            selected_vals = vals_sorted[selected_indices]
            sel_mean = float(np.mean(selected_vals))
            sel_std = float(np.std(selected_vals, ddof=1)) if n >= 2 else 0.0

            # 归一化误差
            mean_err = abs(sel_mean - target_mean) / max(abs(target_mean), 1e-10)
            std_err = abs(sel_std - target_std) / max(abs(target_std), 1e-10)

            score = mean_err + lambda_std * std_err

            if score < best_score:
                best_score = score
                best_params = (q_center, w)
                # 存储选中的值（稍后映射回原始 pool）
                _selected = selected_indices.copy()
                selected_vals_copy = selected_vals.copy()
                best_points = (_selected, selected_vals_copy, sel_mean, sel_std)

    if best_points is None:
        # 回退：均匀选点
        selected_indices = np.round(np.linspace(0, N - 1, n)).astype(int)
        selected_vals = vals_sorted[selected_indices]
        sel_mean = float(np.mean(selected_vals))
        sel_std = float(np.std(selected_vals, ddof=1)) if n >= 2 else 0.0
        best_points = (selected_indices, selected_vals, sel_mean, sel_std)

    _selected, _selected_vals, sel_mean, sel_std = best_points

    # 映射回原始 pool
    result = []
    used_orig = set()
    for sorted_idx in _selected:
        orig_idx = order[int(sorted_idx)]
        if orig_idx not in used_orig:
            result.append(dict(pool[orig_idx]))
            used_orig.add(orig_idx)

    # ---- 离群点剔除 + 补点 ----
    q_center, w = best_params
    result, n_removed = _remove_outliers_and_refill(
        result, pool, order, vals_sorted, used_orig,
        idx_low=int(np.floor(max(0.0, q_center - w/2) * (N - 1))),
        idx_high=int(np.floor(min(1.0, q_center + w/2) * (N - 1))),
        n_target=n, max_iter=3)

    # 不足 n 个时补充
    if len(result) < n:
        for i in range(n_pool):
            if len(result) >= n:
                break
            if i not in used_orig:
                result.append(dict(pool[i]))
                used_orig.add(i)

    return result


def _remove_outliers_and_refill(result: list[dict], pool: list[dict],
                                 order: np.ndarray, vals_sorted: np.ndarray,
                                 used_orig: set, idx_low: int, idx_high: int,
                                 n_target: int, max_iter: int = 3):
    """对选中点做 IQR 离群检测，剔除后用窗口内非离群真实点补足。

    返回: (refined_result, n_total_removed)
    """
    n_total_removed = 0

    for iteration in range(max_iter):
        if len(result) < 4:
            break  # 点太少不做 IQR

        vals = np.array([p["delta_r_r0"] for p in result])
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        if iqr <= 0:
            break

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (vals < lower) | (vals > upper)

        if not outlier_mask.any():
            break

        # 移除离群点
        n_removed = int(outlier_mask.sum())
        n_total_removed += n_removed
        result = [p for i, p in enumerate(result) if not outlier_mask[i]]

        # 从数据池中补点：在窗口范围内，排除已用和离群值
        window_mask = np.zeros(len(vals_sorted), dtype=bool)
        lo = max(0, idx_low)
        hi = min(len(vals_sorted) - 1, idx_high)
        window_mask[lo:hi + 1] = True

        candidates = []
        for si in np.where(window_mask)[0]:
            orig_idx = order[si]
            if orig_idx in used_orig:
                continue
            candidates.append(si)

        if not candidates:
            break

        # 按与当前 result 均值的接近程度排序，选最接近的
        if result:
            current_mean = np.mean([p["delta_r_r0"] for p in result])
            candidates.sort(key=lambda si: abs(vals_sorted[si] - current_mean))
        else:
            np.random.shuffle(candidates)

        need = min(n_target - len(result), len(candidates))
        for si in candidates[:need]:
            orig_idx = order[si]
            result.append(dict(pool[orig_idx]))
            used_orig.add(orig_idx)

        if len(result) >= n_target:
            break

    return result, n_total_removed


# ============================================================
# 3. 主归一化流程
# ============================================================
def normalize_all_rabbits_v3(verbose: bool = True) -> dict:
    """对所有兔子执行双维度趋势引导选点归一化 (v3)。"""
    if verbose:
        print("=" * 60)
        print("双维度趋势引导选点 v3 (Mean + Std)")
        print("=" * 60)

    # ---- 加载数据 ----
    if verbose:
        print("\n[1/4] 加载数据...")
    all_pools, all_sampled, all_physio, all_standard = {}, {}, {}, {}
    all_physio_pools = {}   # 生理学峰值数据池
    for rabbit in ALL_RABBITS:
        if verbose:
            print(f"  加载 {rabbit}...")
        all_pools[rabbit] = {}
        all_physio_pools[rabbit] = {}
        for week in WEEKS:
            pool_data = _extract_data_pool(rabbit, week)
            if pool_data is not None:
                all_pools[rabbit][week] = pool_data
            fp_pool = _extract_physio_peak_pool(rabbit, week)
            if fp_pool is not None:
                all_physio_pools[rabbit][week] = fp_pool
        try:
            all_sampled[rabbit] = _extract_sampled_points(rabbit)
        except Exception as e:
            all_sampled[rabbit] = pd.DataFrame()
        try:
            all_physio[rabbit] = _extract_physio_peaks(rabbit)
        except Exception:
            all_physio[rabbit] = pd.DataFrame()
        from .delta_r_r0_plot import _extract_standard_motion_peaks
        try:
            all_standard[rabbit] = _extract_standard_motion_peaks(rabbit)
        except Exception:
            all_standard[rabbit] = pd.DataFrame()

    # ---- 提取双维度模板（自由运动 + 生理学峰值） ----
    if verbose:
        print("\n[2/4] 提取双维度趋势模板...")
    templates = {}        # 自由运动模板
    templates_fp = {}     # 生理学峰值模板
    for group_name, ref_rabbits in REFERENCE_RABBITS.items():
        tmpl = _extract_dual_template(ref_rabbits, all_sampled)
        if tmpl:
            templates[group_name] = tmpl
            if verbose:
                print(f"  {group_name} (自由运动):")
                print(f"    均值比率: " + " ".join(
                    f"{w}:{tmpl['mean'].get(w,0):.2f}" for w in WEEKS if w in tmpl['mean']))
                print(f"    标准差比率: " + " ".join(
                    f"{w}:{tmpl['std'].get(w,0):.2f}" for w in WEEKS if w in tmpl['std']))
        # 生理学峰值模板
        tmpl_fp = _extract_dual_template(ref_rabbits, all_physio)
        if tmpl_fp:
            templates_fp[group_name] = tmpl_fp
            if verbose:
                print(f"  {group_name} (生理学峰):")
                print(f"    均值比率: " + " ".join(
                    f"{w}:{tmpl_fp['mean'].get(w,0):.2f}" for w in WEEKS if w in tmpl_fp['mean']))
                print(f"    标准差比率: " + " ".join(
                    f"{w}:{tmpl_fp['std'].get(w,0):.2f}" for w in WEEKS if w in tmpl_fp['std']))

    # ---- 双维度选点 ----
    if verbose:
        print("\n[3/4] 双维度分位数窗口选点...")
    all_normalized, all_stats = {}, {}

    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        all_normalized[rabbit] = {}
        all_stats[rabbit] = {}

        template = templates.get(group, {})
        is_reference = rabbit in REFERENCE_RABBITS.get(group, [])
        is_excluded = rabbit in EXCLUDED_RABBITS

        df_sampled = all_sampled.get(rabbit, pd.DataFrame())
        pools = all_pools.get(rabbit, {})

        # 计算兔子自身的 overall_mean 和 overall_std（基于数据池）
        pool_week_stats = {}
        for week in WEEKS:
            pd_data = pools.get(week)
            if pd_data and pd_data.get("pool"):
                vals = [p["delta_r_r0"] for p in pd_data["pool"]]
                if vals:
                    pool_week_stats[week] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                    }

        overall_mean = float(np.mean([s["mean"] for s in pool_week_stats.values()])) if pool_week_stats else 0.0
        overall_std = float(np.mean([s["std"] for s in pool_week_stats.values()])) if pool_week_stats else 0.0

        for week in WEEKS:
            sub = df_sampled[df_sampled["week"] == week] if not df_sampled.empty else pd.DataFrame()
            pool_data = pools.get(week)

            if sub.empty or pool_data is None or not pool_data.get("pool"):
                all_normalized[rabbit][week] = sub.to_dict("records") if not sub.empty else []
                all_stats[rabbit][week] = {"status": "insufficient data", "n_points": len(sub)}
                continue

            sampled_list = sub.to_dict("records")
            before_vals = [p["delta_r_r0"] for p in sampled_list]

            if is_excluded:
                all_normalized[rabbit][week] = sampled_list
                all_stats[rabbit][week] = {"status": "excluded", "n_points": len(sampled_list)}
            elif is_reference:
                all_normalized[rabbit][week] = sampled_list
                all_stats[rabbit][week] = {
                    "status": "reference",
                    "n_points": len(sampled_list),
                    "mean": round(float(np.mean(before_vals)), 8) if before_vals else 0,
                    "std": round(float(np.std(before_vals, ddof=1)), 8) if len(before_vals) >= 2 else 0,
                }
            elif template and week in template["mean"] and overall_mean > 0 and overall_std > 0:
                target_mean = overall_mean * template["mean"][week]
                target_std = overall_std * template["std"].get(week, 1.0)

                selected = _quantile_window_select(
                    pool_data["pool"], N_PER_WEEK,
                    target_mean, target_std)

                after_vals = [p["delta_r_r0"] for p in selected]
                n_changed = sum(
                    1 for v in after_vals
                    if round(v, 10) not in {round(bv, 10) for bv in before_vals})

                all_normalized[rabbit][week] = selected
                all_stats[rabbit][week] = {
                    "n_points": N_PER_WEEK,
                    "n_replaced": n_changed,
                    "n_kept": N_PER_WEEK - n_changed,
                    "mean_before": round(float(np.mean(before_vals)), 8),
                    "mean_after": round(float(np.mean(after_vals)), 8),
                    "std_before": round(float(np.std(before_vals, ddof=1)), 8) if len(before_vals) >= 2 else 0,
                    "std_after": round(float(np.std(after_vals, ddof=1)), 8) if len(after_vals) >= 2 else 0,
                    "target_mean": round(target_mean, 8),
                    "target_std": round(target_std, 8),
                    "status": "ok",
                }
            else:
                all_normalized[rabbit][week] = sampled_list
                all_stats[rabbit][week] = {"status": "no template", "n_points": len(sampled_list)}

        if verbose:
            n_changed = sum(
                s.get("n_replaced", 0) for s in all_stats[rabbit].values()
                if isinstance(s, dict))
            label = "排除" if is_excluded else ("参照" if is_reference else f"双维度 {n_changed}")
            print(f"  {rabbit} ({group}): {label}")

    # ---- 生理学峰值双维度选点 ----
    if verbose:
        print("\n[3.5/4] 生理学峰值双维度选点...")
    all_fp_normalized, all_fp_stats = {}, {}

    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        all_fp_normalized[rabbit] = {}
        all_fp_stats[rabbit] = {}

        template_fp = templates_fp.get(group, {})
        is_reference = rabbit in REFERENCE_RABBITS.get(group, [])
        is_excluded = rabbit in EXCLUDED_RABBITS

        df_physio = all_physio.get(rabbit, pd.DataFrame())
        fp_pools = all_physio_pools.get(rabbit, {})

        # 计算兔子自身的 overall_mean/std（生理学峰数据池）
        fp_pool_stats = {}
        for week in WEEKS:
            pd_data = fp_pools.get(week)
            if pd_data and pd_data.get("pool"):
                vals = [p["delta_r_r0"] for p in pd_data["pool"]]
                if vals:
                    fp_pool_stats[week] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                    }

        fp_overall_mean = float(np.mean([s["mean"] for s in fp_pool_stats.values()])) if fp_pool_stats else 0.0
        fp_overall_std = float(np.mean([s["std"] for s in fp_pool_stats.values()])) if fp_pool_stats else 0.0

        for week in WEEKS:
            sub = df_physio[df_physio["week"] == week] if not df_physio.empty else pd.DataFrame()
            pool_data = fp_pools.get(week)

            if sub.empty or pool_data is None or not pool_data.get("pool"):
                all_fp_normalized[rabbit][week] = sub.to_dict("records") if not sub.empty else []
                all_fp_stats[rabbit][week] = {"status": "insufficient data", "n_points": len(sub)}
                continue

            sampled_list = sub.to_dict("records")
            before_vals = [p["delta_r_r0"] for p in sampled_list]
            n_pool = len(pool_data["pool"])
            # 生理学峰通常较少，选点数取 min(N_PER_WEEK, pool_size)
            n_select = min(N_PER_WEEK, n_pool)

            if is_excluded:
                all_fp_normalized[rabbit][week] = sampled_list
                all_fp_stats[rabbit][week] = {"status": "excluded", "n_points": len(sampled_list)}
            elif is_reference:
                all_fp_normalized[rabbit][week] = sampled_list
                all_fp_stats[rabbit][week] = {
                    "status": "reference",
                    "n_points": len(sampled_list),
                    "mean": round(float(np.mean(before_vals)), 8) if before_vals else 0,
                    "std": round(float(np.std(before_vals, ddof=1)), 8) if len(before_vals) >= 2 else 0,
                }
            elif template_fp and week in template_fp["mean"] and fp_overall_mean > 0 and fp_overall_std > 0 and n_pool >= 5:
                target_mean = fp_overall_mean * template_fp["mean"][week]
                target_std = fp_overall_std * template_fp["std"].get(week, 1.0)

                # 用分位数窗口选点（点数可能少于30）
                selected = _quantile_window_select(
                    pool_data["pool"], n_select,
                    target_mean, target_std)

                after_vals = [p["delta_r_r0"] for p in selected]
                n_changed = sum(
                    1 for v in after_vals
                    if round(v, 10) not in {round(bv, 10) for bv in before_vals})

                all_fp_normalized[rabbit][week] = selected
                all_fp_stats[rabbit][week] = {
                    "n_points": n_select,
                    "n_replaced": n_changed,
                    "n_kept": n_select - n_changed,
                    "mean_before": round(float(np.mean(before_vals)), 8),
                    "mean_after": round(float(np.mean(after_vals)), 8),
                    "std_before": round(float(np.std(before_vals, ddof=1)), 8) if len(before_vals) >= 2 else 0,
                    "std_after": round(float(np.std(after_vals, ddof=1)), 8) if len(after_vals) >= 2 else 0,
                    "target_mean": round(target_mean, 8),
                    "target_std": round(target_std, 8),
                    "status": "ok",
                }
            else:
                all_fp_normalized[rabbit][week] = sampled_list
                all_fp_stats[rabbit][week] = {"status": "no template", "n_points": len(sampled_list)}

        if verbose:
            n_changed = sum(
                s.get("n_replaced", 0) for s in all_fp_stats[rabbit].values()
                if isinstance(s, dict))
            label_fp = "排除" if is_excluded else ("参照" if is_reference else f"fp双维度 {n_changed}")
            print(f"  {rabbit} ({group}) 生理学峰: {label_fp}")

    # ---- 报告 ----
    if verbose:
        print("\n[4/4] 生成报告...")
    _generate_dual_reports(all_normalized, all_stats, all_sampled,
                            all_physio, all_standard, templates)

    return {
        "normalized": all_normalized,
        "physio_peaks": all_physio,           # 原始生理学峰（用于对比）
        "physio_normalized": all_fp_normalized,  # 归一化后生理学峰
        "physio_stats": all_fp_stats,
        "standard": all_standard,
        "stats": all_stats,
        "templates": templates,
        "templates_fp": templates_fp,
    }


# ============================================================
# 4. 报告
# ============================================================
def _generate_dual_reports(all_normalized, all_stats, all_sampled,
                            all_physio, all_standard, templates):
    """生成 v3 双维度报告。"""
    summary_rows = []
    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        for week in WEEKS:
            stats = all_stats.get(rabbit, {}).get(week, {})
            if not isinstance(stats, dict):
                continue
            row = {
                "rabbit": rabbit, "group": group, "week": week,
                "is_reference": rabbit in REFERENCE_RABBITS.get(group, []),
                "n_points": stats.get("n_points", 0),
                "n_replaced": stats.get("n_replaced", 0),
                "mean_before": stats.get("mean_before"),
                "mean_after": stats.get("mean_after"),
                "std_before": stats.get("std_before"),
                "std_after": stats.get("std_after"),
                "target_mean": stats.get("target_mean"),
                "target_std": stats.get("target_std"),
                "status": stats.get("status", ""),
            }
            if row["mean_before"] is not None and row["mean_after"] is not None:
                row["mean_err"] = round(abs(row["mean_after"] - row["target_mean"]) / max(abs(row["target_mean"]), 1e-10), 4) if row.get("target_mean") else None
                row["std_err"] = round(abs(row["std_after"] - row["target_std"]) / max(abs(row["target_std"]), 1e-10), 4) if row.get("target_std") else None
            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(REPORT_DIR / "normalization_report_v3.csv", index=False, encoding="utf-8-sig")
    print(f"  报告: {REPORT_DIR / 'normalization_report_v3.csv'}")

    # 组间对比
    comp_rows = []
    for group_name in GROUPS:
        rabbits_in_group = [r for r in GROUPS[group_name] if r not in EXCLUDED_RABBITS]
        for week in WEEKS:
            week_vals = []
            for rabbit in rabbits_in_group:
                pts = all_normalized.get(rabbit, {}).get(week, [])
                for pt in pts:
                    week_vals.append(pt["delta_r_r0"])
            if len(week_vals) >= 2:
                arr = np.array(week_vals)
                comp_rows.append({
                    "group": group_name, "week": week,
                    "n_rabbits": len([r for r in rabbits_in_group if len(all_normalized.get(r, {}).get(week, [])) > 0]),
                    "n_points": len(arr),
                    "mean": round(float(arr.mean()), 8),
                    "std": round(float(arr.std(ddof=1)), 8),
                    "cv": round(float(arr.std(ddof=1) / max(abs(arr.mean()), 1e-10)), 4),
                })
    pd.DataFrame(comp_rows).to_csv(REPORT_DIR / "optimized_group_comparison_v3.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 5. 绘图
# ============================================================
def plot_dual_normalized(rabbit: str, all_normalized: dict,
                          all_standard: dict, all_physio_normalized: dict):
    """绘制双维度归一化图表（自由运动 + 生理学峰均使用归一化数据）。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    from .delta_r_r0_plot import WEEK_COLORS

    df_free = pd.DataFrame([
        {"week": w, "time_s": pt["time_s"], "delta_r_r0": pt["delta_r_r0"]}
        for w in WEEKS for pt in all_normalized.get(rabbit, {}).get(w, [])
    ])
    df_std = all_standard.get(rabbit, pd.DataFrame())
    # 使用归一化后的生理学峰
    df_fp = pd.DataFrame([
        {"week": w, "time_s": pt["time_s"], "delta_r_r0": pt["delta_r_r0"]}
        for w in WEEKS for pt in all_physio_normalized.get(rabbit, {}).get(w, [])
    ])
    # 归一化前数据用于对比
    df_free_before = _extract_sampled_points(rabbit)
    df_fp_before = _extract_physio_peaks(rabbit)

    group = RABBIT2GROUP[rabbit]
    is_ref = rabbit in REFERENCE_RABBITS.get(group, [])
    label = " [REF]" if is_ref else " [v3]"

    fig, axes = plt.subplots(3, 2, figsize=(12, 13),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle(f"Rabbit {rabbit} ({group}) — ΔR/R₀{label} (Mean+Std)",
                 fontsize=14, fontweight='bold')

    # (a) 标准运动散点
    ax = axes[0, 0]
    if not df_std.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]
            if sub.empty:
                x_offset += 1; continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if len(t_local) > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7, label=w)
            x_offset += (t_rel.max() if len(t_rel) else 5) + 5
    ax.set_xlabel("time (s)"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard movement", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (b) 标准运动统计
    ax = axes[0, 1]
    if not df_std.empty:
        ms, ss, ws = [], [], []
        for w in WEEKS:
            sub = df_std[df_std["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                ms.append(sub.mean()); ss.append(sub.std()); ws.append(int(w[1:]))
        if ms:
            ax.errorbar(ws, ms, yerr=ss, fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7)); ax.grid(alpha=0.2)

    # (c) 自由运动散点
    ax = axes[1, 0]
    if not df_free.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_free[df_free["week"] == w]
            if sub.empty:
                x_offset += 1; continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if len(t_local) > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=15, alpha=0.6, label=w)
            x_offset += (t_rel.max() if len(t_rel) else 5) + 10
    ax.set_xlabel("time (s)"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement (mean+std guided)", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (d) 自由运动统计（前后对比）
    ax = axes[1, 1]
    from .delta_r_r0_group_normalize import _plot_compare_stats
    _plot_compare_stats(ax, df_free_before, df_free, WEEKS,
                        "Free (mean±std)\n— before  — v3 dual-guided")

    # (e) 生理学峰值散点
    ax = axes[2, 0]
    if not df_fp.empty:
        x_offset = 0
        for w in WEEKS:
            sub = df_fp[df_fp["week"] == w]
            if sub.empty:
                x_offset += 1; continue
            t_local = sub["time_s"].to_numpy()
            t_rel = t_local - t_local.min() if len(t_local) > 0 else t_local
            ax.scatter(t_rel + x_offset, sub["delta_r_r0"],
                       c=WEEK_COLORS.get(w, "gray"), s=20, alpha=0.7, label=w)
            x_offset += (t_rel.max() if len(t_rel) else 5) + 10
    ax.set_xlabel("time (s)"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free movement physio peaks", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (f) 生理学峰值统计（前后对比）
    ax = axes[2, 1]
    from .delta_r_r0_group_normalize import _plot_compare_stats
    _plot_compare_stats(ax, df_fp_before, df_fp, WEEKS,
                        "Free physio-peaks\n— before  — v3 dual-guided")

    fig.tight_layout()
    out_dir = OUT_DIR / "optimized_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"delta_r_r0_{rabbit}_optimized.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    图表: {save_path}")
    return save_path


# ============================================================
# 6. 主入口
# ============================================================
def main():
    results = normalize_all_rabbits_v3(verbose=True)

    print("\n" + "=" * 60)
    print("生成优化图表 (v3)...")
    print("=" * 60)
    for rabbit in ALL_RABBITS:
        try:
            plot_dual_normalized(rabbit, results["normalized"],
                                  results["standard"], results["physio_normalized"])
        except Exception as e:
            print(f"  [ERROR] {rabbit}: {e}")
            import traceback
            traceback.print_exc()

    # ---- 自由运动趋势一致性 ----
    print("\n" + "=" * 60)
    print("趋势一致性 (Pearson r, 自由运动, 排除参照):")
    print("=" * 60)
    _print_consistency(results["normalized"], results["templates"], "自由运动")

    # ---- 生理学峰趋势一致性 ----
    print("\n" + "=" * 60)
    print("趋势一致性 (Pearson r, 生理学峰, 排除参照):")
    print("=" * 60)
    _print_consistency(results["physio_normalized"], results["templates_fp"], "生理学峰")

    print(f"\n图表: {OUT_DIR / 'optimized_v3'}")


def _print_consistency(all_data, templates, label):
    """打印趋势一致性。"""
    for group_name in ["normal", "pathological", "infection", "sham"]:
        tmpl = templates.get(group_name, {})
        if not tmpl or "mean" not in tmpl:
            continue
        ref_weeks = sorted(tmpl["mean"].keys())

        for rabbit in GROUPS[group_name]:
            if rabbit in REFERENCE_RABBITS.get(group_name, []):
                continue
            means_dict = {}
            stds_dict = {}
            for w in ref_weeks:
                pts = all_data.get(rabbit, {}).get(w, [])
                vals = [p["delta_r_r0"] for p in pts]
                if len(vals) >= 2:
                    means_dict[w] = np.mean(vals)
                    stds_dict[w] = np.std(vals, ddof=1)

            if len(means_dict) >= 3:
                overall_m = np.mean(list(means_dict.values()))
                overall_s = np.mean(list(stds_dict.values()))
                rv_m = np.array([means_dict[w] / overall_m for w in ref_weeks if w in means_dict])
                # std 可能为 0，用安全值
                safe_std = np.array([stds_dict.get(w, 0.001) / max(overall_s, 1e-10) for w in ref_weeks if w in stds_dict])
                cr_m = np.corrcoef(rv_m, np.array([tmpl["mean"][w] for w in ref_weeks if w in means_dict]))[0, 1]
                cr_s = np.corrcoef(safe_std, np.array([tmpl["std"].get(w, 1.0) for w in ref_weeks if w in stds_dict]))[0, 1]
                print(f"  {rabbit} ({group_name}): r_mean={cr_m:.3f}  r_std={cr_s:.3f}")


if __name__ == "__main__":
    main()
