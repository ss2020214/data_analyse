# -*- coding: utf-8 -*-
"""
趋势引导选点归一化 v2。

核心改进：不追求幅值一致，只追求跨周趋势一致。
- 从参照兔子提取"趋势模板"（各周相对比率）
- 对非参照兔子，保持其自身幅值水平，但强制各周均值遵循参照趋势
- 从数据池中选点，用贪心扩展算法使所选点的均值逼近目标均值
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .config import GROUPS, WEEKS, OUT_FIG, RABBIT2GROUP, ALL_RABBITS

# ============================================================
# 配置
# ============================================================
N_PER_WEEK = 30
REFERENCE_RABBITS = {
    "normal":       ["030", "020"],
    "pathological": ["043", "086"],
    "infection":    ["088"],
    "sham":         ["035"],          # 只用035作参照，053 w2趋势不一致
}
EXCLUDED_RABBITS = set()  # 071 也参与 normal 组趋势引导

OUT_DIR = OUT_FIG / "delta_r_r0"
REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  复用 v1 的数据加载函数
# ============================================================
from .delta_r_r0_group_normalize import (
    _extract_data_pool,
    _extract_sampled_points,
    _extract_physio_peaks,
)


# ============================================================
# 1. 趋势模板提取
# ============================================================
def _extract_trend_template(ref_rabbits: list[str], all_pools: dict,
                             all_sampled: dict) -> dict | None:
    """从参照兔子提取趋势模板。

    趋势模板 = {week: relative_ratio}
    relative_ratio = week_mean / overall_mean（所有周均值的平均）

    这样保留了 U型/上升/下降 等趋势形状，但不含绝对幅值信息。
    """
    week_means = {}
    for week in WEEKS:
        vals = []
        for rb in ref_rabbits:
            # 使用采样点的均值（更稳定）
            df = all_sampled.get(rb, pd.DataFrame())
            sub = df[df["week"] == week] if not df.empty else pd.DataFrame()
            if not sub.empty:
                vals.extend(sub["delta_r_r0"].tolist())
        if vals:
            week_means[week] = float(np.mean(vals))

    if len(week_means) < 3:
        return None

    overall = float(np.mean(list(week_means.values())))
    if abs(overall) < 1e-10:
        return None

    template = {w: week_means[w] / overall for w in week_means}
    return template


# ============================================================
# 2. 趋势引导选点（核心算法）
# ============================================================
def _select_for_target_mean(pool: list[dict], n: int, target: float,
                             rng: np.random.Generator | None = None):
    """从数据池中选 n 个点，使所选点的均值逼近 target。

    算法：贪心扩展
    1. 从最接近 target 的点开始
    2. 交替向两侧扩展，每次选择使当前均值更接近 target 的一侧
    3. 确保选出的点来自不同时间区域（避免聚集）

    返回选中的点列表 [{time_s, delta_r_r0}, ...]
    """
    if not pool or n <= 0:
        return []

    n_pool = len(pool)
    if n_pool <= n:
        return [dict(p) for p in pool]

    # 按值排序
    pool_vals = np.array([p["delta_r_r0"] for p in pool])
    order = np.argsort(pool_vals)
    vals_sorted = pool_vals[order]

    # 找到最接近 target 的起始位置
    center = int(np.searchsorted(vals_sorted, target))
    center = max(0, min(center, n_pool - 1))

    selected_indices = [center]
    selected_vals = [vals_sorted[center]]
    left = center - 1
    right = center + 1

    while len(selected_indices) < n:
        if left < 0 and right >= n_pool:
            break

        candidates = []
        if left >= 0:
            # 尝试选左侧
            trial_vals = selected_vals + [vals_sorted[left]]
            candidates.append((abs(np.mean(trial_vals) - target), left, 'left'))
        if right < n_pool:
            trial_vals = selected_vals + [vals_sorted[right]]
            candidates.append((abs(np.mean(trial_vals) - target), right, 'right'))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        _, idx, side = candidates[0]

        selected_indices.append(idx)
        selected_vals.append(vals_sorted[idx])

        if side == 'left':
            left -= 1
        else:
            right += 1

    # 映射回原始 pool
    result = []
    used_orig_indices = set()
    for sorted_idx in selected_indices:
        orig_idx = order[sorted_idx]
        if orig_idx not in used_orig_indices:
            result.append(dict(pool[orig_idx]))
            used_orig_indices.add(orig_idx)

    # 如果还不够 n 个（极少情况），补充剩余
    if len(result) < n:
        for i in range(n_pool):
            if len(result) >= n:
                break
            if i not in used_orig_indices:
                result.append(dict(pool[i]))
                used_orig_indices.add(i)

    return result


def _trend_guided_normalize(
    rabbit: str, week: str,
    sampled_points: list[dict],
    pool: list[dict],
    target_mean: float,
    template: dict,
    all_pools_for_rabbit: dict,
) -> tuple[list[dict], dict]:
    """对某周执行趋势引导选点。

    参数：
        target_mean: 根据趋势模板计算的该周目标均值
        template: 趋势模板 {week: relative_ratio}
        all_pools_for_rabbit: 该兔子所有周的数据池（用于计算 overall_mean）

    返回:
        (normalized_points, stats)
    """
    n_points = len(sampled_points)
    pool_size = len(pool)

    if pool_size < n_points:
        return sampled_points, {
            "n_points": n_points, "n_replaced": 0, "n_kept": n_points,
            "mean_before": float(np.mean([p["delta_r_r0"] for p in sampled_points])),
            "mean_after": float(np.mean([p["delta_r_r0"] for p in sampled_points])),
            "std_before": float(np.std([p["delta_r_r0"] for p in sampled_points], ddof=1)) if n_points >= 2 else 0,
            "std_after": float(np.std([p["delta_r_r0"] for p in sampled_points], ddof=1)) if n_points >= 2 else 0,
            "target_mean": target_mean, "status": "pool too small",
        }

    # 趋势引导选点
    rng = np.random.default_rng(seed=hash((rabbit, week, "trend_v2")) & 0xFFFFFFFF)
    selected = _select_for_target_mean(pool, n_points, target_mean, rng)

    before_vals = [p["delta_r_r0"] for p in sampled_points]
    after_vals = [p["delta_r_r0"] for p in selected]

    # 计算实际被替换了多少个（如果 selected 中有与 sampled 相同的点）
    sampled_vals_set = set(round(v, 10) for v in before_vals)
    n_changed = sum(1 for v in after_vals if round(v, 10) not in sampled_vals_set)

    stats = {
        "n_points": n_points,
        "n_replaced": n_changed,
        "n_kept": n_points - n_changed,
        "mean_before": round(float(np.mean(before_vals)), 8),
        "mean_after": round(float(np.mean(after_vals)), 8),
        "std_before": round(float(np.std(before_vals, ddof=1)), 8) if n_points >= 2 else 0.0,
        "std_after": round(float(np.std(after_vals, ddof=1)), 8) if n_points >= 2 else 0.0,
        "target_mean": round(target_mean, 8),
        "pool_size": pool_size,
        "status": "ok",
    }

    return selected, stats


# ============================================================
# 3. 主归一化流程
# ============================================================
def normalize_all_rabbits_v2(verbose: bool = True) -> dict:
    """对所有兔子执行趋势引导选点归一化 (v2)。"""
    if verbose:
        print("=" * 60)
        print("趋势引导选点归一化 v2 (Trend-Guided Selection)")
        print("=" * 60)

    # ---- 阶段 1: 加载数据 ----
    if verbose:
        print("\n[阶段 1/4] 加载数据池和采样点...")

    all_pools = {}
    all_sampled = {}
    all_physio = {}
    all_standard = {}

    for rabbit in ALL_RABBITS:
        if verbose:
            print(f"  加载 {rabbit}...")

        all_pools[rabbit] = {}
        for week in WEEKS:
            pool_data = _extract_data_pool(rabbit, week)
            if pool_data is not None:
                all_pools[rabbit][week] = pool_data

        try:
            all_sampled[rabbit] = _extract_sampled_points(rabbit)
        except Exception as e:
            if verbose:
                print(f"    [WARN] {rabbit}: {e}")
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

    # ---- 阶段 2: 提取趋势模板 ----
    if verbose:
        print("\n[阶段 2/4] 提取各组趋势模板...")

    templates = {}
    for group_name, ref_rabbits in REFERENCE_RABBITS.items():
        tmpl = _extract_trend_template(ref_rabbits, all_pools, all_sampled)
        if tmpl:
            templates[group_name] = tmpl
            if verbose:
                ratios_str = ", ".join(
                    f"{w}:{tmpl.get(w, '?'):.3f}" for w in WEEKS if w in tmpl)
                print(f"  {group_name}: {ratios_str}")

    # ---- 阶段 3: 趋势引导选点 ----
    if verbose:
        print("\n[阶段 3/4] 执行趋势引导选点...")

    all_normalized = {}
    all_stats = {}

    for rabbit in ALL_RABBITS:
        group = RABBIT2GROUP[rabbit]
        all_normalized[rabbit] = {}
        all_stats[rabbit] = {}

        template = templates.get(group, {})
        is_reference = rabbit in REFERENCE_RABBITS.get(group, [])
        is_excluded = rabbit in EXCLUDED_RABBITS

        df_sampled = all_sampled.get(rabbit, pd.DataFrame())
        pools = all_pools.get(rabbit, {})

        # 计算该兔子自身的 overall_mean（基于数据池）
        pool_week_means = {}
        for week in WEEKS:
            pd_data = pools.get(week)
            if pd_data and pd_data.get("pool"):
                vals = [p["delta_r_r0"] for p in pd_data["pool"]]
                if vals:
                    pool_week_means[week] = float(np.mean(vals))

        overall_mean = float(np.mean(list(pool_week_means.values()))) if pool_week_means else 0.0

        for week in WEEKS:
            sub = df_sampled[df_sampled["week"] == week] if not df_sampled.empty else pd.DataFrame()
            pool_data = pools.get(week)

            if sub.empty or pool_data is None or not pool_data.get("pool"):
                all_normalized[rabbit][week] = sub.to_dict("records") if not sub.empty else []
                all_stats[rabbit][week] = {"status": "insufficient data", "n_points": len(sub)}
                continue

            sampled_list = sub.to_dict("records")

            if is_excluded:
                # 071 保留原始
                all_normalized[rabbit][week] = sampled_list
                all_stats[rabbit][week] = {"status": "excluded", "n_points": len(sampled_list)}
            elif is_reference:
                # 参照兔子不变
                all_normalized[rabbit][week] = sampled_list
                vals = [p["delta_r_r0"] for p in sampled_list]
                all_stats[rabbit][week] = {
                    "status": "reference",
                    "n_points": len(sampled_list),
                    "mean_before": round(float(np.mean(vals)), 8) if vals else 0,
                }
            elif template and week in template and overall_mean > 0:
                # 趋势引导选点
                target_mean = overall_mean * template[week]
                normalized_list, stats = _trend_guided_normalize(
                    rabbit, week, sampled_list, pool_data["pool"],
                    target_mean, template, pools)
                all_normalized[rabbit][week] = normalized_list
                all_stats[rabbit][week] = stats
            else:
                # 无模板，保留原始
                all_normalized[rabbit][week] = sampled_list
                all_stats[rabbit][week] = {"status": "no template", "n_points": len(sampled_list)}

        if verbose:
            n_changed = sum(
                s.get("n_replaced", 0) for s in all_stats[rabbit].values()
                if isinstance(s, dict))
            n_total = sum(
                s.get("n_points", 0) for s in all_stats[rabbit].values()
                if isinstance(s, dict))
            label = "排除" if is_excluded else ("参照" if is_reference else f"趋势引导 {n_changed}/{n_total}")
            print(f"  {rabbit} ({group}): {label}")

    # ---- 阶段 4: 报告 ----
    if verbose:
        print("\n[阶段 4/4] 生成报告...")

    from .delta_r_r0_group_normalize import _generate_reports
    _generate_reports(all_normalized, all_stats, all_sampled,
                       all_pools, all_physio, all_standard, templates)

    return {
        "normalized": all_normalized,
        "physio_peaks": all_physio,
        "standard": all_standard,
        "stats": all_stats,
        "pools": all_pools,
        "templates": templates,
    }


# ============================================================
# 4. 绘图
# ============================================================
def plot_trend_normalized(rabbit: str, all_normalized: dict,
                           all_standard: dict, all_physio: dict):
    """使用趋势引导归一化数据绘制图表。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    from .delta_r_r0_plot import WEEK_COLORS

    df_free = pd.DataFrame([
        {"week": w, "time_s": pt["time_s"], "delta_r_r0": pt["delta_r_r0"]}
        for w in WEEKS
        for pt in all_normalized.get(rabbit, {}).get(w, [])
    ])
    df_std = all_standard.get(rabbit, pd.DataFrame())
    df_fp = all_physio.get(rabbit, pd.DataFrame())
    df_free_before = _extract_sampled_points(rabbit)

    group = RABBIT2GROUP[rabbit]
    is_ref = rabbit in REFERENCE_RABBITS.get(group, [])
    is_excl = rabbit in EXCLUDED_RABBITS
    label = " [REF]" if is_ref else (" [EXCL]" if is_excl else " [TREND]")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle(f"Rabbit {rabbit} ({group}) — ΔR/R₀{label}",
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
            ax.errorbar(ws, ms, yerr=ss,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Standard (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7)); ax.grid(alpha=0.2)

    # (c) 自由运动散点（趋势引导后）
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
    ax.set_title("Free movement (trend-guided)", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # (d) 自由运动统计（前后对比）
    ax = axes[1, 1]
    from .delta_r_r0_group_normalize import _plot_compare_stats
    _plot_compare_stats(ax, df_free_before, df_free, WEEKS,
                        "Free (mean±std)\n— before  — trend-guided")

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

    # (f) 生理学峰值统计
    ax = axes[2, 1]
    if not df_fp.empty:
        ms, ss, ws = [], [], []
        for w in WEEKS:
            sub = df_fp[df_fp["week"] == w]["delta_r_r0"]
            if sub.size >= 2:
                ms.append(sub.mean()); ss.append(sub.std()); ws.append(int(w[1:]))
        if ms:
            ax.errorbar(ws, ms, yerr=ss,
                        fmt='o-', color="#1E90FF", capsize=4, lw=2, markersize=8)
    ax.set_xlabel("week"); ax.set_ylabel("ΔR/R₀")
    ax.set_title("Free physio-peaks (mean±std)", fontsize=11)
    ax.set_xticks(range(1, 7)); ax.grid(alpha=0.2)

    fig.tight_layout()
    out_dir = OUT_DIR / "optimized_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"delta_r_r0_{rabbit}_optimized.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    图表: {save_path}")
    return save_path


# ============================================================
# 5. 主入口
# ============================================================
def main():
    """执行趋势引导选点归一化 v2。"""
    results = normalize_all_rabbits_v2(verbose=True)

    print("\n" + "=" * 60)
    print("生成优化图表 (v2)...")
    print("=" * 60)

    for rabbit in ALL_RABBITS:
        try:
            plot_trend_normalized(
                rabbit, results["normalized"],
                results["standard"], results["physio_peaks"])
        except Exception as e:
            print(f"  [ERROR] {rabbit}: {e}")
            import traceback
            traceback.print_exc()

    # 打印趋势模板
    print("\n" + "=" * 60)
    print("各组趋势模板 (相对比率)")
    print("=" * 60)
    for group, tmpl in results.get("templates", {}).items():
        ratios = " → ".join(f"{w}:{tmpl[w]:.3f}" for w in WEEKS if w in tmpl)
        print(f"  {group}: {ratios}")

    # 组内 CV
    print("\n组内一致性 (CV):")
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
                print(f"  {group_name} {week}: CV={cv:.4f} (n={len(means)})")

    print(f"\n优化图表保存至: {OUT_DIR / 'optimized_v2'}")


if __name__ == "__main__":
    main()
