# -*- coding: utf-8 -*-
"""临时单元自检：验证 extract_rest_resistance_r0 与 peak_trough_delta。"""
import numpy as np
from analysis.peak_trough_delta import extract_rest_resistance_r0, peak_trough_delta

fs = 5.0
t = np.arange(0, 20, 1.0 / fs)

# 1) 构造已知小信号：恒定基线 1000Ω，中间一个矩形峰 1000->1100->1000
r = np.full(t.size, 1000.0)
peak_lo = int(6.0 * fs)
peak_hi = int(7.0 * fs + 1)
r[peak_lo:peak_hi] = 1100.0

# 活跃窗口（覆盖峰及其上升/下降沿）
active = np.zeros(t.size, dtype=bool)
active[peak_lo - 2:peak_hi + 2] = True

# a) 纯静息段 active_mask：静息全为 1000 → R0 = P20 = 1000
R0, ratio = extract_rest_resistance_r0(t, r, active_mask=active, r_quantile=0.20)
print(f"[T1] R0={R0:.3f}  rest_ratio={ratio:.3f}")
assert abs(R0 - 1000.0) < 1e-6, f"R0 应为 1000，得到 {R0}"
assert ratio > 0.5

# b) 纯静息信号（无峰）返回值即整段 P20
r2 = np.full(t.size, 1200.0)
R0b, _ = extract_rest_resistance_r0(t, r2, active_mask=None, r_quantile=0.20)
print(f"[T2] 纯静息无 active: R0={R0b:.3f}")
assert abs(R0b - 1200.0) < 1e-6, f"应返回 1200，得到 {R0b}"

# c) peak_trough_delta：峰阻值 1100，谷 1000，R0=1000 → delta=0.1
# 用 detrended：需要给 detrended 让峰被检出，且谷在峰脚下（1000）
bl = np.full(t.size, 1000.0)
detrended = r - bl
pos = np.array([peak_lo], dtype=int)  # 峰位置（升到顶的第一点）
delta, valley_r, lb = peak_trough_delta(r, detrended, pos, R0, fs)
print(f"[T3] delta={delta[0]:.4f}  valley_r={valley_r[0]:.1f}  left_base={lb[0]}")
r_peak = r[pos[0]]
expected = (r_peak - valley_r[0]) / R0
print(f"     r_peak={r_peak}  hand calc={(1100-1000)/1000:.4f}")
assert abs(delta[0] - expected) < 1e-9
assert abs(delta[0] - 0.1) < 1e-3, f"应为 0.1，得到 {delta[0]}"
assert valley_r[0] <= 1000.0 + 1e-6

# d) 谷回退：left_base 过近时回退到峰±0.5s 局部最小
# 构造 detrended 使 find_peaks 的 left_base 紧贴峰（台阶峰）
detrended2 = r[:, None].flatten().astype(float)
# 使用更陡台阶，让 left_base 几乎贴住峰
detrended2 = np.concatenate([np.zeros(peak_lo), np.ones(peak_hi - peak_lo + 1)
                             * 100.0, np.zeros(t.size - peak_hi - 1)])
delta2, valley2, lb2 = peak_trough_delta(r, detrended2, pos, R0, fs)
print(f"[T4] 回退后 valley_r={valley2[0]:.1f}  left_base={lb2[0]}  delta={delta2[0]:.4f}")
# 谷回退后应取峰附近最小值 1000
assert abs(valley2[0] - 1000.0) < 1e-6

print("\n全部单元自检通过 ✔")