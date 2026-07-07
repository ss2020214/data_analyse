# 个体级深入分析报告（v5）

> 数据：`features_per_session.csv`，58 sessions，Normal n=6 / Pathological n=3 / Sham n=1。

> 视角：先做跨周池化（W1-W2 早期 vs W5-W6 晚期），再对每只兔子做 W1→W6 时间演变形状分类，最后汇总各组的优势形状。


## 1. 在晚期最能区分 Normal / Pathological 的特征 (Top-8)

Mann-Whitney U（双侧）on W5-W6 池化样本，按 p 值升序：

| feature | n_late_each | p (N vs P, late) | p (N vs P, early) |
|---|---:|---:|---:|
| `burst_duration_median` | 8 | 0.002 | 0.001 |
| `burst_duration_max` | 8 | 0.021 | 0.015 |
| `ipi_cv` | 8 | 0.028 | 0.505 |
| `amp_norm_mean` | 8 | 0.038 | 0.130 |
| `baseline_rms` | 8 | 0.038 | 0.065 |
| `jumps_per_min` | 8 | 0.038 | 0.798 |
| `baseline_short_drift_p95` | 8 | 0.038 | 0.050 |
| `burst_density_per_min` | 8 | 0.065 | 0.002 |


## 2. 时间演变形状最不一致的特征（normal 与 pathological 主导形状不同）

| feature | normal 主导 | pathological 主导 | slope_normal | slope_path | |Δslope| |
|---|:---:|:---:|---:|---:|---:|
| `psd_total_power` | monotone-up | inverted-U | -1985897951355.831 | 135949887707.539 | 2121847839063.370 |
| `amp_trend_slope_per_min` | monotone-up | U-shape | -55726.733 | -16389.121 | 39337.612 |
| `baseline_mean` | inverted-U | flat | -2846.071 | 19731.195 | 22577.266 |
| `baseline_short_drift_p95` | monotone-up | inverted-U | -3735.562 | 2948.365 | 6683.928 |
| `baseline_rms` | monotone-up | inverted-U | 383.175 | 1175.159 | 791.983 |
| `amp_mean` | U-shape | inverted-U | -405.987 | -903.314 | 497.328 |
| `jumps_per_active_min` | inverted-U | monotone-up | 2.079 | 24.989 | 22.910 |
| `burst_count` | inverted-U | U-shape | -8.393 | -0.357 | 8.036 |
| `baseline_drift_pct` | monotone-down | inverted-U | -0.684 | 2.963 | 3.647 |
| `burst_duration_median` | inverted-U | U-shape | 0.679 | -1.182 | 1.861 |


## 3. 各组在关键特征上的形状分布与平均斜率

| feature | group | n | dominant shape | shape distribution | mean slope |
|---|---|---:|:---:|---|---:|
| `amp_norm_mean` | normal | 4 | U-shape | U-shape:2; monotone-up:1; inverted-U:1 | -0.002 |
| `amp_norm_mean` | pathological | 4 | inverted-U | inverted-U:2; monotone-down:2 | -0.005 |
| `amp_norm_mean` | infection | 3 | monotone-down | monotone-down:1; inverted-U:1; monotone-up:1 | 0.002 |
| `amp_norm_mean` | sham | 4 | monotone-up | monotone-up:2; U-shape:1; inverted-U:1 | 0.017 |
| `amp_cv` | normal | 4 | inverted-U | inverted-U:3; U-shape:1 | -0.022 |
| `amp_cv` | pathological | 4 | inverted-U | inverted-U:2; U-shape:2 | 0.165 |
| `amp_cv` | infection | 3 | inverted-U | inverted-U:3 | -0.172 |
| `amp_cv` | sham | 4 | inverted-U | inverted-U:2; U-shape:1; monotone-up:1 | 0.267 |
| `burst_density_per_min` | normal | 4 | U-shape | U-shape:2; inverted-U:1; monotone-down:1 | -2.192 |
| `burst_density_per_min` | pathological | 4 | U-shape | U-shape:2; inverted-U:2 | 0.183 |
| `burst_density_per_min` | infection | 3 | inverted-U | inverted-U:1; monotone-up:1; U-shape:1 | 1.481 |
| `burst_density_per_min` | sham | 4 | inverted-U | inverted-U:1; flat:1; monotone-up:1; U-shape:1 | 0.468 |
| `ipi_cv` | normal | 4 | U-shape | U-shape:2; inverted-U:1; monotone-down:1 | -0.052 |
| `ipi_cv` | pathological | 4 | inverted-U | inverted-U:2; U-shape:2 | -0.110 |
| `ipi_cv` | infection | 3 | inverted-U | inverted-U:2; U-shape:1 | 0.062 |
| `ipi_cv` | sham | 4 | monotone-down | monotone-down:3; inverted-U:1 | -0.099 |
| `peak_fwhm_median` | normal | 4 | inverted-U | inverted-U:2; monotone-down:1; U-shape:1 | 0.162 |
| `peak_fwhm_median` | pathological | 4 | inverted-U | inverted-U:2; monotone-up:1; U-shape:1 | 0.229 |
| `peak_fwhm_median` | infection | 3 | inverted-U | inverted-U:2; U-shape:1 | -0.253 |
| `peak_fwhm_median` | sham | 4 | monotone-up | monotone-up:2; inverted-U:1; monotone-down:1 | 0.080 |
| `peak_asymmetry_median` | normal | 4 | inverted-U | inverted-U:3; U-shape:1 | 0.144 |
| `peak_asymmetry_median` | pathological | 4 | inverted-U | inverted-U:3; U-shape:1 | -0.111 |
| `peak_asymmetry_median` | infection | 3 | inverted-U | inverted-U:2; monotone-down:1 | -0.087 |
| `peak_asymmetry_median` | sham | 4 | U-shape | U-shape:2; inverted-U:1; monotone-down:1 | -0.092 |
| `psd_peak_freq_hz` | normal | 4 | flat | flat:2; inverted-U:2 | 0.039 |
| `psd_peak_freq_hz` | pathological | 4 | U-shape | U-shape:2; inverted-U:1; monotone-up:1 | -0.047 |
| `psd_peak_freq_hz` | infection | 3 | inverted-U | inverted-U:1; U-shape:1; flat:1 | 0.193 |
| `psd_peak_freq_hz` | sham | 4 | flat | flat:2; monotone-down:1; U-shape:1 | -0.078 |
| `psd_jump_band_ratio` | normal | 4 | inverted-U | inverted-U:2; monotone-up:1; U-shape:1 | 0.063 |
| `psd_jump_band_ratio` | pathological | 4 | inverted-U | inverted-U:3; monotone-down:1 | -0.077 |
| `psd_jump_band_ratio` | infection | 3 | inverted-U | inverted-U:1; monotone-up:1; monotone-down:1 | 0.050 |
| `psd_jump_band_ratio` | sham | 4 | monotone-down | monotone-down:2; U-shape:1; flat:1 | -0.082 |
| `silence_median_s` | normal | 4 | inverted-U | inverted-U:3; monotone-up:1 | 2.197 |
| `silence_median_s` | pathological | 4 | inverted-U | inverted-U:3; U-shape:1 | -1.037 |
| `silence_median_s` | infection | 3 | U-shape | U-shape:2; inverted-U:1 | -2.504 |
| `silence_median_s` | sham | 4 | inverted-U | inverted-U:2; U-shape:2 | -0.412 |
| `burst_duration_median` | normal | 4 | inverted-U | inverted-U:2; monotone-up:1; U-shape:1 | 0.679 |
| `burst_duration_median` | pathological | 4 | U-shape | U-shape:2; inverted-U:2 | -1.182 |
| `burst_duration_median` | infection | 3 | inverted-U | inverted-U:3 | 0.480 |
| `burst_duration_median` | sham | 4 | monotone-down | monotone-down:1; U-shape:1; inverted-U:1; monotone-up:1 | -0.446 |
| `amp_autocorr_lag1` | normal | 4 | inverted-U | inverted-U:3; monotone-up:1 | 0.249 |
| `amp_autocorr_lag1` | pathological | 4 | U-shape | U-shape:2; inverted-U:1; monotone-up:1 | -0.004 |
| `amp_autocorr_lag1` | infection | 3 | U-shape | U-shape:2; inverted-U:1 | -0.010 |
| `amp_autocorr_lag1` | sham | 4 | monotone-up | monotone-up:2; U-shape:1; inverted-U:1 | 0.210 |
| `ipi_trend_slope_s_per_min` | normal | 4 | inverted-U | inverted-U:2; U-shape:2 | 1.382 |
| `ipi_trend_slope_s_per_min` | pathological | 4 | inverted-U | inverted-U:3; U-shape:1 | 1.359 |
| `ipi_trend_slope_s_per_min` | infection | 3 | monotone-down | monotone-down:1; monotone-up:1; inverted-U:1 | -0.630 |
| `ipi_trend_slope_s_per_min` | sham | 4 | inverted-U | inverted-U:2; U-shape:1; monotone-up:1 | 1.130 |


## 4. 关键特征早期 vs 晚期的组内对比（W1-W2 → W5-W6）

| feature | group | mean(early) | mean(late) | Δ | %变化 | p |
|---|---|---:|---:|---:|---:|---:|
| `amp_norm_mean` | normal | 0.013 | 0.009 | -0.003 | -26.3% | 0.645 |
| `amp_norm_mean` | pathological | 0.008 | 0.004 | -0.004 | -51.5% | 0.574 |
| `amp_norm_mean` | infection | 0.011 | 0.013 | 0.002 | 17.1% | 0.937 |
| `amp_norm_mean` | sham | 0.012 | — | — | —% | — |
| `amp_cv` | normal | 0.564 | 0.589 | 0.025 | 4.4% | 0.798 |
| `amp_cv` | pathological | 0.519 | 0.694 | 0.175 | 33.8% | 0.279 |
| `amp_cv` | infection | 0.623 | 0.447 | -0.176 | -28.3% | 0.015 |
| `amp_cv` | sham | 0.568 | — | — | —% | — |
| `burst_density_per_min` | normal | 6.086 | 4.700 | -1.386 | -22.8% | 0.161 |
| `burst_density_per_min` | pathological | 3.153 | 3.402 | 0.248 | 7.9% | 0.442 |
| `burst_density_per_min` | infection | 5.579 | 6.591 | 1.011 | 18.1% | 0.589 |
| `burst_density_per_min` | sham | 5.540 | — | — | —% | — |
| `ipi_cv` | normal | 0.583 | 0.582 | -0.001 | -0.1% | 0.878 |
| `ipi_cv` | pathological | 0.582 | 0.481 | -0.101 | -17.3% | 0.050 |
| `ipi_cv` | infection | 0.547 | 0.599 | 0.052 | 9.5% | 0.240 |
| `ipi_cv` | sham | 0.556 | — | — | —% | — |
| `peak_fwhm_median` | normal | 0.571 | 0.788 | 0.218 | 38.1% | 0.645 |
| `peak_fwhm_median` | pathological | 0.557 | 0.649 | 0.092 | 16.5% | 0.279 |
| `peak_fwhm_median` | infection | 0.517 | 0.312 | -0.205 | -39.6% | 0.002 |
| `peak_fwhm_median` | sham | 0.540 | — | — | —% | — |
| `peak_asymmetry_median` | normal | 0.080 | 0.148 | 0.068 | 84.8% | 0.645 |
| `peak_asymmetry_median` | pathological | 0.143 | 0.035 | -0.109 | -75.7% | 0.161 |
| `peak_asymmetry_median` | infection | 0.049 | 0.005 | -0.044 | -90.2% | 0.699 |
| `peak_asymmetry_median` | sham | 0.080 | — | — | —% | — |
| `psd_peak_freq_hz` | normal | 0.703 | 0.762 | 0.059 | 8.3% | 0.642 |
| `psd_peak_freq_hz` | pathological | 0.752 | 0.703 | -0.049 | -6.5% | 0.178 |
| `psd_peak_freq_hz` | infection | 0.703 | 0.807 | 0.104 | 14.8% | 0.784 |
| `psd_peak_freq_hz` | sham | 0.723 | — | — | —% | — |
| `psd_jump_band_ratio` | normal | 0.361 | 0.429 | 0.068 | 18.9% | 0.645 |
| `psd_jump_band_ratio` | pathological | 0.430 | 0.379 | -0.051 | -11.8% | 0.645 |
| `psd_jump_band_ratio` | infection | 0.279 | 0.319 | 0.041 | 14.6% | 0.818 |
| `psd_jump_band_ratio` | sham | 0.367 | — | — | —% | — |
| `silence_median_s` | normal | 5.559 | 6.774 | 1.215 | 21.9% | 0.574 |
| `silence_median_s` | pathological | 8.686 | 6.657 | -2.029 | -23.4% | 0.248 |
| `silence_median_s` | infection | 5.936 | 4.157 | -1.779 | -30.0% | 0.240 |
| `silence_median_s` | sham | 4.801 | — | — | —% | — |
| `burst_duration_median` | normal | 2.308 | 2.771 | 0.463 | 20.0% | 0.574 |
| `burst_duration_median` | pathological | 7.021 | 6.059 | -0.962 | -13.7% | 0.462 |
| `burst_duration_median` | infection | 2.204 | 2.753 | 0.549 | 24.9% | 0.394 |
| `burst_duration_median` | sham | 3.222 | — | — | —% | — |
| `amp_autocorr_lag1` | normal | 0.255 | 0.435 | 0.180 | 70.3% | 0.279 |
| `amp_autocorr_lag1` | pathological | 0.323 | 0.331 | 0.009 | 2.7% | 0.442 |
| `amp_autocorr_lag1` | infection | 0.248 | 0.253 | 0.005 | 2.1% | 1.000 |
| `amp_autocorr_lag1` | sham | 0.212 | — | — | —% | — |
| `ipi_trend_slope_s_per_min` | normal | -0.111 | -0.142 | -0.031 | -27.6% | 0.959 |
| `ipi_trend_slope_s_per_min` | pathological | -0.160 | 0.979 | 1.138 | 712.0% | 0.028 |
| `ipi_trend_slope_s_per_min` | infection | 1.078 | 0.839 | -0.239 | -22.2% | 0.699 |
| `ipi_trend_slope_s_per_min` | sham | -0.103 | — | — | —% | — |


## 5. 自动总结（基于以上数据）

- **normal 组以“倒U（先升后降）”为主导形状的特征**：`active_duration_s`、`amp_autocorr_lag1`、`amp_cv`、`baseline_mean`、`baseline_zero_cross_per_min`、`burst_count`、`burst_duration_max`、`burst_duration_median` 等 19 项

- **pathological 组以“倒U”为主导形状的特征**：`active_duration_s`、`amp_cv`、`amp_mean`、`amp_norm_mean`、`baseline_drift_pct`、`baseline_rms`、`baseline_short_drift_p95`、`burst_duration_max` 等 19 项

- **normal 组以“U 型（先降后升）”为主导**：`amp_mean`、`amp_norm_mean`、`burst_density_per_min`、`ipi_cv`、`ipi_median`

- **pathological 组以“U 型”为主导**：`amp_autocorr_lag1`、`amp_trend_slope_per_min`、`baseline_slope_pct_min`、`burst_count`、`burst_density_per_min`、`burst_duration_median`、`psd_peak_freq_hz`

- **normal 组单调上升的特征**：`amp_trend_slope_per_min`、`baseline_rms`、`baseline_short_drift_p95`、`psd_total_power`

- **pathological 组单调上升的特征**：`jumps_per_active_min`、`peak_rise_time_median`

- **normal 组单调下降的特征**：`baseline_drift_pct`、`baseline_slope_pct_min`

- **pathological 组单调下降的特征**：`baseline_zero_cross_per_min`、`ipi_median`


## 6. 输出物索引

- 详细个体特征表：`output/tables/features_per_session.csv`
- 早晚池化对比：`output/tables/pooled_early_vs_late.csv`
- 个体形状分类：`output/tables/temporal_pattern_per_rabbit.csv`
- 组层形状汇总：`output/tables/temporal_pattern_group_summary.csv`
- 个体轨迹图：`output/figures/trajectory_<feature>.png`（去组均值）
- rabbit×week 热图：`output/figures/heatmap_<feature>.png`
