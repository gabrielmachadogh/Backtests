# Best patterns (by side) — BTC_USDT

- MIN_TRADES: 30
- Buckets: low/high [10, 15, 20, 25, 30, 40, 50, 60]
- Thresholds: thr_lo/thr_hi

## 1h | DL | long | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| pos_in_range_n | high10 | 30 | 70% | 0.908935 |  |
| pos_in_range_n | high15 | 44 | 68,2% | 0.890484 |  |
| vol_z | high20 | 59 | 67,8% | 0.678341 |  |
| vol_z | high10 | 30 | 66,7% | 1.75614 |  |
| pos_in_range_n | high20 | 59 | 66,1% | 0.872753 |  |
| ret_5_pct | high15 | 44 | 65,9% | 1.01729 |  |
| vol_z | high30 | 88 | 63,6% | 0.235014 |  |
| pos_in_range_n | high25 | 74 | 63,5% | 0.85915 |  |
| vol_z | high25 | 74 | 63,5% | 0.420023 |  |
| ret_1_pct | high10 | 30 | 63,3% | 0.566337 |  |
| ret_5_pct | high10 | 30 | 63,3% | 1.1931 |  |
| pos_in_range_n | high30 | 88 | 62,5% | 0.850742 |  |
| dist_to_high_n_pct | low30 | 88 | 62,5% |  | 0.435343 |
| bars_since_new_low | low10 | 32 | 62,5% |  | 10 |
| vol_z | high40 | 117 | 62,4% | -0.048183 |  |
| dist_to_high_n_pct | low25 | 74 | 62,2% |  | 0.382305 |
| context_bars_since_extreme | high15 | 44 | 61,4% | 20.2 |  |
| ret_1_pct | high15 | 44 | 61,4% | 0.435342 |  |
| vol_z | high15 | 44 | 61,4% | 1.15754 |  |
| bars_since_new_high | high15 | 44 | 61,4% | 20.2 |  |

## 1h | DL | long | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| pos_in_range_n | high15 | 44 | 61,4% | 0.89089 |  |
| pos_in_range_n | high20 | 58 | 56,9% | 0.873093 |  |
| dist_to_high_n_pct | low20 | 58 | 55,2% |  | 0.318404 |
| vol_z | high20 | 58 | 55,2% | 0.651402 |  |
| context_bars_since_extreme | high15 | 44 | 54,5% | 20.65 |  |
| vol_z | high15 | 44 | 54,5% | 1.10853 |  |
| bars_since_new_high | high15 | 44 | 54,5% | 20.65 |  |
| pos_in_range_n | high25 | 73 | 53,4% | 0.859648 |  |
| vol_z | high25 | 73 | 53,4% | 0.392845 |  |
| dist_to_high_n_pct | low15 | 44 | 52,3% |  | 0.250762 |
| pos_in_range_n | high30 | 87 | 51,7% | 0.850831 |  |
| vol_z | high40 | 116 | 50,9% | -0.0531492 |  |
| dist_to_high_n_pct | low25 | 73 | 50,7% |  | 0.378532 |
| dist_to_high_n_pct | low30 | 87 | 50,6% |  | 0.4337 |
| vol_z | high30 | 87 | 50,6% | 0.223102 |  |
| ret_1_pct | high20 | 58 | 50% | 0.368223 |  |
| bars_since_new_low | low10 | 30 | 50% |  | 10 |
| vol_z | high50 | 145 | 48,3% | -0.204507 |  |
| atr_pct | low20 | 58 | 48,3% |  | 0.430909 |
| range_pct | high20 | 58 | 48,3% | 0.919271 |  |

## 1h | DL | long | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| pos_in_range_n | high15 | 44 | 50% | 0.891161 |  |
| vol_z | high25 | 72 | 48,6% | 0.387007 |  |
| dist_to_high_n_pct | low20 | 58 | 48,3% |  | 0.318119 |
| vol_z | high20 | 58 | 48,3% | 0.641641 |  |
| dist_to_high_n_pct | low15 | 44 | 47,7% |  | 0.249002 |
| vol_z | high15 | 44 | 47,7% | 1.0721 |  |
| bars_since_new_low | low10 | 30 | 46,7% |  | 10 |
| pos_in_range_n | high20 | 58 | 46,6% | 0.873312 |  |
| pos_in_range_n | high25 | 72 | 45,8% | 0.860839 |  |
| ret_5_pct | high15 | 44 | 45,5% | 1.02065 |  |
| vol_z | high40 | 115 | 45,2% | -0.0531691 |  |
| vol_z | high30 | 87 | 44,8% | 0.219822 |  |
| dist_to_high_n_pct | low25 | 72 | 44,4% |  | 0.381047 |
| pos_in_range_n | high30 | 87 | 43,7% | 0.850752 |  |
| dist_to_high_n_pct | low30 | 87 | 43,7% |  | 0.43467 |
| context_bars_since_extreme | high15 | 44 | 43,2% | 20.95 |  |
| bars_since_new_high | high15 | 44 | 43,2% | 20.95 |  |
| atr_pct | low20 | 58 | 43,1% |  | 0.429634 |
| ret_1_pct | high20 | 58 | 43,1% | 0.352044 |  |
| atr_pct | low30 | 87 | 42,5% |  | 0.498848 |

## 1h | DL | short | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| pullback_from_new_high_pct | low15 | 37 | 64,9% |  | 1.64267 |
| dist_to_sma80_pct | high25 | 61 | 63,9% | -0.804076 |  |
| pullback_from_new_high_pct | low25 | 61 | 63,9% |  | 2.22365 |
| context_bars_since_extreme | high10 | 30 | 63,3% | 21 |  |
| bars_since_new_low | high10 | 30 | 63,3% | 21 |  |
| clv | low20 | 49 | 63,3% |  | 0.0832347 |
| pullback_from_new_high_pct | low20 | 49 | 63,3% |  | 1.98847 |
| pullback_from_new_high_pct | low30 | 73 | 63% |  | 2.44072 |
| ma_gap_pct | high25 | 61 | 62,3% | -0.53212 |  |
| context_bars_since_extreme | high15 | 37 | 62,2% | 17.55 |  |
| rsi | high15 | 37 | 62,2% | 45.1245 |  |
| bars_since_new_low | high15 | 37 | 62,2% | 17.55 |  |
| ma_gap_pct | high30 | 73 | 61,6% | -0.691005 |  |
| dist_to_sma80_pct | high20 | 49 | 61,2% | -0.704626 |  |
| clv | low25 | 61 | 60,7% |  | 0.110754 |
| pos_in_range_n | high30 | 73 | 60,3% | 0.31382 |  |
| pos_in_range_n | high40 | 98 | 60,2% | 0.273509 |  |
| context_bars_since_extreme | high20 | 50 | 60% | 7 |  |
| bars_since_new_low | high20 | 50 | 60% | 7 |  |
| pos_in_range_n | high50 | 122 | 59,8% | 0.245358 |  |

## 1h | DL | short | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| dist_to_sma80_pct | high25 | 61 | 54,1% | -0.790849 |  |
| rsi | high15 | 37 | 54,1% | 45.2124 |  |
| context_bars_since_extreme | high10 | 30 | 53,3% | 21 |  |
| bars_since_new_high | low10 | 30 | 53,3% |  | 9 |
| bars_since_new_low | high10 | 30 | 53,3% | 21 |  |
| ma_gap_pct | high25 | 61 | 52,5% | -0.531969 |  |
| pos_in_range_n | high40 | 97 | 51,5% | 0.273584 |  |
| context_bars_since_extreme | high15 | 37 | 51,4% | 18 |  |
| bars_since_new_low | high15 | 37 | 51,4% | 18 |  |
| rsi | high20 | 49 | 51% | 43.3415 |  |
| ma_gap_pct | high30 | 73 | 50,7% | -0.687374 |  |
| context_pullback_pct | high50 | 121 | 49,6% | 0.869713 |  |
| pos_in_range_n | high50 | 121 | 49,6% | 0.24644 |  |
| pullback_from_new_low_pct | high50 | 121 | 49,6% | 0.869713 |  |
| pullback_from_new_high_pct | low30 | 73 | 49,3% |  | 2.44355 |
| rsi | high25 | 61 | 49,2% | 41.89 |  |
| pullback_from_new_high_pct | low25 | 61 | 49,2% |  | 2.22932 |
| dist_to_sma80_pct | high20 | 49 | 49% | -0.702351 |  |
| bars_since_new_high | low15 | 39 | 48,7% |  | 10 |
| bars_since_new_high | high15 | 37 | 48,6% | 53 |  |

## 1h | DL | short | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| dist_to_sma80_pct | high25 | 61 | 49,2% | -0.790849 |  |
| ma_gap_pct | high25 | 61 | 47,5% | -0.531969 |  |
| context_bars_since_extreme | high10 | 30 | 46,7% | 21 |  |
| bars_since_new_high | low10 | 30 | 46,7% |  | 9 |
| bars_since_new_low | high10 | 30 | 46,7% | 21 |  |
| ma_gap_pct | high30 | 73 | 46,6% | -0.687374 |  |
| context_bars_since_extreme | high15 | 37 | 45,9% | 18 |  |
| rsi | high15 | 37 | 45,9% | 45.2124 |  |
| bars_since_new_low | high15 | 37 | 45,9% | 18 |  |
| pullback_from_new_high_pct | low25 | 61 | 44,3% |  | 2.22932 |
| dist_to_sma80_pct | high30 | 73 | 43,8% | -0.945623 |  |
| pullback_from_new_high_pct | low30 | 73 | 43,8% |  | 2.44355 |
| bars_since_new_high | low15 | 39 | 43,6% |  | 10 |
| pos_in_range_n | high40 | 97 | 43,3% | 0.273584 |  |
| slope_strength | low15 | 37 | 43,2% |  | 28.8 |
| context_pullback_pct | high50 | 121 | 43% | 0.869713 |  |
| pullback_from_new_low_pct | high50 | 121 | 43% | 0.869713 |  |
| slope_strength | low20 | 49 | 42,9% |  | 33.675 |
| dist_to_sma80_pct | high20 | 49 | 42,9% | -0.702351 |  |
| rsi | high20 | 49 | 42,9% | 43.3415 |  |

## 1h | PFR | long | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ret_5_pct | high15 | 30 | 70% | 1.01496 |  |
| pos_in_range_n | high15 | 30 | 66,7% | 0.906349 |  |
| pos_in_range_n | high25 | 50 | 66% | 0.876764 |  |
| ret_5_pct | high20 | 40 | 65% | 0.917548 |  |
| pos_in_range_n | high20 | 40 | 65% | 0.892041 |  |
| vol_z | high20 | 40 | 65% | 1.02691 |  |
| vol_z | high40 | 79 | 64,6% | 0.0375505 |  |
| slope_strength | high15 | 30 | 63,3% | 176.392 |  |
| body_pct | high15 | 30 | 63,3% | 64.5871 |  |
| ret_1_pct | high15 | 30 | 63,3% | 0.565214 |  |
| vol_z | high15 | 30 | 63,3% | 1.46606 |  |
| pos_in_range_n | high30 | 59 | 62,7% | 0.869995 |  |
| vol_z | high30 | 59 | 62,7% | 0.258838 |  |
| lower_wick_pct | low20 | 40 | 62,5% |  | 25.5881 |
| context_bars_since_extreme | high15 | 32 | 62,5% | 23 |  |
| bars_since_new_high | high15 | 32 | 62,5% | 23 |  |
| ret_1_pct | high25 | 50 | 62% | 0.419538 |  |
| ret_1_pct | high30 | 59 | 61% | 0.368223 |  |
| dist_to_high_n_pct | low40 | 79 | 60,8% |  | 0.450413 |
| bars_since_new_low | low30 | 60 | 60% |  | 16 |

## 1h | PFR | long | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| pos_in_range_n | high15 | 30 | 60% | 0.907068 |  |
| pos_in_range_n | high25 | 49 | 57,1% | 0.877098 |  |
| dist_to_high_n_pct | low15 | 30 | 56,7% |  | 0.213085 |
| vol_z | high15 | 30 | 56,7% | 1.28596 |  |
| pos_in_range_n | high20 | 39 | 56,4% | 0.89243 |  |
| vol_z | high20 | 39 | 56,4% | 0.934882 |  |
| dist_to_high_n_pct | low20 | 39 | 53,8% |  | 0.244433 |
| body_pct | high15 | 30 | 53,3% | 64.6844 |  |
| ret_1_pct | high15 | 30 | 53,3% | 0.561817 |  |
| ret_3_pct | high15 | 30 | 53,3% | 0.365014 |  |
| context_bars_since_extreme | high15 | 32 | 53,1% | 23 |  |
| bars_since_new_high | high15 | 32 | 53,1% | 23 |  |
| dist_to_high_n_pct | low25 | 49 | 53,1% |  | 0.30497 |
| pos_in_range_n | high30 | 59 | 52,5% | 0.870667 |  |
| dist_to_high_n_pct | low40 | 78 | 51,3% |  | 0.44922 |
| vol_z | high40 | 78 | 51,3% | 0.00226347 |  |
| body_pct | high20 | 39 | 51,3% | 59.0969 |  |
| ret_1_pct | high25 | 49 | 51% | 0.413224 |  |
| ret_1_pct | high30 | 59 | 50,8% | 0.362836 |  |
| dist_to_high_n_pct | low30 | 59 | 50,8% |  | 0.362857 |

## 1h | PFR | long | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| dist_to_high_n_pct | low20 | 39 | 51,3% |  | 0.243716 |
| vol_z | high20 | 39 | 51,3% | 0.925692 |  |
| pos_in_range_n | high25 | 49 | 49% | 0.877432 |  |
| dist_to_high_n_pct | low25 | 49 | 49% |  | 0.304824 |
| pos_in_range_n | high20 | 39 | 48,7% | 0.892791 |  |
| context_bars_since_extreme | high15 | 32 | 46,9% | 23 |  |
| bars_since_new_high | high15 | 32 | 46,9% | 23 |  |
| dist_to_high_n_pct | low40 | 77 | 46,8% |  | 0.449281 |
| pos_in_range_n | high30 | 58 | 46,6% | 0.871001 |  |
| dist_to_high_n_pct | low30 | 58 | 46,6% |  | 0.364885 |
| vol_z | high40 | 77 | 45,5% | 0.00209834 |  |
| dist_to_low_n_pct | low25 | 49 | 44,9% |  | 1.41783 |
| vol_z | high25 | 49 | 44,9% | 0.555168 |  |
| slope_strength | low20 | 39 | 43,6% |  | 29.5775 |
| context_pullback_pct | low20 | 39 | 43,6% |  | 0.289702 |
| atr_pct | low20 | 39 | 43,6% |  | 0.417442 |
| body_pct | high20 | 39 | 43,6% | 58.9287 |  |
| lower_wick_pct | low20 | 39 | 43,6% |  | 25.6835 |
| ret_1_pct | high20 | 39 | 43,6% | 0.449165 |  |
| pullback_from_new_high_pct | low20 | 39 | 43,6% |  | 0.289702 |

## 1h | PFR | short | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| pullback_from_new_high_pct | low20 | 35 | 65,7% |  | 1.92766 |
| pullback_from_new_high_pct | low30 | 52 | 65,4% |  | 2.3055 |
| dist_to_sma80_pct | high25 | 43 | 65,1% | -0.790832 |  |
| pullback_from_new_high_pct | low25 | 43 | 65,1% |  | 2.14118 |
| context_pullback_pct | high40 | 69 | 63,8% | 0.937614 |  |
| pullback_from_new_low_pct | high40 | 69 | 63,8% | 0.937614 |  |
| context_pullback_pct | high30 | 52 | 63,5% | 1.18453 |  |
| pullback_from_new_low_pct | high30 | 52 | 63,5% | 1.18453 |  |
| context_bars_since_extreme | high25 | 49 | 63,3% | 6 |  |
| bars_since_new_low | high25 | 49 | 63,3% | 6 |  |
| context_bars_since_extreme | high20 | 35 | 62,9% | 11.6 |  |
| dist_to_sma80_pct | high20 | 35 | 62,9% | -0.703109 |  |
| bars_since_new_low | high20 | 35 | 62,9% | 11.6 |  |
| pos_in_range_n | high40 | 69 | 62,3% | 0.254548 |  |
| slope_strength | low30 | 52 | 61,5% |  | 45.49 |
| clv | low30 | 52 | 61,5% |  | 0.0887647 |
| lower_wick_pct | low30 | 52 | 61,5% |  | 8.87647 |
| context_bars_since_extreme | high30 | 62 | 61,3% | 5 |  |
| bars_since_new_low | high30 | 62 | 61,3% | 5 |  |
| dist_to_low_n_pct | high40 | 69 | 60,9% | 0.819684 |  |

## 1h | PFR | short | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| dist_to_sma80_pct | high25 | 43 | 55,8% | -0.790783 |  |
| slope_strength | low30 | 51 | 54,9% |  | 44.5225 |
| context_pullback_pct | high30 | 51 | 54,9% | 1.21447 |  |
| pullback_from_new_low_pct | high30 | 51 | 54,9% | 1.21447 |  |
| slope_strength | low25 | 43 | 53,5% |  | 38.275 |
| rsi | high25 | 43 | 53,5% | 42.6173 |  |
| context_pullback_pct | high40 | 68 | 52,9% | 0.94593 |  |
| pullback_from_new_low_pct | high40 | 68 | 52,9% | 0.94593 |  |
| rsi | high30 | 51 | 52,9% | 41.5778 |  |
| pullback_from_new_high_pct | low30 | 51 | 52,9% |  | 2.31436 |
| context_bars_since_extreme | high20 | 34 | 52,9% | 12.4 |  |
| rsi | high20 | 34 | 52,9% | 43.7162 |  |
| pullback_from_new_high_pct | low20 | 34 | 52,9% |  | 1.91213 |
| bars_since_new_low | high20 | 34 | 52,9% | 12.4 |  |
| ret_3_pct | high40 | 68 | 51,5% | 0.0290407 |  |
| pos_in_range_n | high40 | 68 | 51,5% | 0.255947 |  |
| context_pullback_pct | high25 | 43 | 51,2% | 1.32146 |  |
| ma_gap_pct | high25 | 43 | 51,2% | -0.515437 |  |
| dist_to_low_n_pct | high25 | 43 | 51,2% | 1.09702 |  |
| pullback_from_new_high_pct | low25 | 43 | 51,2% |  | 2.11379 |

## 1h | PFR | short | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| slope_strength | low30 | 51 | 49% |  | 44.5225 |
| dist_to_sma80_pct | high25 | 43 | 48,8% | -0.790783 |  |
| rsi | high30 | 51 | 47,1% | 41.5778 |  |
| pullback_from_new_high_pct | low30 | 51 | 47,1% |  | 2.31436 |
| rsi | high20 | 34 | 47,1% | 43.7162 |  |
| slope_strength | low25 | 43 | 46,5% |  | 38.275 |
| rsi | high25 | 43 | 46,5% | 42.6173 |  |
| context_pullback_pct | high40 | 68 | 45,6% | 0.94593 |  |
| pullback_from_new_low_pct | high40 | 68 | 45,6% | 0.94593 |  |
| context_pullback_pct | high30 | 51 | 45,1% | 1.21447 |  |
| ret_3_pct | high30 | 51 | 45,1% | 0.107137 |  |
| pullback_from_new_low_pct | high30 | 51 | 45,1% | 1.21447 |  |
| ma_gap_pct | high25 | 43 | 44,2% | -0.515437 |  |
| dist_to_low_n_pct | high25 | 43 | 44,2% | 1.09702 |  |
| pullback_from_new_high_pct | low25 | 43 | 44,2% |  | 2.11379 |
| ret_3_pct | high40 | 68 | 44,1% | 0.0290407 |  |
| rsi | high40 | 68 | 44,1% | 38.0138 |  |
| pos_in_range_n | high40 | 68 | 44,1% | 0.255947 |  |
| context_bars_since_extreme | high20 | 34 | 44,1% | 12.4 |  |
| dist_to_sma80_pct | high20 | 34 | 44,1% | -0.696082 |  |

## 2h | DL | long | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ret_5_pct | high30 | 45 | 75,6% | 0.988739 |  |
| ret_5_pct | high25 | 38 | 73,7% | 1.16006 |  |
| vol_z | high20 | 30 | 73,3% | 1.18959 |  |
| vol_z | high30 | 45 | 71,1% | 0.716975 |  |
| vol_z | high25 | 38 | 71,1% | 0.924203 |  |
| ret_5_pct | high20 | 30 | 70% | 1.36917 |  |
| ret_5_pct | high50 | 75 | 66,7% | 0.656147 |  |
| ret_5_pct | high40 | 60 | 66,7% | 0.83371 |  |
| range_pct | high20 | 30 | 66,7% | 1.21916 |  |
| range_pct | high25 | 38 | 65,8% | 1.08823 |  |
| vol_z | high40 | 60 | 65% | 0.31993 |  |
| range_pct | high40 | 60 | 63,3% | 0.898529 |  |
| dist_to_sma80_pct | low20 | 30 | 63,3% |  | 1.18847 |
| clv | high20 | 30 | 63,3% | 0.909514 |  |
| upper_wick_pct | low20 | 30 | 63,3% |  | 5.40951 |
| lower_wick_pct | high20 | 30 | 63,3% | 68.6894 |  |
| range_pct | high50 | 75 | 62,7% | 0.793912 |  |
| range_pct | high30 | 45 | 62,2% | 1.02529 |  |
| lower_wick_pct | high30 | 45 | 62,2% | 61.2583 |  |
| clv | high40 | 60 | 61,7% | 0.815012 |  |

## 2h | DL | long | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| lower_wick_pct | high20 | 30 | 63,3% | 68.8245 |  |
| range_pct | high20 | 30 | 60% | 1.18676 |  |
| lower_wick_pct | high25 | 37 | 59,5% | 64.9732 |  |
| ret_5_pct | high25 | 37 | 59,5% | 1.07797 |  |
| lower_wick_pct | high30 | 44 | 59,1% | 62.2347 |  |
| ret_5_pct | high30 | 44 | 59,1% | 0.958908 |  |
| clv | high50 | 74 | 58,1% | 0.755056 |  |
| clv | high60 | 88 | 58% | 0.711779 |  |
| lower_wick_pct | high40 | 59 | 57,6% | 51.1665 |  |
| body_pct | low25 | 37 | 56,8% |  | 13.9982 |
| body_pct | low20 | 30 | 56,7% |  | 10.7385 |
| upper_wick_pct | low20 | 30 | 56,7% |  | 6.19151 |
| ret_5_pct | high20 | 30 | 56,7% | 1.30238 |  |
| vol_z | high20 | 30 | 56,7% | 1.05946 |  |
| ret_5_pct | high40 | 59 | 55,9% | 0.814348 |  |
| lower_wick_pct | high50 | 74 | 55,4% | 45.7772 |  |
| context_bars_since_extreme | high20 | 31 | 54,8% | 11 |  |
| bars_since_new_high | high20 | 31 | 54,8% | 11 |  |
| body_pct | low30 | 44 | 54,5% |  | 17.2079 |
| vol_z | high30 | 44 | 54,5% | 0.653105 |  |

## 2h | DL | long | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| lower_wick_pct | high20 | 30 | 60% | 68.8696 |  |
| lower_wick_pct | high30 | 44 | 56,8% | 62.4161 |  |
| lower_wick_pct | high25 | 37 | 56,8% | 64.9821 |  |
| lower_wick_pct | high40 | 59 | 55,9% | 51.1217 |  |
| range_pct | high20 | 30 | 53,3% | 1.1795 |  |
| body_pct | low20 | 30 | 53,3% |  | 10.6742 |
| upper_wick_pct | low20 | 30 | 53,3% |  | 6.17105 |
| ret_5_pct | high30 | 44 | 52,3% | 0.952757 |  |
| vol_z | high30 | 44 | 52,3% | 0.630121 |  |
| body_pct | low25 | 37 | 51,4% |  | 13.9069 |
| ret_5_pct | high25 | 37 | 51,4% | 1.06012 |  |
| body_pct | low40 | 59 | 50,8% |  | 24.8378 |
| lower_wick_pct | high50 | 73 | 50,7% | 45.7489 |  |
| body_pct | low30 | 44 | 50% |  | 17.0066 |
| clv | high20 | 30 | 50% | 0.900274 |  |
| ret_5_pct | high20 | 30 | 50% | 1.28783 |  |
| vol_z | high20 | 30 | 50% | 0.995532 |  |
| clv | high50 | 73 | 49,3% | 0.754847 |  |
| ret_5_pct | high50 | 73 | 49,3% | 0.629733 |  |
| ret_5_pct | high40 | 59 | 49,2% | 0.805799 |  |

## 2h | DL | short | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| range_pct | high30 | 34 | 58,8% | 1.28675 |  |
| ret_5_pct | high30 | 34 | 58,8% | -0.345546 |  |
| rsi | low30 | 34 | 58,8% |  | 27.9151 |
| clv | low40 | 45 | 57,8% |  | 0.171632 |
| context_bars_since_extreme | high30 | 42 | 57,1% | 4 |  |
| bars_since_new_low | high30 | 42 | 57,1% | 4 |  |
| clv | low30 | 34 | 55,9% |  | 0.136109 |
| ret_3_pct | high30 | 34 | 55,9% | 0.39245 |  |
| rsi | high30 | 34 | 55,9% | 37.5307 |  |
| lower_wick_pct | low40 | 45 | 55,6% |  | 12.7068 |
| rsi | low40 | 45 | 55,6% |  | 29.7753 |
| lower_wick_pct | low50 | 56 | 55,4% |  | 16.8733 |
| ret_3_pct | high50 | 56 | 55,4% | 0.0416799 |  |
| atr_pct | high60 | 67 | 53,7% | 0.878516 |  |
| lower_wick_pct | low60 | 67 | 53,7% |  | 21.883 |
| dist_to_high_n_pct | high60 | 67 | 53,7% | 3.10485 |  |
| dist_to_low_n_pct | high60 | 67 | 53,7% | 0.847429 |  |
| context_pullback_pct | high50 | 56 | 53,6% | 1.12032 |  |
| clv | low50 | 56 | 53,6% |  | 0.265387 |
| pos_in_range_n | low50 | 56 | 53,6% |  | 0.230285 |

## 2h | DL | short | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| clv | low40 | 44 | 47,7% |  | 0.18292 |
| atr_pct | high40 | 44 | 45,5% | 1.07542 |  |
| dist_to_high_n_pct | high40 | 44 | 45,5% | 3.99109 |  |
| atr_pct | high30 | 33 | 45,5% | 1.1865 |  |
| ret_3_pct | high30 | 33 | 45,5% | 0.392881 |  |
| ret_5_pct | high30 | 33 | 45,5% | -0.342405 |  |
| atr_pct | high50 | 55 | 43,6% | 0.981405 |  |
| context_bars_since_extreme | high30 | 42 | 42,9% | 4 |  |
| bars_since_new_low | high30 | 42 | 42,9% | 4 |  |
| atr_pct | high60 | 66 | 42,4% | 0.877944 |  |
| dist_to_high_n_pct | high60 | 66 | 42,4% | 3.10343 |  |
| slope_strength | high30 | 33 | 42,4% | 251.644 |  |
| clv | low30 | 33 | 42,4% |  | 0.136222 |
| range_pct | high30 | 33 | 42,4% | 1.28349 |  |
| upper_wick_pct | low30 | 33 | 42,4% |  | 31.6647 |
| rsi | low30 | 33 | 42,4% |  | 28.0553 |
| dist_to_high_n_pct | high30 | 33 | 42,4% | 4.52918 |  |
| context_pullback_pct | high50 | 55 | 41,8% | 1.12134 |  |
| ma_gap_pct | low50 | 55 | 41,8% |  | -1.96413 |
| clv | low50 | 55 | 41,8% |  | 0.268804 |

## 2h | DL | short | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| clv | low40 | 44 | 40,9% |  | 0.19066 |
| atr_pct | high30 | 33 | 39,4% | 1.18314 |  |
| dist_to_high_n_pct | high40 | 44 | 38,6% | 4.00335 |  |
| context_bars_since_extreme | high25 | 41 | 36,6% | 4 |  |
| context_bars_since_extreme | high30 | 41 | 36,6% | 4 |  |
| bars_since_new_low | high25 | 41 | 36,6% | 4 |  |
| bars_since_new_low | high30 | 41 | 36,6% | 4 |  |
| pos_in_range_n | low50 | 55 | 36,4% |  | 0.230285 |
| ret_5_pct | low40 | 44 | 36,4% |  | -0.845015 |
| ma_gap_pct | low30 | 33 | 36,4% |  | -2.65192 |
| clv | low30 | 33 | 36,4% |  | 0.136646 |
| range_pct | high30 | 33 | 36,4% | 1.25743 |  |
| upper_wick_pct | low30 | 33 | 36,4% |  | 31.5735 |
| ret_3_pct | high30 | 33 | 36,4% | 0.393313 |  |
| ret_5_pct | high30 | 33 | 36,4% | -0.35335 |  |
| pos_in_range_n | low30 | 33 | 36,4% |  | 0.1798 |
| pullback_from_new_high_pct | high30 | 33 | 36,4% | 5.93224 |  |
| ma_gap_pct | low60 | 65 | 35,4% |  | -1.48831 |
| dist_to_sma80_pct | low60 | 65 | 35,4% |  | -1.87771 |
| pullback_from_new_high_pct | high60 | 65 | 35,4% | 4.17277 |  |

## 2h | PFR | long | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ret_5_pct | high40 | 39 | 76,9% | 0.881583 |  |
| ret_5_pct | high50 | 48 | 72,9% | 0.693356 |  |
| lower_wick_pct | high40 | 39 | 71,8% | 47.7532 |  |
| ma_gap_pct | high40 | 39 | 69,2% | 2.69581 |  |
| dist_to_sma80_pct | high40 | 39 | 69,2% | 3.36084 |  |
| ret_5_pct | high60 | 58 | 69% | 0.471854 |  |
| dist_to_sma80_pct | high50 | 48 | 68,8% | 2.37545 |  |
| dist_to_sma80_pct | high60 | 58 | 67,2% | 1.99215 |  |
| body_pct | low50 | 48 | 66,7% |  | 37.2866 |
| range_pct | high40 | 39 | 66,7% | 0.968245 |  |
| rsi | high40 | 39 | 66,7% | 72.7671 |  |
| vol_z | high40 | 39 | 66,7% | 0.463286 |  |
| pullback_from_new_low_pct | high40 | 39 | 66,7% | 7.36915 |  |
| ma_gap_pct | high60 | 58 | 65,5% | 1.55175 |  |
| pos_in_range_n | low60 | 58 | 65,5% |  | 0.848397 |
| slope_strength | high50 | 48 | 64,6% | 130.225 |  |
| ma_gap_pct | high50 | 48 | 64,6% | 1.74879 |  |
| range_pct | high50 | 48 | 64,6% | 0.880671 |  |
| lower_wick_pct | high50 | 48 | 64,6% | 42.6519 |  |
| pos_in_range_n | low50 | 48 | 64,6% |  | 0.815644 |

## 2h | PFR | long | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| lower_wick_pct | high40 | 37 | 73% | 48.9288 |  |
| body_pct | low50 | 47 | 68,1% |  | 36.5838 |
| ret_5_pct | high40 | 37 | 67,6% | 0.850036 |  |
| lower_wick_pct | high50 | 47 | 66% | 43.3623 |  |
| ret_5_pct | high50 | 47 | 66% | 0.661318 |  |
| ret_3_pct | low40 | 37 | 64,9% |  | 0.00801099 |
| body_pct | low60 | 56 | 64,3% |  | 43.8795 |
| lower_wick_pct | high60 | 56 | 64,3% | 38.6147 |  |
| pos_in_range_n | low50 | 47 | 63,8% |  | 0.810769 |
| ret_5_pct | high60 | 56 | 62,5% | 0.424402 |  |
| pos_in_range_n | low60 | 56 | 62,5% |  | 0.848261 |
| context_pullback_pct | high40 | 37 | 62,2% | 0.994222 |  |
| body_pct | low40 | 37 | 62,2% |  | 27.3804 |
| pullback_from_new_high_pct | high40 | 37 | 62,2% | 0.994222 |  |
| context_pullback_pct | high50 | 47 | 61,7% | 0.842715 |  |
| dist_to_sma80_pct | high50 | 47 | 61,7% | 2.28418 |  |
| pullback_from_new_high_pct | high50 | 47 | 61,7% | 0.842715 |  |
| ma_gap_pct | high60 | 56 | 60,7% | 1.51227 |  |
| dist_to_sma80_pct | high60 | 56 | 60,7% | 1.96268 |  |
| ma_gap_pct | high50 | 47 | 59,6% | 1.73534 |  |

## 2h | PFR | long | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| lower_wick_pct | high40 | 37 | 64,9% | 48.3657 |  |
| lower_wick_pct | high50 | 46 | 58,7% | 43.0404 |  |
| ret_5_pct | high50 | 46 | 58,7% | 0.639364 |  |
| body_pct | low40 | 37 | 56,8% |  | 27.6357 |
| ret_5_pct | high40 | 37 | 56,8% | 0.837832 |  |
| body_pct | low50 | 46 | 56,5% |  | 36.5895 |
| dist_to_sma80_pct | high50 | 46 | 54,3% | 2.28196 |  |
| rsi | high50 | 46 | 54,3% | 68.15 |  |
| ret_1_pct | low40 | 37 | 54,1% |  | 0.171666 |
| rsi | high40 | 37 | 54,1% | 72.9831 |  |
| ma_gap_pct | high60 | 55 | 52,7% | 1.48359 |  |
| body_pct | low60 | 55 | 52,7% |  | 43.977 |
| lower_wick_pct | high60 | 55 | 52,7% | 38.4684 |  |
| ret_5_pct | high60 | 55 | 52,7% | 0.420364 |  |
| ma_gap_pct | high50 | 46 | 52,2% | 1.73334 |  |
| ret_1_pct | low50 | 46 | 52,2% |  | 0.221261 |
| dist_to_sma80_pct | high40 | 37 | 51,4% | 3.00026 |  |
| atr_pct | low40 | 37 | 51,4% |  | 0.786672 |
| ret_3_pct | low40 | 37 | 51,4% |  | 0.00672943 |
| vol_z | high40 | 37 | 51,4% | 0.41875 |  |

## 2h | PFR | short | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| slope_strength | high50 | 41 | 58,5% | 170.362 |  |
| rsi | low40 | 33 | 57,6% |  | 29.9037 |
| context_bars_since_extreme | high40 | 39 | 56,4% | 4 |  |
| bars_since_new_low | high40 | 39 | 56,4% | 4 |  |
| ma_gap_pct | low50 | 41 | 56,1% |  | -2.10764 |
| slope_strength | high60 | 49 | 55,1% | 136.16 |  |
| ma_gap_pct | low60 | 49 | 55,1% |  | -1.56429 |
| clv | low60 | 49 | 55,1% |  | 0.202941 |
| lower_wick_pct | low60 | 49 | 55,1% |  | 20.2941 |
| slope_strength | high40 | 33 | 54,5% | 187.677 |  |
| context_pullback_pct | high40 | 33 | 54,5% | 1.1274 |  |
| ma_gap_pct | low40 | 33 | 54,5% |  | -2.37027 |
| atr_pct | high40 | 33 | 54,5% | 1.09004 |  |
| clv | low40 | 33 | 54,5% |  | 0.128174 |
| range_pct | high40 | 33 | 54,5% | 1.20744 |  |
| lower_wick_pct | low40 | 33 | 54,5% |  | 12.8174 |
| ret_3_pct | high40 | 33 | 54,5% | 0.0351188 |  |
| ret_5_pct | high40 | 33 | 54,5% | -0.569884 |  |
| dist_to_low_n_pct | high40 | 33 | 54,5% | 0.996404 |  |
| vol_z | low40 | 33 | 54,5% |  | -0.373536 |

## 2h | PFR | short | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| atr_pct | high40 | 33 | 48,5% | 1.09659 |  |
| slope_strength | high40 | 33 | 45,5% | 187.438 |  |
| clv | low60 | 49 | 44,9% |  | 0.205853 |
| lower_wick_pct | low60 | 49 | 44,9% |  | 20.5853 |
| slope_strength | high50 | 41 | 43,9% | 166.575 |  |
| ma_gap_pct | low50 | 41 | 43,9% |  | -1.99747 |
| atr_pct | high50 | 41 | 43,9% | 0.992575 |  |
| clv | low50 | 41 | 43,9% |  | 0.164714 |
| lower_wick_pct | low50 | 41 | 43,9% |  | 16.4714 |
| ma_gap_pct | low60 | 49 | 42,9% |  | -1.5543 |
| dist_to_sma80_pct | low60 | 49 | 42,9% |  | -2.09412 |
| dist_to_high_n_pct | high60 | 49 | 42,9% | 3.25024 |  |
| ma_gap_pct | low40 | 33 | 42,4% |  | -2.37676 |
| clv | low40 | 33 | 42,4% |  | 0.136109 |
| lower_wick_pct | low40 | 33 | 42,4% |  | 13.6109 |
| ret_5_pct | high40 | 33 | 42,4% | -0.569592 |  |
| rsi | low40 | 33 | 42,4% |  | 30.0963 |
| dist_to_high_n_pct | high40 | 33 | 42,4% | 4.18798 |  |
| pullback_from_new_high_pct | high50 | 41 | 41,5% | 4.72719 |  |
| context_bars_since_extreme | high40 | 39 | 41% | 4 |  |

## 2h | PFR | short | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| atr_pct | high40 | 32 | 40,6% | 1.08676 |  |
| clv | low60 | 48 | 37,5% |  | 0.211044 |
| lower_wick_pct | low60 | 48 | 37,5% |  | 21.1044 |
| clv | low50 | 40 | 37,5% |  | 0.166763 |
| lower_wick_pct | low50 | 40 | 37,5% |  | 16.6763 |
| ma_gap_pct | low40 | 32 | 37,5% |  | -2.36702 |
| ret_5_pct | high40 | 32 | 37,5% | -0.57003 |  |
| ma_gap_pct | low60 | 48 | 35,4% |  | -1.54477 |
| dist_to_sma80_pct | low60 | 48 | 35,4% |  | -2.05658 |
| dist_to_high_n_pct | high60 | 48 | 35,4% | 3.24333 |  |
| ma_gap_pct | low50 | 40 | 35% |  | -1.96413 |
| atr_pct | high50 | 40 | 35% | 0.991079 |  |
| dist_to_sma80_pct | low40 | 32 | 34,4% |  | -2.98958 |
| clv | low40 | 32 | 34,4% |  | 0.136206 |
| lower_wick_pct | low40 | 32 | 34,4% |  | 13.6206 |
| ret_3_pct | high40 | 32 | 34,4% | 0.052673 |  |
| ret_5_pct | low40 | 32 | 34,4% |  | -0.855546 |
| dist_to_high_n_pct | high40 | 32 | 34,4% | 4.20152 |  |
| pullback_from_new_high_pct | high40 | 32 | 34,4% | 5.32325 |  |
| context_bars_since_extreme | high40 | 38 | 34,2% | 4 |  |

## 4h | DL | long | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ma_gap_pct | low40 | 34 | 61,8% |  | 2.83752 |
| dist_to_sma80_pct | low40 | 34 | 61,8% |  | 3.36774 |
| upper_wick_pct | high40 | 34 | 58,8% | 17.2067 |  |
| bars_since_new_low | low40 | 34 | 58,8% |  | 29.6 |
| ret_5_pct | low40 | 34 | 55,9% |  | 0.373118 |
| upper_wick_pct | high50 | 42 | 54,8% | 13.7529 |  |
| ret_5_pct | low50 | 42 | 54,8% |  | 0.622672 |
| bars_since_new_low | low50 | 42 | 54,8% |  | 37.5 |
| context_bars_since_extreme | high40 | 37 | 54,1% | 4 |  |
| bars_since_new_high | high40 | 37 | 54,1% | 4 |  |
| ret_1_pct | high60 | 50 | 54% | 0.0640304 |  |
| context_bars_since_extreme | high50 | 51 | 52,9% | 3 |  |
| context_bars_since_extreme | high60 | 51 | 52,9% | 3 |  |
| bars_since_new_high | high50 | 51 | 52,9% | 3 |  |
| bars_since_new_high | high60 | 51 | 52,9% | 3 |  |
| lower_wick_pct | high40 | 34 | 52,9% | 52.6777 |  |
| ret_1_pct | high40 | 34 | 52,9% | 0.362214 |  |
| pullback_from_new_low_pct | low40 | 34 | 52,9% |  | 8.98312 |
| ma_gap_pct | low50 | 42 | 52,4% |  | 3.50998 |
| dist_to_sma80_pct | low50 | 42 | 52,4% |  | 4.26432 |

## 4h | DL | long | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ma_gap_pct | low40 | 33 | 60,6% |  | 2.88162 |
| dist_to_sma80_pct | low40 | 33 | 60,6% |  | 3.43072 |
| upper_wick_pct | high40 | 33 | 54,5% | 17.486 |  |
| bars_since_new_low | low40 | 34 | 52,9% |  | 29 |
| pullback_from_new_low_pct | low40 | 33 | 51,5% |  | 8.87888 |
| ma_gap_pct | low50 | 42 | 50% |  | 3.72774 |
| dist_to_sma80_pct | low50 | 42 | 50% |  | 4.35849 |
| upper_wick_pct | high50 | 42 | 47,6% | 13.8041 |  |
| ret_1_pct | high60 | 50 | 46% | 0.0534816 |  |
| context_pullback_pct | high40 | 33 | 45,5% | 1.51635 |  |
| lower_wick_pct | high40 | 33 | 45,5% | 52.5863 |  |
| rsi | high40 | 33 | 45,5% | 71.9102 |  |
| vol_z | low40 | 33 | 45,5% |  | -0.334863 |
| pullback_from_new_high_pct | high40 | 33 | 45,5% | 1.51635 |  |
| range_pct | low50 | 42 | 45,2% |  | 1.156 |
| rsi | high50 | 42 | 45,2% | 70.3343 |  |
| bars_since_new_low | low50 | 42 | 45,2% |  | 38 |
| slope_strength | low60 | 50 | 44% |  | 244.86 |
| range_pct | low60 | 50 | 44% |  | 1.35532 |
| upper_wick_pct | high60 | 50 | 44% | 10.5859 |  |

## 4h | DL | long | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ma_gap_pct | low40 | 33 | 54,5% |  | 2.88162 |
| dist_to_sma80_pct | low40 | 33 | 54,5% |  | 3.43072 |
| upper_wick_pct | high40 | 33 | 48,5% | 17.486 |  |
| pullback_from_new_low_pct | low40 | 33 | 45,5% |  | 8.87888 |
| bars_since_new_low | low40 | 34 | 44,1% |  | 29 |
| ma_gap_pct | low50 | 42 | 42,9% |  | 3.72774 |
| dist_to_sma80_pct | low50 | 42 | 42,9% |  | 4.35849 |
| upper_wick_pct | high50 | 42 | 42,9% | 13.8041 |  |
| rsi | high40 | 33 | 42,4% | 71.9102 |  |
| vol_z | low40 | 33 | 42,4% |  | -0.334863 |
| rsi | high50 | 42 | 40,5% | 70.3343 |  |
| slope_strength | high60 | 50 | 40% | 132.653 |  |
| upper_wick_pct | high60 | 50 | 40% | 10.5859 |  |
| rsi | high60 | 50 | 40% | 64.9353 |  |
| context_pullback_pct | high40 | 33 | 39,4% | 1.51635 |  |
| lower_wick_pct | high40 | 33 | 39,4% | 52.5863 |  |
| ret_5_pct | low40 | 33 | 39,4% |  | 0.350177 |
| dist_to_high_n_pct | high40 | 33 | 39,4% | 1.48996 |  |
| pullback_from_new_high_pct | high40 | 33 | 39,4% | 1.51635 |  |
| slope_strength | high50 | 42 | 38,1% | 174.462 |  |

## 4h | DL | short | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| ret_1_pct | low60 | 34 | 55,9% |  | -0.148323 |
| ret_3_pct | low60 | 34 | 55,9% |  | -0.188431 |
| range_pct | high60 | 34 | 52,9% | 1.54957 |  |
| slope_strength | low60 | 34 | 50% |  | 322.488 |
| clv | low60 | 34 | 50% |  | 0.264657 |
| body_pct | high60 | 34 | 50% | 32.4194 |  |
| lower_wick_pct | low60 | 34 | 50% |  | 21.6109 |
| ret_5_pct | high60 | 34 | 50% | -1.9203 |  |
| rsi | low60 | 34 | 50% |  | 37.3371 |
| dist_to_high_n_pct | high60 | 34 | 50% | 4.60846 |  |
| dist_to_low_n_pct | low60 | 34 | 50% |  | 1.47019 |
| slope_strength | high60 | 34 | 47,1% | 163.55 |  |
| context_pullback_pct | low60 | 34 | 47,1% |  | 1.56374 |
| ma_gap_pct | high60 | 34 | 47,1% | -3.50344 |  |
| atr_pct | low60 | 34 | 47,1% |  | 1.66724 |
| atr_pct | high60 | 34 | 47,1% | 1.35918 |  |
| upper_wick_pct | low60 | 34 | 47,1% |  | 34.7178 |
| pos_in_range_n | low60 | 34 | 47,1% |  | 0.222399 |
| vol_z | high60 | 34 | 47,1% | -0.258045 |  |
| bars_since_new_high | high60 | 34 | 47,1% | 21 |  |

## 4h | DL | short | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| body_pct | high60 | 34 | 47,1% | 32.4194 |  |
| ret_3_pct | low60 | 34 | 47,1% |  | -0.188431 |
| lower_wick_pct | low60 | 34 | 44,1% |  | 21.6109 |
| ret_1_pct | low60 | 34 | 44,1% |  | -0.148323 |
| clv | low60 | 34 | 41,2% |  | 0.264657 |
| range_pct | high60 | 34 | 41,2% | 1.54957 |  |
| upper_wick_pct | low60 | 34 | 41,2% |  | 34.7178 |
| ret_5_pct | high60 | 34 | 41,2% | -1.9203 |  |
| dist_to_high_n_pct | high60 | 34 | 41,2% | 4.60846 |  |
| dist_to_low_n_pct | low60 | 34 | 41,2% |  | 1.47019 |
| slope_strength | low60 | 34 | 38,2% |  | 322.488 |
| atr_pct | high60 | 34 | 38,2% | 1.35918 |  |
| rsi | high60 | 34 | 38,2% | 32.0007 |  |
| pos_in_range_n | low60 | 34 | 38,2% |  | 0.222399 |
| vol_z | high60 | 34 | 38,2% | -0.258045 |  |
| pullback_from_new_high_pct | high60 | 34 | 38,2% | 6.42766 |  |
| context_bars_since_extreme | high50 | 37 | 37,8% | 2 |  |
| context_bars_since_extreme | high60 | 37 | 37,8% | 2 |  |
| bars_since_new_low | high50 | 37 | 37,8% | 2 |  |
| bars_since_new_low | high60 | 37 | 37,8% | 2 |  |

## 4h | DL | short | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| body_pct | high60 | 34 | 41,2% | 32.4194 |  |
| ret_5_pct | high60 | 34 | 38,2% | -1.9203 |  |
| slope_strength | low60 | 34 | 35,3% |  | 322.488 |
| upper_wick_pct | low60 | 34 | 35,3% |  | 34.7178 |
| lower_wick_pct | low60 | 34 | 35,3% |  | 21.6109 |
| ret_1_pct | low60 | 34 | 35,3% |  | -0.148323 |
| ret_3_pct | low60 | 34 | 35,3% |  | -0.188431 |
| context_bars_since_extreme | high50 | 37 | 35,1% | 2 |  |
| context_bars_since_extreme | high60 | 37 | 35,1% | 2 |  |
| bars_since_new_low | high50 | 37 | 35,1% | 2 |  |
| bars_since_new_low | high60 | 37 | 35,1% | 2 |  |
| clv | low60 | 34 | 32,4% |  | 0.264657 |
| rsi | high60 | 34 | 32,4% | 32.0007 |  |
| pos_in_range_n | low60 | 34 | 32,4% |  | 0.222399 |
| dist_to_high_n_pct | low60 | 34 | 32,4% |  | 6.04498 |
| dist_to_high_n_pct | high60 | 34 | 32,4% | 4.60846 |  |
| dist_to_low_n_pct | low60 | 34 | 32,4% |  | 1.47019 |
| context_after_extreme_flag | val=1 | 43 | 30,2% | 1 | 1 |
| context_pullback_pct | high60 | 34 | 29,4% | 1.00222 |  |
| ma_gap_pct | high60 | 34 | 29,4% | -3.50344 |  |

## 4h | PFR | long | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| dist_to_high_n_pct | high60 | 32 | 62,5% | 0.865008 |  |
| bars_since_new_low | low60 | 32 | 62,5% |  | 38.8 |
| context_pullback_pct | high60 | 32 | 59,4% | 0.939478 |  |
| clv | low60 | 32 | 59,4% |  | 0.923187 |
| upper_wick_pct | high60 | 32 | 59,4% | 7.68135 |  |
| rsi | high60 | 32 | 59,4% | 65.8062 |  |
| pullback_from_new_high_pct | high60 | 32 | 59,4% | 0.939478 |  |
| context_bars_since_extreme | high50 | 36 | 58,3% | 3 |  |
| context_bars_since_extreme | high60 | 36 | 58,3% | 3 |  |
| bars_since_new_high | high50 | 36 | 58,3% | 3 |  |
| bars_since_new_high | high60 | 36 | 58,3% | 3 |  |
| slope_strength | low60 | 32 | 56,2% |  | 218.343 |
| atr_pct | high60 | 32 | 56,2% | 1.13189 |  |
| body_pct | high60 | 32 | 56,2% | 36.3932 |  |
| lower_wick_pct | high60 | 32 | 56,2% | 33.1527 |  |
| ret_5_pct | low60 | 32 | 56,2% |  | 1.07743 |
| context_after_extreme_flag | val=1 | 47 | 53,2% | 1 | 1 |
| slope_strength | high60 | 32 | 53,1% | 128.503 |  |
| ma_gap_pct | low60 | 32 | 53,1% |  | 4.81373 |
| dist_to_sma80_pct | low60 | 32 | 53,1% |  | 5.68153 |

## 4h | PFR | long | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| rsi | high60 | 32 | 56,2% | 65.5835 |  |
| dist_to_high_n_pct | high60 | 32 | 56,2% | 0.920247 |  |
| context_pullback_pct | high60 | 32 | 53,1% | 0.968118 |  |
| clv | low60 | 32 | 53,1% |  | 0.917005 |
| upper_wick_pct | high60 | 32 | 53,1% | 8.29947 |  |
| pullback_from_new_high_pct | high60 | 32 | 53,1% | 0.968118 |  |
| bars_since_new_low | low60 | 32 | 53,1% |  | 39.2 |
| slope_strength | low60 | 32 | 50% |  | 224.268 |
| range_pct | low60 | 32 | 50% |  | 1.32451 |
| lower_wick_pct | high60 | 32 | 50% | 33.0619 |  |
| ret_1_pct | low60 | 32 | 50% |  | 0.54029 |
| pullback_from_new_low_pct | low60 | 32 | 50% |  | 11.6404 |
| context_bars_since_extreme | high50 | 35 | 48,6% | 3 |  |
| context_bars_since_extreme | high60 | 35 | 48,6% | 3 |  |
| bars_since_new_high | high50 | 35 | 48,6% | 3 |  |
| bars_since_new_high | high60 | 35 | 48,6% | 3 |  |
| slope_strength | high60 | 32 | 46,9% | 129.673 |  |
| ma_gap_pct | low60 | 32 | 46,9% |  | 4.91423 |
| dist_to_sma80_pct | low60 | 32 | 46,9% |  | 5.76092 |
| atr_pct | high60 | 32 | 46,9% | 1.13254 |  |

## 4h | PFR | long | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|
| rsi | high60 | 32 | 50% | 65.5835 |  |
| clv | low60 | 32 | 46,9% |  | 0.917005 |
| upper_wick_pct | high60 | 32 | 46,9% | 8.29947 |  |
| slope_strength | high60 | 32 | 43,8% | 129.673 |  |
| ret_3_pct | low60 | 32 | 43,8% |  | 0.216519 |
| dist_to_high_n_pct | high60 | 32 | 43,8% | 0.920247 |  |
| bars_since_new_low | low60 | 32 | 43,8% |  | 39.2 |
| pullback_from_new_low_pct | low60 | 32 | 43,8% |  | 11.6404 |
| context_bars_since_extreme | high50 | 35 | 42,9% | 3 |  |
| context_bars_since_extreme | high60 | 35 | 42,9% | 3 |  |
| bars_since_new_high | high50 | 35 | 42,9% | 3 |  |
| bars_since_new_high | high60 | 35 | 42,9% | 3 |  |
| slope_strength | low60 | 32 | 40,6% |  | 224.268 |
| context_pullback_pct | high60 | 32 | 40,6% | 0.968118 |  |
| atr_pct | high60 | 32 | 40,6% | 1.13254 |  |
| range_pct | low60 | 32 | 40,6% |  | 1.32451 |
| range_pct | high60 | 32 | 40,6% | 1.02017 |  |
| lower_wick_pct | high60 | 32 | 40,6% | 33.0619 |  |
| ret_1_pct | low60 | 32 | 40,6% |  | 0.54029 |
| ret_5_pct | low60 | 32 | 40,6% |  | 1.07649 |

## 4h | PFR | short | RR 1.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|

## 4h | PFR | short | RR 1.5

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|

## 4h | PFR | short | RR 2.0

| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |
|---|---|---:|---:|---:|---:|

