# Best patterns — BTC_USDT

- MIN_TRADES: 30
- Buckets: low/high [10, 15, 20, 25, 30, 40, 50, 60]

## 1h | DL | RR 1.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| pos_in_range_n | high10 | 54 | 66,7% |
| pos_in_range_n | high15 | 81 | 64,2% |
| dist_to_high_n_pct | low15 | 81 | 64,2% |
| ret_5_pct | high10 | 54 | 63% |
| clv | low10 | 54 | 61,1% |
| dist_to_high_n_pct | low20 | 108 | 60,2% |
| ret_5_pct | high15 | 81 | 59,3% |
| upper_wick_pct | low10 | 54 | 59,3% |
| dist_to_high_n_pct | low10 | 54 | 59,3% |
| ret_5_pct | high20 | 108 | 58,3% |
| pos_in_range_n | high20 | 108 | 57,4% |
| clv | high10 | 54 | 57,4% |
| ret_5_pct | high25 | 135 | 57% |
| lower_wick_pct | low15 | 81 | 56,8% |
| slope_strength | low20 | 108 | 56,5% |
| dist_to_high_n_pct | low25 | 135 | 56,3% |
| ret_5_pct | high30 | 162 | 56,2% |
| slope_strength | low30 | 162 | 55,6% |
| atr_pct | low30 | 162 | 55,6% |
| slope_strength | low25 | 135 | 55,6% |

## 1h | DL | RR 1.5

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| pos_in_range_n | high10 | 54 | 59,3% |
| pos_in_range_n | high15 | 80 | 53,8% |
| dist_to_high_n_pct | low15 | 80 | 53,8% |
| dist_to_high_n_pct | low10 | 54 | 51,9% |
| dist_to_high_n_pct | low20 | 107 | 49,5% |
| ret_5_pct | high10 | 54 | 48,1% |
| pos_in_range_n | high20 | 107 | 47,7% |
| ret_3_pct | high20 | 107 | 46,7% |
| upper_wick_pct | low10 | 54 | 46,3% |
| ret_3_pct | high10 | 54 | 46,3% |
| ret_3_pct | high15 | 80 | 46,2% |
| slope_strength | low20 | 107 | 45,8% |
| dist_to_high_n_pct | low25 | 133 | 45,1% |
| vol_z | high25 | 133 | 45,1% |
| ret_3_pct | high40 | 213 | 45,1% |
| ma_gap_pct | high30 | 160 | 45% |
| atr_pct | low30 | 160 | 45% |
| slope_strength | low15 | 80 | 45% |
| ma_gap_pct | high15 | 80 | 45% |
| ret_5_pct | high15 | 80 | 45% |

## 1h | DL | RR 2.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| pos_in_range_n | high10 | 53 | 49,1% |
| pos_in_range_n | high15 | 80 | 45% |
| dist_to_high_n_pct | low15 | 80 | 45% |
| upper_wick_pct | low10 | 53 | 43,4% |
| ret_5_pct | high10 | 53 | 43,4% |
| dist_to_high_n_pct | low10 | 53 | 43,4% |
| dist_to_high_n_pct | low20 | 106 | 42,5% |
| vol_z | high25 | 133 | 41,4% |
| atr_pct | low30 | 159 | 40,9% |
| pos_in_range_n | high20 | 106 | 40,6% |
| vol_z | high20 | 106 | 40,6% |
| atr_pct | low25 | 133 | 39,8% |
| slope_strength | low20 | 106 | 39,6% |
| vol_z | high10 | 53 | 39,6% |
| vol_z | high40 | 212 | 39,2% |
| dist_to_high_n_pct | low25 | 133 | 39,1% |
| vol_z | high30 | 159 | 39% |
| slope_strength | low15 | 80 | 38,8% |
| ret_3_pct | high20 | 106 | 38,7% |
| slope_strength | low30 | 159 | 38,4% |

## 1h | PFR | RR 1.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| ret_5_pct | high10 | 37 | 67,6% |
| dist_to_high_n_pct | low20 | 74 | 64,9% |
| ret_3_pct | high10 | 37 | 64,9% |
| pos_in_range_n | high10 | 37 | 64,9% |
| pos_in_range_n | high15 | 56 | 64,3% |
| pos_in_range_n | high20 | 74 | 63,5% |
| lower_wick_pct | low15 | 56 | 62,5% |
| lower_wick_pct | low20 | 74 | 62,2% |
| body_pct | high10 | 37 | 62,2% |
| upper_wick_pct | low10 | 37 | 62,2% |
| clv | low15 | 56 | 60,7% |
| body_pct | high15 | 56 | 60,7% |
| ret_1_pct | high15 | 56 | 60,7% |
| ret_3_pct | high15 | 56 | 60,7% |
| dist_to_high_n_pct | low15 | 56 | 60,7% |
| upper_wick_pct | low20 | 74 | 59,5% |
| ret_1_pct | high20 | 74 | 59,5% |
| slope_strength | low10 | 37 | 59,5% |
| ret_1_pct | high10 | 37 | 59,5% |
| lower_wick_pct | low25 | 93 | 59,1% |

## 1h | PFR | RR 1.5

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| pos_in_range_n | high10 | 37 | 59,5% |
| dist_to_high_n_pct | low20 | 73 | 54,8% |
| pos_in_range_n | high15 | 55 | 54,5% |
| dist_to_high_n_pct | low15 | 55 | 54,5% |
| slope_strength | low10 | 37 | 54,1% |
| upper_wick_pct | low10 | 37 | 54,1% |
| ret_3_pct | high10 | 37 | 54,1% |
| ret_3_pct | high15 | 55 | 52,7% |
| dist_to_high_n_pct | low10 | 37 | 51,4% |
| slope_strength | low25 | 92 | 51,1% |
| body_pct | high15 | 55 | 50,9% |
| pos_in_range_n | high20 | 73 | 50,7% |
| ret_3_pct | high20 | 73 | 49,3% |
| slope_strength | low30 | 110 | 49,1% |
| ret_1_pct | high15 | 55 | 49,1% |
| rsi | high25 | 92 | 48,9% |
| ret_1_pct | high10 | 37 | 48,6% |
| ret_5_pct | high10 | 37 | 48,6% |
| ret_3_pct | high40 | 146 | 47,9% |
| dist_to_high_n_pct | low25 | 92 | 47,8% |

## 1h | PFR | RR 2.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| upper_wick_pct | low10 | 37 | 51,4% |
| pos_in_range_n | high10 | 37 | 51,4% |
| dist_to_high_n_pct | low10 | 37 | 48,6% |
| dist_to_high_n_pct | low20 | 73 | 47,9% |
| pos_in_range_n | high15 | 55 | 47,3% |
| dist_to_high_n_pct | low15 | 55 | 47,3% |
| slope_strength | low10 | 37 | 45,9% |
| ret_3_pct | high10 | 37 | 45,9% |
| pos_in_range_n | high20 | 73 | 45,2% |
| slope_strength | low25 | 91 | 45,1% |
| ret_3_pct | high20 | 73 | 43,8% |
| ret_3_pct | high15 | 55 | 43,6% |
| clv | high10 | 37 | 43,2% |
| slope_strength | low30 | 109 | 43,1% |
| dist_to_high_n_pct | low25 | 91 | 42,9% |
| ret_3_pct | high40 | 145 | 42,1% |
| slope_strength | low15 | 55 | 41,8% |
| body_pct | high15 | 55 | 41,8% |
| upper_wick_pct | low15 | 55 | 41,8% |
| ret_1_pct | high15 | 55 | 41,8% |

## 2h | DL | RR 1.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| ret_5_pct | high15 | 40 | 75% |
| ret_5_pct | high20 | 53 | 69,8% |
| ret_5_pct | high30 | 79 | 65,8% |
| ret_5_pct | high25 | 66 | 65,2% |
| vol_z | high15 | 40 | 65% |
| lower_wick_pct | high20 | 53 | 64,2% |
| vol_z | high20 | 53 | 64,2% |
| clv | high20 | 53 | 62,3% |
| range_pct | high20 | 53 | 62,3% |
| clv | high25 | 66 | 62,1% |
| clv | high30 | 79 | 62% |
| ret_5_pct | high40 | 105 | 61% |
| lower_wick_pct | high30 | 79 | 60,8% |
| vol_z | high30 | 79 | 60,8% |
| range_pct | high25 | 66 | 60,6% |
| lower_wick_pct | high25 | 66 | 60,6% |
| vol_z | high25 | 66 | 60,6% |
| atr_pct | high20 | 53 | 60,4% |
| clv | high15 | 40 | 60% |
| lower_wick_pct | high15 | 40 | 60% |

## 2h | DL | RR 1.5

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| lower_wick_pct | high15 | 39 | 61,5% |
| lower_wick_pct | high20 | 52 | 59,6% |
| clv | high30 | 78 | 59% |
| ret_5_pct | high20 | 52 | 57,7% |
| lower_wick_pct | high25 | 65 | 56,9% |
| lower_wick_pct | high30 | 78 | 56,4% |
| ret_5_pct | high15 | 39 | 56,4% |
| clv | high20 | 52 | 55,8% |
| clv | high25 | 65 | 55,4% |
| ret_5_pct | high25 | 65 | 55,4% |
| clv | high40 | 103 | 54,4% |
| ret_5_pct | high30 | 78 | 53,8% |
| atr_pct | high20 | 52 | 53,8% |
| ret_5_pct | high40 | 103 | 51,5% |
| clv | high15 | 39 | 51,3% |
| body_pct | low15 | 39 | 51,3% |
| body_pct | low25 | 65 | 50,8% |
| lower_wick_pct | high40 | 103 | 50,5% |
| ret_5_pct | high50 | 129 | 50,4% |
| body_pct | low20 | 52 | 50% |

## 2h | DL | RR 2.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| lower_wick_pct | high15 | 39 | 59% |
| lower_wick_pct | high20 | 52 | 57,7% |
| lower_wick_pct | high25 | 64 | 54,7% |
| lower_wick_pct | high30 | 77 | 53,2% |
| ret_5_pct | high25 | 64 | 50% |
| clv | high20 | 52 | 50% |
| ret_5_pct | high20 | 52 | 50% |
| clv | high30 | 77 | 49,4% |
| ret_5_pct | high15 | 39 | 48,7% |
| ret_5_pct | high30 | 77 | 48,1% |
| clv | high25 | 64 | 46,9% |
| body_pct | low25 | 64 | 46,9% |
| clv | high40 | 103 | 46,6% |
| ret_5_pct | high40 | 103 | 46,6% |
| body_pct | low20 | 52 | 46,2% |
| rsi | high20 | 52 | 46,2% |
| clv | high15 | 39 | 46,2% |
| body_pct | low15 | 39 | 46,2% |
| rsi | high15 | 39 | 46,2% |
| dist_to_high_n_pct | low15 | 39 | 46,2% |

## 2h | PFR | RR 1.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| ret_5_pct | high20 | 36 | 77,8% |
| lower_wick_pct | high20 | 36 | 75% |
| lower_wick_pct | high25 | 45 | 73,3% |
| ret_5_pct | high25 | 45 | 73,3% |
| ret_5_pct | high30 | 54 | 72,2% |
| ret_5_pct | high40 | 71 | 67,6% |
| dist_to_sma80_pct | high30 | 54 | 66,7% |
| dist_to_sma80_pct | high25 | 45 | 66,7% |
| range_pct | high25 | 45 | 66,7% |
| ma_gap_pct | high20 | 36 | 66,7% |
| dist_to_sma80_pct | high20 | 36 | 66,7% |
| rsi | high20 | 36 | 66,7% |
| ma_gap_pct | high30 | 54 | 64,8% |
| lower_wick_pct | high30 | 54 | 64,8% |
| ma_gap_pct | high25 | 45 | 64,4% |
| rsi | high25 | 45 | 64,4% |
| atr_pct | high20 | 36 | 63,9% |
| clv | high20 | 36 | 63,9% |
| range_pct | high20 | 36 | 63,9% |
| vol_z | high20 | 36 | 63,9% |

## 2h | PFR | RR 1.5

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| lower_wick_pct | high20 | 35 | 68,6% |
| lower_wick_pct | high25 | 44 | 65,9% |
| ret_5_pct | high25 | 44 | 65,9% |
| ret_5_pct | high20 | 35 | 65,7% |
| ret_5_pct | high30 | 52 | 63,5% |
| lower_wick_pct | high30 | 52 | 61,5% |
| dist_to_sma80_pct | high25 | 44 | 61,4% |
| ret_5_pct | high40 | 70 | 60% |
| ma_gap_pct | high20 | 35 | 60% |
| dist_to_sma80_pct | high20 | 35 | 60% |
| ma_gap_pct | high30 | 52 | 59,6% |
| dist_to_sma80_pct | high30 | 52 | 59,6% |
| clv | high40 | 70 | 58,6% |
| pos_in_range_n | high40 | 70 | 58,6% |
| ma_gap_pct | high40 | 70 | 57,1% |
| rsi | high20 | 35 | 57,1% |
| ma_gap_pct | high25 | 44 | 56,8% |
| rsi | high25 | 44 | 56,8% |
| clv | high30 | 52 | 55,8% |
| rsi | high30 | 52 | 55,8% |

## 2h | PFR | RR 2.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| lower_wick_pct | high20 | 35 | 65,7% |
| lower_wick_pct | high25 | 43 | 60,5% |
| ret_5_pct | high25 | 43 | 58,1% |
| ret_5_pct | high20 | 35 | 57,1% |
| rsi | high25 | 43 | 55,8% |
| rsi | high20 | 35 | 54,3% |
| lower_wick_pct | high30 | 52 | 53,8% |
| ret_5_pct | high30 | 52 | 53,8% |
| dist_to_sma80_pct | high25 | 43 | 53,5% |
| ma_gap_pct | high20 | 35 | 51,4% |
| dist_to_sma80_pct | high20 | 35 | 51,4% |
| ret_5_pct | high40 | 69 | 50,7% |
| ma_gap_pct | high30 | 52 | 50% |
| dist_to_sma80_pct | high30 | 52 | 50% |
| clv | high30 | 52 | 50% |
| pos_in_range_n | high40 | 69 | 49,3% |
| ma_gap_pct | high25 | 43 | 48,8% |
| clv | high25 | 43 | 48,8% |
| body_pct | low25 | 43 | 48,8% |
| clv | high20 | 35 | 48,6% |

## 4h | DL | RR 1.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| dist_to_low_n_pct | low30 | 42 | 54,8% |
| dist_to_sma80_pct | low50 | 70 | 54,3% |
| ret_1_pct | high25 | 35 | 54,3% |
| dist_to_low_n_pct | low25 | 35 | 54,3% |
| dist_to_sma80_pct | low60 | 84 | 53,6% |
| body_pct | high40 | 56 | 51,8% |
| ma_gap_pct | low50 | 70 | 51,4% |
| atr_pct | low25 | 35 | 51,4% |
| pos_in_range_n | high25 | 35 | 51,4% |
| ma_gap_pct | low60 | 84 | 51,2% |
| upper_wick_pct | high60 | 84 | 50% |
| pos_in_range_n | low40 | 56 | 50% |
| pos_in_range_n | high40 | 56 | 50% |
| atr_pct | low30 | 42 | 50% |
| upper_wick_pct | high30 | 42 | 50% |
| ret_1_pct | high30 | 42 | 50% |
| dist_to_low_n_pct | high30 | 42 | 50% |
| slope_strength | low60 | 84 | 48,8% |
| lower_wick_pct | low60 | 84 | 48,8% |
| ret_3_pct | low60 | 84 | 48,8% |

## 4h | DL | RR 1.5

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| dist_to_sma80_pct | low60 | 83 | 47% |
| dist_to_sma80_pct | low50 | 70 | 45,7% |
| range_pct | high25 | 35 | 45,7% |
| ret_1_pct | high25 | 35 | 45,7% |
| rsi | high25 | 35 | 45,7% |
| dist_to_low_n_pct | low25 | 35 | 45,7% |
| rsi | high30 | 42 | 45,2% |
| dist_to_low_n_pct | low30 | 42 | 45,2% |
| body_pct | high40 | 56 | 44,6% |
| ma_gap_pct | low60 | 83 | 43,4% |
| ma_gap_pct | low50 | 70 | 42,9% |
| range_pct | high30 | 42 | 42,9% |
| body_pct | high30 | 42 | 42,9% |
| ret_1_pct | high30 | 42 | 42,9% |
| atr_pct | low25 | 35 | 42,9% |
| atr_pct | high25 | 35 | 42,9% |
| ret_1_pct | low25 | 35 | 42,9% |
| pos_in_range_n | high25 | 35 | 42,9% |
| ret_3_pct | low60 | 83 | 42,2% |
| atr_pct | high50 | 70 | 41,4% |

## 4h | DL | RR 2.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| rsi | high25 | 35 | 42,9% |
| body_pct | high40 | 56 | 41,1% |
| rsi | high30 | 42 | 40,5% |
| ret_1_pct | high25 | 35 | 40% |
| dist_to_sma80_pct | low60 | 83 | 39,8% |
| dist_to_sma80_pct | low50 | 70 | 38,6% |
| body_pct | high30 | 42 | 38,1% |
| vol_z | low30 | 42 | 38,1% |
| body_pct | high25 | 35 | 37,1% |
| ret_1_pct | low25 | 35 | 37,1% |
| pos_in_range_n | high25 | 35 | 37,1% |
| vol_z | low25 | 35 | 37,1% |
| ma_gap_pct | low60 | 83 | 36,1% |
| ret_3_pct | low60 | 83 | 36,1% |
| ma_gap_pct | low50 | 70 | 35,7% |
| range_pct | low50 | 70 | 35,7% |
| body_pct | high50 | 70 | 35,7% |
| ret_3_pct | low50 | 70 | 35,7% |
| rsi | high50 | 70 | 35,7% |
| rsi | high40 | 56 | 35,7% |

## 4h | PFR | RR 1.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| rsi | high40 | 37 | 62,2% |
| upper_wick_pct | high50 | 47 | 61,7% |
| slope_strength | low60 | 56 | 60,7% |
| ma_gap_pct | low60 | 56 | 60,7% |
| dist_to_sma80_pct | low60 | 56 | 60,7% |
| dist_to_high_n_pct | high60 | 56 | 60,7% |
| ma_gap_pct | low50 | 47 | 59,6% |
| dist_to_sma80_pct | low50 | 47 | 59,6% |
| atr_pct | high50 | 47 | 59,6% |
| dist_to_high_n_pct | high50 | 47 | 59,6% |
| upper_wick_pct | high40 | 37 | 59,5% |
| rsi | low40 | 37 | 59,5% |
| body_pct | low60 | 56 | 58,9% |
| slope_strength | low50 | 47 | 57,4% |
| clv | low50 | 47 | 57,4% |
| lower_wick_pct | high50 | 47 | 57,4% |
| ret_1_pct | low50 | 47 | 57,4% |
| ret_5_pct | low50 | 47 | 57,4% |
| slope_strength | high60 | 56 | 57,1% |
| clv | low60 | 56 | 57,1% |

## 4h | PFR | RR 1.5

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| rsi | high40 | 37 | 54,1% |
| atr_pct | high50 | 46 | 52,2% |
| slope_strength | low60 | 55 | 50,9% |
| ret_5_pct | high60 | 55 | 50,9% |
| ma_gap_pct | low60 | 55 | 49,1% |
| dist_to_sma80_pct | low60 | 55 | 49,1% |
| ret_3_pct | low60 | 55 | 49,1% |
| atr_pct | high40 | 37 | 48,6% |
| slope_strength | low50 | 46 | 47,8% |
| upper_wick_pct | high50 | 46 | 47,8% |
| ret_3_pct | low50 | 46 | 47,8% |
| ret_5_pct | high50 | 46 | 47,8% |
| rsi | high50 | 46 | 47,8% |
| dist_to_high_n_pct | high60 | 55 | 47,3% |
| range_pct | high40 | 37 | 45,9% |
| upper_wick_pct | high40 | 37 | 45,9% |
| lower_wick_pct | high40 | 37 | 45,9% |
| ret_5_pct | high40 | 37 | 45,9% |
| pos_in_range_n | high40 | 37 | 45,9% |
| dist_to_low_n_pct | low40 | 37 | 45,9% |

## 4h | PFR | RR 2.0

| Feature | Bucket | Trades | WinRate |
|---|---|---:|---:|
| rsi | high40 | 37 | 45,9% |
| ret_5_pct | high60 | 55 | 43,6% |
| atr_pct | high50 | 46 | 43,5% |
| slope_strength | low60 | 55 | 41,8% |
| ret_3_pct | low60 | 55 | 41,8% |
| body_pct | high50 | 46 | 41,3% |
| ret_3_pct | low50 | 46 | 41,3% |
| rsi | high50 | 46 | 41,3% |
| atr_pct | high40 | 37 | 40,5% |
| dist_to_low_n_pct | high40 | 37 | 40,5% |
| ma_gap_pct | low60 | 55 | 40% |
| dist_to_sma80_pct | low60 | 55 | 40% |
| dist_to_high_n_pct | high60 | 55 | 40% |
| slope_strength | low50 | 46 | 39,1% |
| clv | high50 | 46 | 39,1% |
| upper_wick_pct | high50 | 46 | 39,1% |
| ret_5_pct | high50 | 46 | 39,1% |
| pos_in_range_n | high50 | 46 | 39,1% |
| dist_to_low_n_pct | high50 | 46 | 39,1% |
| slope_strength | high60 | 55 | 38,2% |

