# PFR 4H - Step 4 Summary

- SYMBOLS=['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
- WINDOW_DAYS_4H=365
- MAX_BARS_FETCH_4H=40000
- RRs=[1.0, 1.5, 2.0, 3.0]
- Rank RR=1.5
- Train fraction=0.7
- MIN_TRADES_FOR_RANK_TEST=80
- MAX_HOLD_BARS=50
- BE_TARGET_FRACTION=0.7

Outputs:
- results/pfr4h_trades_<SYMBOL>.csv
- results/pfr4h_stage1_<SYMBOL>.csv
- results/pfr4h_best_by_exitprofile_<SYMBOL>.csv
- results/pfr4h_stage1_MASTER.csv
- results/pfr4h_best_by_exitprofile_MASTER.csv
