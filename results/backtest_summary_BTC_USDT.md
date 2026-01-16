# Backtest PFR (MEXC Perps) — BTC_USDT

- Gerado em UTC: 2026-01-16 05:28:03
- Timeframes: `1h, 2h, 4h, 1d, 1w`
- Tendência: SMA10/SMA100 (close acima/abaixo das duas + cruzamento)
- Setup: PFR (buy/sell) com entrada 1 tick acima/abaixo do candle sinal
- MAX_ENTRY_WAIT_BARS: `10` | MAX_HOLD_BARS: `50` | AMBIGUOUS_POLICY: `loss`

| TF | RR | Trades | Wins | Losses | NoHit | Skipped | WinRate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1d | 1.0 | 27 | 11 | 16 | 0 | 0 | 40,7% |
| 1d | 1.5 | 27 | 10 | 17 | 0 | 0 | 37% |
| 1d | 2.0 | 27 | 9 | 18 | 0 | 0 | 33,3% |
| 1d | 3.0 | 27 | 8 | 19 | 0 | 0 | 29,6% |
| 1h | 1.0 | 746 | 375 | 366 | 5 | 0 | 50,6% |
| 1h | 1.5 | 746 | 305 | 428 | 13 | 0 | 41,6% |
| 1h | 2.0 | 746 | 256 | 471 | 19 | 0 | 35,2% |
| 1h | 3.0 | 746 | 187 | 521 | 38 | 0 | 26,4% |
| 2h | 1.0 | 361 | 182 | 178 | 1 | 0 | 50,6% |
| 2h | 1.5 | 361 | 154 | 201 | 6 | 0 | 43,4% |
| 2h | 2.0 | 361 | 123 | 225 | 13 | 0 | 35,3% |
| 2h | 3.0 | 361 | 91 | 248 | 22 | 0 | 26,8% |
| 4h | 1.0 | 208 | 106 | 100 | 2 | 0 | 51,5% |
| 4h | 1.5 | 208 | 84 | 119 | 5 | 0 | 41,4% |
| 4h | 2.0 | 208 | 74 | 129 | 5 | 0 | 36,5% |
| 4h | 3.0 | 208 | 52 | 139 | 17 | 0 | 27,2% |
