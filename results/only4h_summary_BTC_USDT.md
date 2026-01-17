# Only 4H Summary - BTC_USDT

- Trades: somente 4h
- SETUPS=['MA_LADDER', 'PFR', 'DL', '8.2', '8.3']
- RANK_SETUP=ALL | RANK_RR=1.5
- Train fraction: 0.7
- Time exit bars (4h): 50
- Break-even fraction of target: 0.7
Note: configs have _c10{0|1} for close>sma10 gating on classic setups.

## Stage 1 (Structural)

| Config       |   Train Trades |   Train WR |   Train AvgR |   Test Trades |   Test WR |   Test AvgR | Eligible   |
|:-------------|---------------:|-----------:|-------------:|--------------:|----------:|------------:|:-----------|
| tx1_be1_c100 |            754 |   0.374005 |    0.0613802 |           331 |  0.347432 |  -0.0371277 | True       |
| tx0_be1_c100 |            754 |   0.374005 |    0.0517241 |           331 |  0.347432 |  -0.0498489 | True       |
| tx1_be0_c100 |            754 |   0.420424 |    0.0665232 |           331 |  0.371601 |  -0.0513069 | True       |
| tx0_be0_c100 |            754 |   0.420424 |    0.051061  |           331 |  0.371601 |  -0.070997  | True       |
| tx1_be1_c101 |            613 |   0.368679 |    0.0559228 |           253 |  0.316206 |  -0.0979813 | True       |
| tx1_be0_c101 |            613 |   0.414356 |    0.0549079 |           253 |  0.343874 |  -0.114556  | True       |
| tx0_be1_c101 |            613 |   0.368679 |    0.0440457 |           253 |  0.316206 |  -0.114625  | True       |
| tx0_be0_c101 |            613 |   0.414356 |    0.0358891 |           253 |  0.343874 |  -0.140316  | True       |