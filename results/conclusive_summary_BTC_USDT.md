# Conclusive Summary - BTC_USDT

- Train fraction: 0.7
- Rank RR: 1.5
- Min trades for rank: 80
- Time exit bars: 50
- BE target fraction: 0.7
- HTF require slope: True
- Stretch grid:
  - DIST_SMA8_MAX_LIST=[0.01, 0.02, 0.03]
  - DIST_SMA80_MAX_LIST=[0.03, 0.05, 0.07]
  - SMA_GAP_MIN_LIST=[0.0, 0.005, 0.01]

## Stage 1 (Structural, no stretch)

| Config       |   Train Trades |   Train WR |   Train AvgR |   Test Trades |   Test WR |   Test AvgR | Eligible   |
|:-------------|---------------:|-----------:|-------------:|--------------:|----------:|------------:|:-----------|
| htf1_tx0_be1 |            604 |   0.342715 |  -0.0289735  |           254 |  0.314961 |  -0.0787402 | True       |
| htf1_tx1_be1 |            604 |   0.342715 |  -0.00847848 |           254 |  0.314961 |  -0.0787402 | True       |
| htf1_tx1_be0 |            604 |   0.38245  |  -0.0188515  |           254 |  0.362205 |  -0.0917587 | True       |
| htf1_tx0_be0 |            604 |   0.38245  |  -0.0438742  |           254 |  0.362205 |  -0.0944882 | True       |
| htf0_tx1_be1 |           1131 |   0.329797 |  -0.046084   |           484 |  0.295455 |  -0.15061   | True       |
| htf0_tx0_be1 |           1131 |   0.329797 |  -0.0570292  |           484 |  0.295455 |  -0.15186   | True       |
| htf0_tx1_be0 |           1131 |   0.381079 |  -0.0326117  |           484 |  0.334711 |  -0.160541  | True       |
| htf0_tx0_be0 |           1131 |   0.381079 |  -0.0473033  |           484 |  0.334711 |  -0.163223  | True       |

## Stage 2 (Best stretch per selected config)

| Config       |   BASE Train Trades |   BASE Train AvgR |   BASE Test Trades |   BASE Test AvgR |   BEST dist_sma8_max |   BEST dist_sma80_max |   BEST sma_gap_min |   BEST Train Trades |   BEST Train AvgR |   BEST Test Trades |   BEST Test AvgR |
|:-------------|--------------------:|------------------:|-------------------:|-----------------:|---------------------:|----------------------:|-------------------:|--------------------:|------------------:|-------------------:|-----------------:|
| htf1_tx0_be1 |                 604 |       -0.0289735  |                254 |       -0.0787402 |                 0.02 |                  0.03 |              0.005 |                 200 |         0.075     |                153 |      -0.00326797 |
| htf1_tx1_be1 |                 604 |       -0.00847848 |                254 |       -0.0787402 |                 0.02 |                  0.03 |              0.005 |                 200 |         0.0918071 |                153 |      -0.00326797 |