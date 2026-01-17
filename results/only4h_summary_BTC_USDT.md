# Only 4H Summary - BTC_USDT

- Trades: somente 4h
- HTF: somente se 1D nativo baixar separado (available=True)
- Train fraction: 0.7
- Rank RR: 1.5
- Min test trades rank: 80
- Min train trades select: 80
- Time exit bars (4h): 50
- Break-even fraction of target: 0.7
- HTF require slope: True

## Stage 1 (Structural, sem stretch)

| Config       |   Train Trades |   Train WR |   Train AvgR |   Test Trades |   Test WR |   Test AvgR | Eligible   |
|:-------------|---------------:|-----------:|-------------:|--------------:|----------:|------------:|:-----------|
| htf0_tx0_be1 |            452 |   0.298673 |   -0.147124  |           185 |  0.345946 |  -0.0864865 | True       |
| htf0_tx1_be1 |            452 |   0.298673 |   -0.147124  |           185 |  0.345946 |  -0.0864865 | True       |
| htf0_tx0_be0 |            452 |   0.338496 |   -0.153761  |           185 |  0.356757 |  -0.108108  | True       |
| htf0_tx1_be0 |            452 |   0.338496 |   -0.150437  |           185 |  0.356757 |  -0.108108  | True       |
| htf1_tx0_be1 |            210 |   0.342857 |   -0.0333333 |            97 |  0.28866  |  -0.206186  | True       |
| htf1_tx1_be1 |            210 |   0.342857 |   -0.0333333 |            97 |  0.28866  |  -0.206186  | True       |
| htf1_tx0_be0 |            210 |   0.385714 |   -0.0357143 |            97 |  0.309278 |  -0.226804  | True       |
| htf1_tx1_be0 |            210 |   0.385714 |   -0.0357143 |            97 |  0.309278 |  -0.226804  | True       |

## Stage 2 (Best stretch per selected config)

| Config       |   BASE Train Trades |   BASE Train AvgR |   BASE Test Trades |   BASE Test AvgR |   BEST dist_sma8_max |   BEST dist_sma80_max |   BEST sma_gap_min |   BEST Train Trades |   BEST Train AvgR |   BEST Test Trades |   BEST Test AvgR |
|:-------------|--------------------:|------------------:|-------------------:|-----------------:|---------------------:|----------------------:|-------------------:|--------------------:|------------------:|-------------------:|-----------------:|
| htf0_tx0_be1 |                 452 |         -0.147124 |                185 |       -0.0864865 |                 0.01 |                  0.03 |                  0 |                  93 |        -0.0376344 |                 55 |         0.472727 |
| htf0_tx1_be1 |                 452 |         -0.147124 |                185 |       -0.0864865 |                 0.01 |                  0.03 |                  0 |                  93 |        -0.0376344 |                 55 |         0.472727 |