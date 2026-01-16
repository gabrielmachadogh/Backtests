import pandas as pd
import numpy as np
import os
from tabulate import tabulate

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
FILE = f"results/backtest_trades_{SYMBOL}.csv"

def analyze():
    if not os.path.exists(FILE):
        print("Arquivo de trades não encontrado.")
        return

    df = pd.read_csv(FILE)
    
    # Features para testar cortes finos
    features = [
        'slope_strength', 'bars_since_new_high', 'pullback_from_new_high_pct',
        'dist_to_sma80_pct', 'atr_pct', 'clv', 'lower_wick_pct', 
        'pos_in_range_n', 'ret_5_pct', 'vol_z'
    ]
    
    results = []
    
    # RRs detectados dinamicamente
    rr_cols = [c for c in df.columns if c.startswith('rr_')]
    
    for rr_col in rr_cols:
        rr_val = rr_col.split('_')[1]
        
        # Apenas trades resolvidos
        valid = df[df[rr_col].isin(['win', 'loss'])].copy()
        
        for (tf, setup), g in valid.groupby(['timeframe', 'setup']):
            if len(g) < 30: continue
            
            # Base WR
            wr = (g[rr_col] == 'win').mean()
            results.append([tf, setup, rr_val, 'ALL', 'ALL', len(g), wr, np.nan])
            
            # Testar Decis (10%, 20%... 90%)
            for feat in features:
                try:
                    # Testar corte "High" (acima de X)
                    for q in [0.7, 0.8, 0.9]:
                        thresh = g[feat].quantile(q)
                        sub = g[g[feat] >= thresh]
                        if len(sub) >= 15:
                            wr_sub = (sub[rr_col] == 'win').mean()
                            if wr_sub > wr * 1.1: # Só salva se melhorar 10%
                                results.append([tf, setup, rr_val, feat, f"High {int(q*100)}%", len(sub), wr_sub, thresh])
                                
                    # Testar corte "Low" (abaixo de X)
                    for q in [0.1, 0.2, 0.3]:
                        thresh = g[feat].quantile(q)
                        sub = g[g[feat] <= thresh]
                        if len(sub) >= 15:
                            wr_sub = (sub[rr_col] == 'win').mean()
                            if wr_sub > wr * 1.1:
                                results.append([tf, setup, rr_val, feat, f"Low {int(q*100)}%", len(sub), wr_sub, thresh])
                except: pass

    res_df = pd.DataFrame(results, columns=['TF', 'Setup', 'RR', 'Feature', 'Bucket', 'Trades', 'WinRate', 'Threshold'])
    res_df = res_df.sort_values('WinRate', ascending=False)
    
    print(tabulate(res_df.head(50), headers='keys', tablefmt='pipe'))
    res_df.to_csv(f"results/detailed_analysis_{SYMBOL}.csv", index=False)

if __name__ == "__main__":
    analyze()
