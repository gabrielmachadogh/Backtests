import os
import time
import ccxt
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime, timedelta

# ================= CONFIGURAÇÕES =================
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", 
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT"
]

DAYS_HISTORY = 200 

SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8
MAX_HOLD_BARS = 50

# ================= DEFINIÇÃO DOS SETUPS (DIAMANTES) =================
def check_setups(row, tf):
    s = {}
    
    # Filtros Base
    if not (row['trend_up'] and row['slope_up']):
        # Se não for tendência de alta, checa tendência de baixa para os shorts
        if not (row['trend_down'] and row['slope_down']):
            return s

    trend_up = row['trend_up'] and row['slope_up']
    trend_down = row['trend_down'] and row['slope_down']

    # 1. "O Rejeitador" (2h PFR Long)
    if tf == '2h' and trend_up and row['setup_pfr_buy']:
        if row['lower_wick_pct'] >= 45.0:
            s['1_Rejeitador_2h'] = {'side': 'long', 'rr': 1.0}

    # 2. "O Rompedor" (1h DL Long)
    if tf == '1h' and trend_up and row['setup_dl_buy']:
        if row['pos_in_range_n'] >= 0.90:
            s['2_Rompedor_1h'] = {'side': 'long', 'rr': 1.0}

    # 3. "O Foguete" (1h PFR Long)
    if tf == '1h' and trend_up and row['setup_pfr_buy']:
        if row['ret_5_pct'] >= 1.0:
            s['3_Foguete_1h'] = {'side': 'long', 'rr': 1.0}

    # 4. "DL Volátil" (2h DL Long)
    if tf == '2h' and trend_up and row['setup_dl_buy']:
        if row['vol_z'] >= 1.0:
            s['4_DL_Volatil_2h'] = {'side': 'long', 'rr': 1.5}

    # 5. "PFR Pullback Raso" (1h PFR Long)
    if tf == '1h' and trend_up and row['setup_pfr_buy']:
        if row['pullback_from_high'] <= 2.0:
            s['5_PFR_Pullback_1h'] = {'side': 'long', 'rr': 1.5}

    # 6. "PFR Sniper" (4h PFR Long)
    if tf == '4h' and trend_up and row['setup_pfr_buy']:
        if row['ma_gap_pct'] < 3.0: 
            s['6_PFR_Sniper_4h'] = {'side': 'long', 'rr': 1.0}

    # 7. "Queda Livre" (1h PFR Short)
    if tf == '1h' and trend_down and row['setup_pfr_sell']:
        if row['pullback_from_low'] <= 2.0:
            s['7_QuedaLivre_Short_1h'] = {'side': 'short', 'rr': 1.0}

    # 8. "Short Esticado" (1h DL Short)
    if tf == '1h' and trend_down and row['setup_dl_sell']:
        if row['pos_in_range_n'] <= 0.25:
            s['8_ShortEsticado_1h'] = {'side': 'short', 'rr': 1.0}

    # 9. "PFR Tendência" (1h PFR Long)
    if tf == '1h' and trend_up and row['setup_pfr_buy']:
        if row['bars_since_high'] <= 20:
            s['9_PFR_Trend_1h'] = {'side': 'long', 'rr': 1.5}

    # 10. "DL Correção" (4h DL Long)
    if tf == '4h' and trend_up and row['setup_dl_buy']:
        if row['upper_wick_pct'] < 20.0:
            s['10_DL_Clean_4h'] = {'side': 'long', 'rr': 1.0}

    return s

# ================= ENGINE =================

def fetch_data_fast(symbol):
    ex = ccxt.mexc({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    
    since = int((datetime.now() - timedelta(days=DAYS_HISTORY)).timestamp() * 1000)
    all_ohlcv = []
    
    print(f"   -> Baixando {symbol} (últimos {DAYS_HISTORY} dias)...")
    
    for _ in range(10):
        try:
            ohlcv = ex.fetch_ohlcv(symbol, '1h', limit=1000, since=since)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break 
            time.sleep(0.2)
        except:
            break
            
    if not all_ohlcv: return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)

def resample_data(df, rule):
    if rule == '1h': return df.copy()
    
    mapping = {'2h': '2h', '4h': '4h', '1d': '1d'} # Correção do warning (lowercase)
    if rule not in mapping: return df
    
    d = df.set_index('ts').resample(mapping[rule]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return d

def prepare_indicators(df):
    x = df.copy()
    
    # Médias & Trend
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    x['trend_down'] = (x['close'] < x['sma_s']) & (x['close'] < x['sma_l']) & (x['sma_s'] < x['sma_l'])
    
    # Slope
    prev_max = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).max()
    prev_min = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).min()
    x['slope_up'] = x['sma_s'] > prev_max
    x['slope_down'] = x['sma_s'] < prev_min # Correção lógica do slope down

    # Features
    x['range'] = x['high'] - x['low']
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    x['lower_wick_pct'] = np.where(x['range'] > 0, (x['lower_wick'] / x['range']) * 100, 0)
    
    x['upper_wick'] = x['high'] - np.maximum(x['open'], x['close'])
    x['upper_wick_pct'] = np.where(x['range'] > 0, (x['upper_wick'] / x['range']) * 100, 0)
    
    x['ret_5_pct'] = x['close'].pct_change(5) * 100
    x['ma_gap_pct'] = ((x['sma_s'] - x['sma_l']) / x['sma_l']) * 100
    
    # Vol Z
    vol_mean = x['volume'].rolling(20).mean()
    vol_std = x['volume'].rolling(20).std()
    x['vol_z'] = (x['volume'] - vol_mean) / vol_std
    
    # Pos in Range
    roll_hi = x['high'].rolling(20).max()
    roll_lo = x['low'].rolling(20).min()
    rng_div = roll_hi - roll_lo
    x['pos_in_range_n'] = np.where(rng_div > 0, (x['close'] - roll_lo) / rng_div, 0)
    
    # New High/Pullback
    x['new_high'] = x['high'] > x['high'].shift(1).rolling(20).max()
    x['grp_hi'] = x['new_high'].cumsum()
    x['bars_since_high'] = x.groupby('grp_hi').cumcount()
    x['last_high_price'] = x['high'].where(x['new_high']).ffill()
    x['pullback_from_high'] = (x['last_high_price'] - x['close']) / x['last_high_price'] * 100
    
    # New Low/Pullback (p/ shorts)
    x['new_low'] = x['low'] < x['low'].shift(1).rolling(20).min()
    x['grp_lo'] = x['new_low'].cumsum()
    x['last_low_price'] = x['low'].where(x['new_low']).ffill()
    x['pullback_from_low'] = (x['close'] - x['last_low_price']) / x['last_low_price'] * 100

    # Signals
    p_low = x['low'].shift(1)
    p2_low = x['low'].shift(2)
    p_high = x['high'].shift(1)
    p2_high = x['high'].shift(2)
    p_close = x['close'].shift(1)
    
    x['setup_pfr_buy'] = (x['low'] < p_low) & (x['low'] < p2_low) & (x['close'] > p_close)
    x['setup_pfr_sell'] = (x['high'] > p_high) & (x['high'] > p2_high) & (x['close'] < p_close)
    x['setup_dl_buy'] = (x['low'] < p_low) & (x['low'] < p2_low)
    x['setup_dl_sell'] = (x['high'] > p_high) & (x['high'] > p2_high)
    
    return x

def run_simulation(df, i, side, rr):
    tick = df['close'].iloc[i] * 0.0001
    
    if side == 'long':
        entry_trigger = df.iloc[i]['high'] + tick
        stop_price = df.iloc[i]['low'] - tick
    else:
        entry_trigger = df.iloc[i]['low'] - tick
        stop_price = df.iloc[i]['high'] + tick
    
    risk = abs(entry_trigger - stop_price)
    if risk == 0: return None
    
    target = entry_trigger + (risk * rr) if side == 'long' else entry_trigger - (risk * rr)
    
    if i + 1 >= len(df): return None
    next_c = df.iloc[i+1]
    
    filled = False
    if side == 'long':
        if next_c['high'] >= entry_trigger: filled = True
    else:
        if next_c['low'] <= entry_trigger: filled = True
        
    if not filled: return "no_fill"
    
    for j in range(i+1, min(i+1+MAX_HOLD_BARS, len(df))):
        c = df.iloc[j]
        if side == 'long':
            if c['low'] <= stop_price: return "loss"
            if c['high'] >= target: return "win"
        else:
            if c['high'] >= stop_price: return "loss"
            if c['low'] <= target: return "win"
            
    return "loss"

def main():
    results = []
    print(f"--- Validação Rápida (200 dias) em {len(SYMBOLS)} moedas ---")
    
    for symbol in SYMBOLS:
        clean_sym = symbol.split(":")[0]
        
        df_1h = fetch_data_fast(clean_sym)
        if df_1h.empty or len(df_1h) < 200:
            continue
            
        dfs = {
            '1h': prepare_indicators(df_1h),
            '2h': prepare_indicators(resample_data(df_1h, '2h')),
            '4h': prepare_indicators(resample_data(df_1h, '4h'))
        }
        
        for tf, df in dfs.items():
            if len(df) < 100: continue
            
            # Convertendo para dicionário para acesso mais rápido na iteração
            records = df.to_dict('records')
            
            for i in range(100, len(records)-1):
                row = records[i]
                active_setups = check_setups(row, tf)
                
                for name, params in active_setups.items():
                    # Passamos o dataframe original para a simulação (precisa olhar o futuro)
                    res = run_simulation(df, i, params['side'], params['rr'])
                    if res in ['win', 'loss']:
                        results.append({
                            'Symbol': clean_sym,
                            'Setup': name,
                            'Outcome': res
                        })

    if not results:
        print("Nenhum trade encontrado.")
        return

    df_res = pd.DataFrame(results)
    
    def agg_fmt(x):
        total = len(x)
        wins = (x == 'win').sum()
        wr = 100 * wins / total if total > 0 else 0
        return f"{wr:.0f}% ({total})"

    pivot = df_res.pivot_table(index='Setup', columns='Symbol', values='Outcome', aggfunc=agg_fmt, fill_value="-")
    
    total_stats = df_res.groupby('Setup')['Outcome'].apply(lambda x: f"{100 * (x == 'win').sum() / len(x):.1f}% ({len(x)})")
    pivot['ALL_COINS'] = total_stats

    pivot = pivot.sort_index()

    print("\n=== RESULTADO ===")
    print(tabulate(pivot, headers='keys', tablefmt='grid'))
    pivot.to_csv("results/validation_turbo.csv")

if __name__ == "__main__":
    main()
