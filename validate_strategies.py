import os
import time
import ccxt
import pandas as pd
import numpy as np
from tabulate import tabulate

# ================= CONFIGURAÇÕES =================
# Lista de ativos para validar (Top Liquid)
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", 
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT"
]

# Configurações Técnicas
SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8
MAX_ENTRY_WAIT_BARS = 1 # Regra absoluta
MAX_HOLD_BARS = 50
RR_DEFAULT = 1.0 # Vamos validar a taxa de acerto base (1:1) primeiro
RR_ALT = 1.5     # Para os setups que pedem 1.5

# Histórico (ajuste conforme a paciência/limite da API)
LIMIT_CANDLES = 5000 # ~200 dias de 1h

# ================= DEFINIÇÃO DOS 10 SETUPS =================
# Aqui traduzimos a análise para regras universais (percentuais)

def check_setups(row, tf):
    s = {}
    
    # Base Filters
    trend_up = row['trend_up'] and row['slope_up']
    trend_down = row['trend_down'] and row['slope_down']
    
    # 1. "O Rejeitador" (2h PFR Long) - WR Esperada: ~72%
    if tf == '2h' and trend_up and row['setup_pfr_buy']:
        if row['lower_wick_pct'] >= 45.0:
            s['1_Rejeitador_2h'] = {'side': 'long', 'rr': 1.0}

    # 2. "O Rompedor" (1h DL Long) - WR Esperada: ~70%
    if tf == '1h' and trend_up and row['setup_dl_buy']:
        if row['pos_in_range_n'] >= 0.90:
            s['2_Rompedor_1h'] = {'side': 'long', 'rr': 1.0}

    # 3. "O Foguete" (1h PFR Long) - WR Esperada: ~70%
    if tf == '1h' and trend_up and row['setup_pfr_buy']:
        if row['ret_5_pct'] >= 1.0:
            s['3_Foguete_1h'] = {'side': 'long', 'rr': 1.0}

    # 4. "DL Volátil" (2h DL Long) - WR Esperada: ~65%
    if tf == '2h' and trend_up and row['setup_dl_buy']:
        if row['vol_z'] >= 1.0:
            s['4_DL_Volatil_2h'] = {'side': 'long', 'rr': 1.5}

    # 5. "PFR Pullback Raso" (1h PFR Long) - WR Esperada: ~62%
    if tf == '1h' and trend_up and row['setup_pfr_buy']:
        if row['pullback_from_high'] <= 2.0:
            s['5_PFR_Pullback_1h'] = {'side': 'long', 'rr': 1.5}

    # 6. "PFR Sniper" (4h PFR Long) - WR Esperada: ~61%
    # Adaptação: Slope Strength absoluto não funciona entre moedas. 
    # Usamos MA GAP < 3% como proxy de "não esticado".
    if tf == '4h' and trend_up and row['setup_pfr_buy']:
        if row['ma_gap_pct'] < 3.0: 
            s['6_PFR_Sniper_4h'] = {'side': 'long', 'rr': 1.0}

    # 7. "Queda Livre" (1h PFR Short) - WR Esperada: ~65%
    if tf == '1h' and trend_down and row['setup_pfr_sell']:
        if row['pullback_from_low'] <= 2.0:
            s['7_QuedaLivre_Short_1h'] = {'side': 'short', 'rr': 1.0}

    # 8. "Short Esticado" (1h DL Short) - WR Esperada: ~60%
    if tf == '1h' and trend_down and row['setup_dl_sell']:
        if row['pos_in_range_n'] <= 0.25:
            s['8_ShortEsticado_1h'] = {'side': 'short', 'rr': 1.0}

    # 9. "PFR Tendência" (1h PFR Long) - WR Esperada: ~55%
    if tf == '1h' and trend_up and row['setup_pfr_buy']:
        if row['bars_since_high'] <= 20:
            s['9_PFR_Trend_1h'] = {'side': 'long', 'rr': 1.5}

    # 10. "DL Correção" (4h DL Long) - WR Esperada: ~58%
    if tf == '4h' and trend_up and row['setup_dl_buy']:
        if row['upper_wick_pct'] < 20.0:
            s['10_DL_Clean_4h'] = {'side': 'long', 'rr': 1.0}

    return s

# ================= ENGINE =================

def fetch_data(symbol, tf):
    ex = ccxt.mexc({'options': {'defaultType': 'swap'}})
    try:
        # Tenta baixar o máximo permitido paginando
        all_ohlcv = []
        since = None
        for _ in range(3): # Tenta pegar 3 lotes de 1000 candles
            ohlcv = ex.fetch_ohlcv(symbol, tf, limit=1000, since=since)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.5)
            
        df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def prepare_indicators(df):
    x = df.copy()
    # SMAs
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # Trends
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    x['trend_down'] = (x['close'] < x['sma_s']) & (x['close'] < x['sma_l']) & (x['sma_s'] < x['sma_l'])
    
    # Slope (Simples)
    x['slope_up'] = x['sma_s'] > x['sma_s'].shift(SLOPE_LOOKBACK)
    x['slope_down'] = x['sma_s'] < x['sma_s'].shift(SLOPE_LOOKBACK)
    
    # Features
    x['range'] = x['high'] - x['low']
    x['body'] = (x['close'] - x['open']).abs()
    x['upper_wick'] = x['high'] - np.maximum(x['open'], x['close'])
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    
    x['lower_wick_pct'] = (x['lower_wick'] / x['range']) * 100
    x['upper_wick_pct'] = (x['upper_wick'] / x['range']) * 100
    x['ret_5_pct'] = x['close'].pct_change(5) * 100
    x['ma_gap_pct'] = ((x['sma_s'] - x['sma_l']) / x['sma_l']) * 100
    
    # Vol Z
    x['vol_z'] = (x['volume'] - x['volume'].rolling(20).mean()) / x['volume'].rolling(20).std()
    
    # Pos in Range
    roll_hi = x['high'].rolling(20).max()
    roll_lo = x['low'].rolling(20).min()
    x['pos_in_range_n'] = (x['close'] - roll_lo) / (roll_hi - roll_lo)
    
    # New High/Low Context
    x['new_high'] = x['high'] > x['high'].shift(1).rolling(20).max()
    x['new_low'] = x['low'] < x['low'].shift(1).rolling(20).min()
    
    # Bars since (vectorized logic simplified)
    x['bars_since_high'] = x.groupby(x['new_high'].cumsum()).cumcount()
    x['last_high_price'] = x['high'].where(x['new_high']).ffill()
    x['pullback_from_high'] = (x['last_high_price'] - x['close']) / x['last_high_price'] * 100
    
    x['bars_since_low'] = x.groupby(x['new_low'].cumsum()).cumcount()
    x['last_low_price'] = x['low'].where(x['new_low']).ffill()
    x['pullback_from_low'] = (x['close'] - x['last_low_price']) / x['last_low_price'] * 100

    # Signals
    p_low = x['low'].shift(1)
    p2_low = x['low'].shift(2)
    p_high = x['high'].shift(1)
    p2_high = x['high'].shift(2)
    p_close = x['close'].shift(1)
    
    # PFR
    x['setup_pfr_buy'] = (x['low'] < p_low) & (x['low'] < p2_low) & (x['close'] > p_close)
    x['setup_pfr_sell'] = (x['high'] > p_high) & (x['high'] > p2_high) & (x['close'] < p_close)
    
    # DL
    x['setup_dl_buy'] = (x['low'] < p_low) & (x['low'] < p2_low)
    x['setup_dl_sell'] = (x['high'] > p_high) & (x['high'] > p2_high)
    
    return x

def run_simulation(df, i, side, rr):
    entry_price = df.loc[i, 'high'] if side == 'long' else df.loc[i, 'low'] # Tick simulado na execução
    stop_price = df.loc[i, 'low'] if side == 'long' else df.loc[i, 'high']
    
    risk = abs(entry_price - stop_price)
    target = entry_price + (risk * rr) if side == 'long' else entry_price - (risk * rr)
    
    # Check next candle (MAX_WAIT = 1) for entry
    if i + 1 >= len(df): return None
    next_c = df.loc[i+1]
    
    filled = False
    if side == 'long':
        if next_c['high'] >= entry_price: filled = True
    else:
        if next_c['low'] <= entry_price: filled = True
        
    if not filled: return "no_fill"
    
    # Check outcome
    for j in range(i+1, min(i+1+MAX_HOLD_BARS, len(df))):
        c = df.loc[j]
        if side == 'long':
            if c['low'] <= stop_price: return "loss"
            if c['high'] >= target: return "win"
        else:
            if c['high'] >= stop_price: return "loss"
            if c['low'] <= target: return "win"
            
    return "timeout"

def main():
    results = []
    
    for symbol in SYMBOLS:
        clean_sym = symbol.split(":")[0]
        print(f"Processing {clean_sym}...")
        
        # Cache data to avoid refetching for each setup
        data_cache = {}
        for tf in ['1h', '2h', '4h']:
            df = fetch_data(clean_sym, tf)
            if not df.empty:
                data_cache[tf] = prepare_indicators(df)
        
        # Scan
        for tf, df in data_cache.items():
            if df.empty: continue
            
            for i in range(80, len(df)-1):
                # Check definitions
                active_setups = check_setups(df.iloc[i], tf)
                
                for name, params in active_setups.items():
                    res = run_simulation(df, i, params['side'], params['rr'])
                    if res in ['win', 'loss']:
                        results.append({
                            'Symbol': clean_sym,
                            'Setup': name,
                            'Outcome': res
                        })

    # Report
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("Nenhum trade encontrado nos dados baixados.")
        return

    # Pivot Table
    pivot = df_res.pivot_table(index='Setup', columns='Symbol', values='Outcome', 
                               aggfunc=lambda x: f"{100 * (x == 'win').sum() / len(x):.0f}% ({len(x)})")
    
    # Total Avg
    total_stats = df_res.groupby('Setup')['Outcome'].apply(lambda x: f"{100 * (x == 'win').sum() / len(x):.1f}% ({len(x)})")
    pivot['ALL_COINS'] = total_stats

    print("\n=== VALIDAÇÃO MULTI-MOEDA DOS 10 DIAMANTES ===")
    print("Formato: Winrate% (Qtd Trades)")
    print(tabulate(pivot, headers='keys', tablefmt='grid'))
    
    # Save
    pivot.to_csv("results/validation_matrix.csv")

if __name__ == "__main__":
    main()
