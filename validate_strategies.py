import os
import time
import ccxt
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime, timedelta

# ================= CONFIGURAÇÕES =================
# Top 10 Alts com boa liquidez histórica na MEXC
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", 
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT"
]

# Configurações de Download
# Vamos buscar aprox. 24 meses para trás (aprox 17.500 horas)
DAYS_HISTORY = 730 
MAX_RETRIES = 5

# Configurações Técnicas (Iguais ao seu backtest original)
SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8
MAX_ENTRY_WAIT_BARS = 1 
MAX_HOLD_BARS = 50

# ================= DEFINIÇÃO DOS 10 SETUPS =================
def check_setups(row, tf):
    s = {}
    
    # Base Filters (comuns a todos)
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

def fetch_deep_history(symbol, tf):
    """
    Baixa histórico profundo paginando via 'since'.
    Tenta pegar até 2 anos atrás ou o máximo que a API permitir.
    """
    ex = ccxt.mexc({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    
    # Calcular timestamp de início (X dias atrás)
    start_time = int((datetime.now() - timedelta(days=DAYS_HISTORY)).timestamp() * 1000)
    
    all_ohlcv = []
    since = start_time
    
    print(f"   -> Baixando histórico para {symbol} [{tf}] desde {datetime.fromtimestamp(start_time/1000)}...")
    
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, tf, limit=1000, since=since)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Atualiza 'since' para o último candle + 1ms
            last_ts = ohlcv[-1][0]
            since = last_ts + 1
            
            # Se chegou perto de agora, para
            if last_ts >= (time.time() * 1000) - 60000 * 60: # 1h atrás
                break
                
            # Limite de segurança (ex: 25k candles)
            if len(all_ohlcv) > 25000:
                break
                
            time.sleep(0.5) # Pausa amigável
            
        except Exception as e:
            print(f"      Erro de conexão: {e}. Tentando continuar...")
            time.sleep(2)
            continue

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    print(f"      Total baixado: {len(df)} candles.")
    return df

def prepare_indicators(df):
    x = df.copy()
    
    # 1. Médias
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # 2. Trend & Slope
    # Trend
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    x['trend_down'] = (x['close'] < x['sma_s']) & (x['close'] < x['sma_l']) & (x['sma_s'] < x['sma_l'])
    
    # Slope (usando sua regra final: acima/abaixo dos ultimos 8)
    prev_max = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).max()
    prev_min = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).min()
    x['slope_up'] = x['sma_s'] > prev_max
    x['slope_down'] = x['sma_s'] < prev_min

    # 3. Features
    x['range'] = x['high'] - x['low']
    x['upper_wick'] = x['high'] - np.maximum(x['open'], x['close'])
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    
    # Evitar div por zero
    x['lower_wick_pct'] = 0.0
    x['upper_wick_pct'] = 0.0
    mask_rng = x['range'] > 0
    x.loc[mask_rng, 'lower_wick_pct'] = (x.loc[mask_rng, 'lower_wick'] / x.loc[mask_rng, 'range']) * 100
    x.loc[mask_rng, 'upper_wick_pct'] = (x.loc[mask_rng, 'upper_wick'] / x.loc[mask_rng, 'range']) * 100
    
    x['ret_5_pct'] = x['close'].pct_change(5) * 100
    x['ma_gap_pct'] = ((x['sma_s'] - x['sma_l']) / x['sma_l']) * 100
    
    # Vol Z
    vol_mean = x['volume'].rolling(20).mean()
    vol_std = x['volume'].rolling(20).std()
    x['vol_z'] = 0.0
    mask_vol = vol_std > 0
    x.loc[mask_vol, 'vol_z'] = (x.loc[mask_vol, 'volume'] - vol_mean) / vol_std
    
    # Pos in Range
    roll_hi = x['high'].rolling(20).max()
    roll_lo = x['low'].rolling(20).min()
    x['pos_in_range_n'] = 0.0
    mask_rng2 = (roll_hi - roll_lo) > 0
    x.loc[mask_rng2, 'pos_in_range_n'] = (x.loc[mask_rng2, 'close'] - roll_lo) / (roll_hi - roll_lo)
    
    # New High/Low Context
    x['new_high'] = x['high'] > x['high'].shift(1).rolling(20).max()
    x['new_low'] = x['low'] < x['low'].shift(1).rolling(20).min()
    
    # Bars since & Pullback
    # (Simulação de vetorização para performance)
    x['grp_hi'] = x['new_high'].cumsum()
    x['bars_since_high'] = x.groupby('grp_hi').cumcount()
    x['last_high_price'] = x['high'].where(x['new_high']).ffill()
    x['pullback_from_high'] = (x['last_high_price'] - x['close']) / x['last_high_price'] * 100
    
    x['grp_lo'] = x['new_low'].cumsum()
    x['bars_since_low'] = x.groupby('grp_lo').cumcount()
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
    # Tick size aproximado (0.0001% do preço para simplificar)
    tick = df.loc[i, 'close'] * 0.0001
    
    # Entrada no rompimento (candle i+1)
    if side == 'long':
        entry_trigger = df.loc[i, 'high'] + tick
        stop_price = df.loc[i, 'low'] - tick
    else:
        entry_trigger = df.loc[i, 'low'] - tick
        stop_price = df.loc[i, 'high'] + tick
    
    risk = abs(entry_trigger - stop_price)
    if risk == 0: return None
    
    target = entry_trigger + (risk * rr) if side == 'long' else entry_trigger - (risk * rr)
    
    # 1. Fill Check (MAX_ENTRY_WAIT_BARS = 1)
    if i + 1 >= len(df): return None
    next_c = df.loc[i+1]
    
    filled = False
    if side == 'long':
        if next_c['high'] >= entry_trigger: filled = True
    else:
        if next_c['low'] <= entry_trigger: filled = True
        
    if not filled: return "no_fill"
    
    # 2. Outcome Check
    # Começa a checar do próprio candle de entrada (i+1)
    for j in range(i+1, min(i+1+MAX_HOLD_BARS, len(df))):
        c = df.loc[j]
        
        # Política conservadora (LOSS): se bater stop e target no mesmo candle, assume loss
        if side == 'long':
            hit_stop = c['low'] <= stop_price
            hit_target = c['high'] >= target
            if hit_stop: return "loss"
            if hit_target: return "win"
        else:
            hit_stop = c['high'] >= stop_price
            hit_target = c['low'] <= target
            if hit_stop: return "loss"
            if hit_target: return "win"
            
    return "timeout"

def main():
    results = []
    
    print(f"Iniciando Validação de {len(SYMBOLS)} moedas com Deep History...")
    
    for symbol in SYMBOLS:
        clean_sym = symbol.split(":")[0]
        
        # Cache data
        data_cache = {}
        for tf in ['1h', '2h', '4h']:
            df = fetch_deep_history(clean_sym, tf)
            if not df.empty:
                # Filtrar para ter dados suficientes para indicadores
                if len(df) > 200:
                    data_cache[tf] = prepare_indicators(df)
        
        # Scan
        for tf, df in data_cache.items():
            if df.empty: continue
            
            # Loop nos candles
            start_idx = 100 # Margem segura para indicadores
            for i in range(start_idx, len(df)-1):
                # Verifica se algum setup ativou neste candle
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
        print("Nenhum trade encontrado.")
        return

    # Pivot Table
    def agg_fmt(x):
        total = len(x)
        wins = (x == 'win').sum()
        wr = 100 * wins / total if total > 0 else 0
        return f"{wr:.0f}% ({total})"

    pivot = df_res.pivot_table(index='Setup', columns='Symbol', values='Outcome', aggfunc=agg_fmt, fill_value="-")
    
    # Total Avg
    total_stats = df_res.groupby('Setup')['Outcome'].apply(lambda x: f"{100 * (x == 'win').sum() / len(x):.1f}% ({len(x)})")
    pivot['ALL_COINS'] = total_stats

    # Sort index to keep order
    try:
        pivot = pivot.sort_index(key=lambda x: x.str.split('_').str[0].astype(int))
    except:
        pass

    print("\n=== VALIDAÇÃO DEEP HISTORY (2 ANOS) ===")
    print(tabulate(pivot, headers='keys', tablefmt='grid'))
    
    pivot.to_csv("results/validation_deep.csv")

if __name__ == "__main__":
    main()
