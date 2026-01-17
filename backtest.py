import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate  # <--- Faltava este import

BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

TIMEFRAMES = [x.strip().lower() for x in os.getenv("TIMEFRAMES", "1h,2h,4h,1d,1w").split(",") if x.strip()]
SETUPS = [x.strip().upper() for x in os.getenv("SETUPS", "PFR,DL").split(",") if x.strip()]

# Médias
SMA_SHORT = int(os.getenv("SMA_SHORT", "8"))
SMA_LONG = int(os.getenv("SMA_LONG", "80"))
SLOPE_LOOKBACK = int(os.getenv("SLOPE_LOOKBACK", "8"))

# Indicadores
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
LOOKBACK_N = int(os.getenv("LOOKBACK_N", "20"))
EXTREME_LOOKBACK = int(os.getenv("EXTREME_LOOKBACK", "20"))

# RRs (1.0, 1.5, 2.0)
RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]

# Execução
AMBIGUOUS_POLICY = os.getenv("AMBIGUOUS_POLICY", "loss").lower()
MAX_ENTRY_WAIT_BARS = int(os.getenv("MAX_ENTRY_WAIT_BARS", "1")) 
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "50"))

# Dados
MAX_BARS_1H = int(os.getenv("MAX_BARS_1H", "25000"))
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "30"))
TICK_SIZE = float(os.getenv("TICK_SIZE", "0"))

DEBUG = os.getenv("DEBUG", "0") == "1"

# --- UTIL ---
def fmt_pct(val):
    try:
        if val is None or np.isnan(val): return "-"
        return f"{val*100:.1f}%".replace(".", ",")
    except: return "-"

def http_get_json(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(1)
    return None

# --- DADOS ---
def parse_kline(payload):
    if not payload: return pd.DataFrame()
    data = payload.get("data", []) or payload.get("result", [])
    if not data: return pd.DataFrame()
    
    if isinstance(data, dict) and "time" in data:
        df = pd.DataFrame(data)
    elif isinstance(data, list):
        if len(data) == 0: return pd.DataFrame()
        if isinstance(data[0], list):
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        else:
            df = pd.DataFrame(data)
            
    rename_map = {'time': 'ts', 'vol': 'volume'}
    df = df.rename(columns=rename_map)
    cols = ['ts', 'open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    
    df = df[cols].copy()
    df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='s' if df['ts'].median() < 1e11 else 'ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df.sort_values('ts').reset_index(drop=True)

def fetch_history(symbol, max_bars):
    all_dfs = []
    end_ts = int(time.time())
    step = WINDOW_DAYS * 86400
    
    # Loop seguro para garantir histórico
    while len(all_dfs) * 24 * WINDOW_DAYS < max_bars + 5000: 
        start_ts = end_ts - step
        url = f"{BASE_URL}/contract/kline/{symbol}"
        data = http_get_json(url, {'interval': 'Min60', 'start': start_ts, 'end': end_ts})
        df = parse_kline(data)
        
        if df.empty: break
        all_dfs.append(df)
        
        first_ts = int(df.iloc[0]['ts'].timestamp())
        if first_ts >= end_ts: break
        end_ts = first_ts - 1
        time.sleep(0.2)
        
    if not all_dfs: return pd.DataFrame()
    full_df = pd.concat(all_dfs).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return full_df.tail(max_bars).reset_index(drop=True)

def resample(df, rule):
    df = df.set_index('ts')
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    res = df.resample(rule).agg(agg).dropna()
    return res.reset_index()

# --- INDICADORES ---
def add_indicators(df):
    x = df.copy()
    
    # Médias
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # Tendência de Alta
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    
    # Inclinação (Slope)
    prev_max = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).max()
    x['slope_up'] = x['sma_s'] > prev_max
    x['slope_strength'] = x['sma_s'] - prev_max 
    
    # ATR
    x['tr'] = np.maximum(x['high'] - x['low'], 
                         np.maximum(abs(x['high'] - x['close'].shift(1)), 
                                    abs(x['low'] - x['close'].shift(1))))
    x['atr'] = x['tr'].rolling(ATR_PERIOD).mean()
    x['atr_pct'] = (x['atr'] / x['close']) * 100
    
    # Candle Stats
    x['range'] = x['high'] - x['low']
    x['body'] = abs(x['close'] - x['open'])
    x['upper_wick'] = x['high'] - np.maximum(x['open'], x['close'])
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    
    x['range_pct'] = (x['range'] / x['close']) * 100
    x['body_pct'] = (x['body'] / x['range']) * 100
    x['lower_wick_pct'] = (x['lower_wick'] / x['range']) * 100
    x['clv'] = (x['close'] - x['low']) / x['range']
    
    # Momentum
    x['ret_1_pct'] = x['close'].pct_change(1) * 100
    x['ret_5_pct'] = x['close'].pct_change(5) * 100
    
    # RSI
    delta = x['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(RSI_PERIOD).mean()
    ma_down = down.rolling(RSI_PERIOD).mean()
    rs = ma_up / ma_down
    x['rsi'] = 100 - (100 / (1 + rs))
    
    # Contexto de Topo
    roll_hi = x['high'].rolling(EXTREME_LOOKBACK).max()
    x['is_new_high'] = x['high'] >= roll_hi
    
    x['grp_hi'] = x['is_new_high'].cumsum()
    x['bars_since_new_high'] = x.groupby('grp_hi').cumcount()
    
    x['last_high_price'] = x['high'].where(x['is_new_high']).ffill()
    x['pullback_from_new_high_pct'] = (x['last_high_price'] - x['close']) / x['last_high_price'] * 100
    
    x['dist_to_sma80_pct'] = (x['close'] - x['sma_l']) / x['sma_l'] * 100
    
    rn_hi = x['high'].rolling(LOOKBACK_N).max()
    rn_lo = x['low'].rolling(LOOKBACK_N).min()
    x['pos_in_range_n'] = (x['close'] - rn_lo) / (rn_hi - rn_lo)
    
    x['vol_z'] = (x['volume'] - x['volume'].rolling(20).mean()) / x['volume'].rolling(20).std()

    return x

# --- SINAIS (LONG ONLY) ---
def check_signals(x, i):
    pfr 
