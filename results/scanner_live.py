import os
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# ================= CONFIGURAÇÕES =================
SYMBOL_LIST = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "MATIC/USDT:USDT", "DOT/USDT:USDT", "TRX/USDT:USDT",
    "LTC/USDT:USDT", "SHIB/USDT:USDT", "UNI/USDT:USDT", "ATOM/USDT:USDT"
] 
# (Adicione mais pares do Top 50 se quiser volume)

TIMEFRAMES = ["1h", "2h", "4h"]

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Médias
SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8

# Filtros Absolutos (Baseado nos seus dados)
# Se o setup acontecer mas não bater nesses valores, o bot IGNORA.
FILTERS = {
    # Formato: "TF_SETUP_SIDE": lambda row: condicao
    
    # 1h DL Long (Winrate 70% no backtest quando pos_in_range > 0.9)
    "1h_DL_long": lambda r: r['pos_in_range_n'] >= 0.89,

    # 1h PFR Long (Winrate 70% quando ret_5_pct > 1.0)
    "1h_PFR_long": lambda r: r['ret_5_pct'] >= 1.0,

    # 2h PFR Long (Winrate ~72% quando lower_wick > 47%)
    "2h_PFR_long": lambda r: r['lower_wick_pct'] >= 45.0,

    # 2h DL Long (Winrate ~75% quando ret_5 > 1.3 ou vol_z > 1.1)
    "2h_DL_long": lambda r: (r['ret_5_pct'] >= 1.2) or (r['vol_z'] >= 1.0),

    # 1h PFR Short (Winrate ~65% quando pullback é raso < 2%)
    "1h_PFR_short": lambda r: r['pullback_from_new_low_pct'] < 2.0,
    
    # 4h PFR Long (Winrate ~60% quando dist_to_high é baixo)
    "4h_PFR_long": lambda r: r['dist_to_high_n_pct'] < 0.9,
}

# ================= FUNÇÕES TÉCNICAS =================

def get_exchange():
    return ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

def fetch_candles(ex, symbol, tf, limit=150):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        print(f"Erro ao baixar {symbol} {tf}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    x = df.copy()
    
    # Médias
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # Tendência & Inclinação
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    x['trend_down'] = (x['close'] < x['sma_s']) & (x['close'] < x['sma_l']) & (x['sma_s'] < x['sma_l'])
    
    prev_max = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).max()
    prev_min = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).min()
    x['slope_up'] = x['sma_s'] > prev_max
    x['slope_down'] = x['sma_s'] < prev_min

    # Features Avançadas (Do seu relatório)
    
    # 1. Retorno 5 candles (%)
    x['ret_5_pct'] = x['close'].pct_change(5) * 100.0
    
    # 2. Position in Range (N=20)
    lookback_n = 20
    roll_hi = x['high'].rolling(lookback_n).max()
    roll_lo = x['low'].rolling(lookback_n).min()
    x['pos_in_range_n'] = (x['close'] - roll_lo) / (roll_hi - roll_lo)
    x['dist_to_high_n_pct'] = (roll_hi - x['close']) / x['close'] * 100.0
    
    # 3. Candle Wick/Body stats
    x['range'] = x['high'] - x['low']
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    x['lower_wick_pct'] = (x['lower_wick'] / x['range']) * 100.0
    
    # 4. Volume Z-Score
    vol_mean = x['volume'].rolling(20).mean()
    vol_std = x['volume'].rolling(20).std()
    x['vol_z'] = (x['volume'] - vol_mean) / vol_std

    # 5. Pullback context (simplificado para o live)
    # Quanto subiu desde a mínima de 20 periodos? (para short)
    # Quanto caiu desde a máxima de 20 periodos? (para long)
    x['pullback_from_new_high_pct'] = (roll_hi - x['close']) / roll_hi * 100.0
    x['pullback_from_new_low_pct'] = (x['close'] - roll_lo) / roll_lo * 100.0

    return x

def check_signals(df, symbol, tf):
    last = df.iloc[-2] # Candle Fechado (sinal confirmada)
    curr = df.iloc[-1] # Candle Aberto (onde entramos no rompimento)
    
    # Índices para checar setups (i, i-1, i-2)
    # Usamos iloc[-2] como o candle do sinal (i)
    
    # --- DADOS DO CANDLE SINAL (fechado) ---
    i_close = last['close']
    i_low = last['low']
    i_high = last['high']
    
    prev_close = df.iloc[-3]['close']
    prev_low = df.iloc[-3]['low']
    prev_high = df.iloc[-3]['high']
    
    prev2_low = df.iloc[-4]['low']
    prev2_high = df.iloc[-4]['high']
    
    signals = []

    # --- PFR ---
    pfr_buy = (i_low < prev_low) and (i_low < prev2_low) and (i_close > prev_close)
    pfr_sell = (i_high > prev_high) and (i_high > prev2_high) and (i_close < prev_close)

    # --- DL (Dave Landry) ---
    dl_buy = (i_low < prev_low) and (i_low < prev2_low)
    dl_sell = (i_high > prev_high) and (i_high > prev2_high)

    # --- FILTROS DE TENDÊNCIA E INCLINAÇÃO ---
    valid_long = last['trend_up'] and last['slope_up']
    valid_short = last['trend_down'] and last['slope_down']

    # Checar Longs
    if valid_long:
        setup_name = None
        if pfr_buy: setup_name = "PFR"
        elif dl_buy: setup_name = "DL"
        
        if setup_name:
            key = f"{tf}_{setup_name}_long"
            # Se tivermos uma regra específica de 'diamante' para esse TF/Setup
            if key in FILTERS:
                if FILTERS[key](last):
                    signals.append({
                        "symbol": symbol, "tf": tf, "side": "LONG", "setup": setup_name,
                        "entry": i_high, # Stop buy 1 tick acima (lógica aproximada)
                        "stop": i_low,
                        "metrics": f"PosRange: {last['pos_in_range_n']:.2f} | Ret5: {last['ret_5_pct']:.2f}%"
                    })
            else:
                # Se não tem filtro específico mas é PFR em TF alto (geralmente bom), avisa
                if setup_name == "PFR" and tf in ["2h", "4h"]:
                     signals.append({
                        "symbol": symbol, "tf": tf, "side": "LONG", "setup": setup_name,
                        "entry": i_high, "stop": i_low,
                        "metrics": "Padrão Base (Sem filtro extra)"
                    })

    # Checar Shorts
    if valid_short:
        setup_name = None
        if pfr_sell: setup_name = "PFR"
        elif dl_sell: setup_name = "DL"
        
        if setup_name:
            key = f"{tf}_{setup_name}_short"
            if key in FILTERS:
                if FILTERS[key](last):
                    signals.append({
                        "symbol": symbol, "tf": tf, "side": "SHORT", "setup": setup_name,
                        "entry": i_low,
                        "stop": i_high,
                        "metrics": f"PullbackNewLow: {last['pullback_from_new_low_pct']:.2f}%"
                    })

    return signals

def send_telegram(sig):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    
    emoji = "🟢" if sig['side'] == "LONG" else "🔴"
    tick_calc = sig['entry'] * 0.0001 # tick aproximado para mensagem
    entry_price = sig['entry'] + tick_calc if sig['side'] == "LONG" else sig['entry'] - tick_calc
    
    msg = (
        f"{emoji} <b>DIAMOND SETUP DETECTED</b>\n\n"
        f"💎 <b>{sig['symbol']}</b> ({sig['tf']})\n"
        f"Setup: <b>{sig['setup']} {sig['side']}</b>\n"
        f"Entry (Stop Lmt): {entry_price:.5g}\n"
        f"Stop Loss: {sig['stop']:.5g}\n\n"
        f"📊 <i>Stats: {sig['metrics']}</i>\n"
        f"⏳ <i>Valid only for current candle!</i>"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": msg, 
            "parse_mode": "HTML"
        })
    except Exception as e:
        print(f"Erro Telegram: {e}")

# ================= LOOP PRINCIPAL =================
def run_scanner():
    print(f"--- Iniciando Scan em {datetime.now()} ---")
    ex = get_exchange()
    
    for symbol in SYMBOL_LIST:
        # A API da MEXC usa formato "BTC/USDT" para spot, para perp via ccxt depende.
        # Ajuste o símbolo conforme o ccxt.mexc mapeia perps. Normalmente é "BTC/USDT:USDT" no swap.
        clean_sym = symbol.split(":")[0] 
        
        for tf in TIMEFRAMES:
            df = fetch_candles(ex, clean_sym, tf)
            if df.empty: continue
            
            df_ind = calculate_indicators(df)
            signals = check_signals(df_ind, clean_sym, tf)
            
            for s in signals:
                print(f"!!! SINAL ENCONTRADO: {s}")
                send_telegram(s)
            
            time.sleep(0.2) # Evitar rate limit

if __name__ == "__main__":
    try:
        run_scanner()
    except Exception as e:
        print(f"Fatal error: {e}")
