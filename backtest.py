import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate

BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

# Timeframes solicitados
TIMEFRAMES = ["15m", "1h", "2h", "4h", "1d"]
SETUPS = ["PFR", "DL", "8.2", "8.3"]

# Parâmetros da Tendência (Regra Absoluta)
SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8  # Inclinação da SMA8

# Parâmetros de Execução
MAX_HOLD_BARS = 50
TICK_SIZE = 0.5 # Aproximado para BTC

# Risco Retorno
RRS = [1.0, 1.5, 2.0]

# Histórico: Precisamos de MUITOS candles de 15m para formar o Diário
# 50.000 candles de 15m = ~520 dias (Suficiente para SMA80 no Diário)
MAX_BARS_FETCH = 50000 
WINDOW_DAYS = 15 # Baixa de 15 em 15 dias para não estourar

# --- UTIL ---
def fmt_pct(val):
    try:
        return f"{val*100:.1f}%".replace(".", ",")
    except: return "-"

def http_get_json(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except: time.sleep(1)
    return None

# --- DADOS ---
def parse_kline(payload):
    if not payload or not payload.get("data"): return pd.DataFrame()
    data = payload["data"]
    if isinstance(data, list) and len(data) > 0:
        cols = ['ts', 'open', 'high', 'low', 'close', 'vol']
        if isinstance(data[0], list): df = pd.DataFrame(data, columns=['time'] + cols[1:])
        else: df = pd.DataFrame(data)
        
        df = df.rename(columns={'time': 'ts', 'vol': 'volume'})
        df = df[['ts', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='s' if df['ts'].iloc[0] < 1e11 else 'ms', utc=True)
        for c in cols[1:]: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.sort_values('ts').reset_index(drop=True)
    return pd.DataFrame()

def fetch_history(symbol):
    all_dfs = []
    end_ts = int(time.time())
    step = WINDOW_DAYS * 86400
    
    print(f"   -> Baixando histórico base (15m) para {symbol}...")
    
    while len(all_dfs) * 24 * 4 * WINDOW_DAYS < MAX_BARS_FETCH + 5000:
        start_ts = end_ts - step
        url = f"{BASE_URL}/contract/kline/{symbol}"
        data = http_get_json(url, {'interval': 'Min15', 'start': start_ts, 'end': end_ts})
        df = parse_kline(data)
        
        if df.empty: break
        all_dfs.append(df)
        
        first_ts = int(df.iloc[0]['ts'].timestamp())
        if first_ts >= end_ts: break
        end_ts = first_ts - 1
        time.sleep(0.1)
        
    if not all_dfs: return pd.DataFrame()
    full_df = pd.concat(all_dfs).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return full_df.tail(MAX_BARS_FETCH).reset_index(drop=True)

def resample(df, rule):
    if rule == '15m': return df.copy()
    mapping = {'1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d'}
    df = df.set_index('ts')
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    res = df.resample(mapping[rule]).agg(agg).dropna()
    return res.reset_index()

# --- INDICADORES ---
def add_indicators(df):
    x = df.copy()
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # Tendência de Alta (Obrigatória)
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    
    # Inclinação (Slope) - Regra: Acima da máxima dos últimos N
    prev_max = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).max()
    x['slope_up'] = x['sma_s'] > prev_max
    
    return x

# --- SINAIS ---
def check_signals(x, i):
    # PFR: Low < Low-1 e Low < Low-2 e Close > Close-1
    pfr = (x['low'].iloc[i] < x['low'].iloc[i-1]) and \
          (x['low'].iloc[i] < x['low'].iloc[i-2]) and \
          (x['close'].iloc[i] > x['close'].iloc[i-1])
          
    # DL: Low < Low-1 e Low < Low-2
    dl = (x['low'].iloc[i] < x['low'].iloc[i-1]) and \
         (x['low'].iloc[i] < x['low'].iloc[i-2])
         
    # 8.2: Fechou abaixo da mínima anterior
    s82 = x['close'].iloc[i] < x['low'].iloc[i-1]
    
    # 8.3: Fechou abaixo do ref i-2 duas vezes
    s83 = (x['close'].iloc[i] < x['close'].iloc[i-2]) and \
          (x['close'].iloc[i-1] < x['close'].iloc[i-2]) and \
          (x['close'].iloc[i-1] >= x['low'].iloc[i-2]) 
               
    return pfr, dl, s82, s83

# --- SIMULAÇÃO ---
def run_backtest(df, tf):
    x = add_indicators(df)
    tick = TICK_SIZE
    trades = []
    
    # Precisa de dados para médias
    start_idx = max(SMA_LONG, SLOPE_LOOKBACK) + 10
    
    for i in range(start_idx, len(x) - 10):
        # 1. Filtro de Tendência + Inclinação
        if not (x['trend_up'].iloc[i] and x['slope_up'].iloc[i]):
            continue
            
        pfr, dl, s82, s83 = check_signals(x, i)
        
        active_setups = []
        if pfr: active_setups.append('PFR')
        if dl and not pfr: active_setups.append('DL') # DL é subconjunto, evita duplicar
        if s82: active_setups.append('8.2')
        if s83: active_setups.append('8.3')
        
        if not active_setups: continue
        
        for setup in active_setups:
            # Regras de Execução Diferenciadas
            if setup in ['8.2', '8.3']:
                max_wait = 3
            else:
                max_wait = 1 # PFR e DL: Vai ou Racha
            
            # Setup Inicial
            trigger_idx = i
            entry_price = x['high'].iloc[i] + tick
            stop_price = x['low'].iloc[i] - tick
            
            filled = False
            fill_idx = -1
            
            # Loop de Entrada (Trailing para 8.2/8.3)
            for w in range(1, max_wait + 1):
                curr_idx = i + w
                if curr_idx >= len(x): break
                curr_bar = x.iloc[curr_idx]
                
                # Check Fill
                if curr_bar['high'] >= entry_price:
                    filled = True
                    fill_idx = curr_idx
                    break
                
                # Logic Update (Se não pegou)
                if setup in ['8.2', '8.3']:
                    # Média tem que continuar subindo
                    if not x['slope_up'].iloc[curr_idx]: break
                    
                    # Se fez nova máxima menor, abaixa a entrada
                    if curr_bar['high'] < (entry_price - tick):
                        entry_price = curr_bar['high'] + tick
                        stop_price = min(stop_price, curr_bar['low'] - tick)
                else:
                    # PFR/DL não espera mais que 1
                    break
            
            if not filled: continue
            
            # Simular Resultado
            res_row = {'timeframe': tf, 'setup': setup}
            
            for rr in RRS:
                risk = entry_price - stop_price
                target = entry_price + (risk * rr)
                outcome = 'loss' # timeout default
                
                for k in range(fill_idx, min(fill_idx + MAX_HOLD_BARS, len(x))):
                    curr = x.iloc[k]
                    hit_stop = curr['low'] <= stop_price
                    hit_target = curr['high'] >= target
                    
                    if hit_stop and hit_target:
                        outcome = 'loss' # Conservador
                        break
                    elif hit_stop:
                        outcome = 'loss'
                        break
                    elif hit_target:
                        outcome = 'win'
                        break
                
                res_row[f"rr_{rr}"] = outcome
            
            trades.append(res_row)
            
    return pd.DataFrame(trades)

def main():
    print(f"--- Backtest Baseline LONG ONLY ({SYMBOL}) ---")
    
    # 1. Download Dados Base (15m)
    df_base = fetch_history(SYMBOL)
    if df_base.empty:
        print("Erro: Sem dados.")
        return

    all_trades = []
    
    # 2. Loop Timeframes
    for tf_name in TIMEFRAMES:
        print(f"Processando {tf_name}...")
        try:
            df_tf = resample(df_base, tf_name)
            if len(df_tf) < SMA_LONG + 50:
                print(f"  > Dados insuficientes para {tf_name}")
                continue
                
            trades = run_backtest(df_tf, tf_name)
            if not trades.empty:
                all_trades.append(trades)
        except Exception as e:
            print(f"  > Erro em {tf_name}: {e}")

    # 3. Consolidar
    if not all_trades:
        print("Nenhum trade gerado.")
        # Salva vazio para não quebrar
        pd.DataFrame(columns=['timeframe', 'setup']).to_csv(f"results/baseline_trades_{SYMBOL}.csv", index=False)
        return
        
    final_df = pd.concat(all_trades, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    final_df.to_csv(f"results/baseline_trades_{SYMBOL}.csv", index=False)
    
    # 4. Relatório Simples
    report_rows = []
    for (tf, setup), g in final_df.groupby(['timeframe', 'setup']):
        row = {'TF': tf, 'Setup': setup, 'Trades': len(g)}
        for rr in RRS:
            col = f"rr_{rr}"
            wins = (g[col] == 'win').sum()
            row[f"WinRate {rr}"] = fmt_pct(wins / len(g))
        report_rows.append(row)
        
    summary = pd.DataFrame(report_rows)
    summary = summary.sort_values(['TF', 'Setup'])
    
    print("\n=== RESULTADOS BASELINE (SEM FILTROS EXTRAS) ===")
    print(tabulate(summary, headers='keys', tablefmt='grid'))
    
    with open(f"results/baseline_summary_{SYMBOL}.md", "w") as f:
        f.write(f"# Baseline Long Only - {SYMBOL}\n\n")
        f.write(tabulate(summary, headers="keys", tablefmt="pipe", showindex=False))

if __name__ == "__main__":
    main()
