import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime

# ================= CONFIGURAÇÕES =================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

TIMEFRAMES = [x.strip().lower() for x in os.getenv("TIMEFRAMES", "1h,2h,4h,1d,1w").split(",") if x.strip()]
SETUPS = [x.strip().upper() for x in os.getenv("SETUPS", "PFR,DL,8.2,8.3").split(",") if x.strip()]

SMA_SHORT = int(os.getenv("SMA_SHORT", "8"))
SMA_LONG = int(os.getenv("SMA_LONG", "80"))
SLOPE_LOOKBACK = int(os.getenv("SLOPE_LOOKBACK", "8"))

ATR_PERIOD = 14
RSI_PERIOD = 14
LOOKBACK_N = 20
EXTREME_LOOKBACK = 20

RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]
AMBIGUOUS_POLICY = os.getenv("AMBIGUOUS_POLICY", "loss").lower()
MAX_ENTRY_WAIT_BARS = int(os.getenv("MAX_ENTRY_WAIT_BARS", "3"))
MAX_HOLD_BARS = 50

MAX_BARS_1H = int(os.getenv("MAX_BARS_1H", "25000"))
WINDOW_DAYS = 30
TICK_SIZE = float(os.getenv("TICK_SIZE", "0"))

DEBUG = os.getenv("DEBUG", "0") == "1"

# ================= FUNÇÕES UTILITÁRIAS =================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

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
        except Exception as e:
            if i == tries - 1: log(f"Erro HTTP: {e}")
            time.sleep(1)
    return None

# ================= DADOS =================
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
    
    log(f"Baixando dados para {symbol} (Alvo: {max_bars} candles)...")
    
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
    log(f"Download concluído: {len(full_df)} candles baixados.")
    return full_df.tail(max_bars).reset_index(drop=True)

def resample(df, rule):
    df = df.set_index('ts')
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    res = df.resample(rule).agg(agg).dropna()
    return res.reset_index()

# ================= INDICADORES =================
def add_indicators(df):
    x = df.copy()
    x['sma_s'] = x['close'].rolling(SMA_SHORT).mean()
    x['sma_l'] = x['close'].rolling(SMA_LONG).mean()
    
    # Tendência & Slope
    x['trend_up'] = (x['close'] > x['sma_s']) & (x['close'] > x['sma_l']) & (x['sma_s'] > x['sma_l'])
    prev_max = x['sma_s'].shift(1).rolling(SLOPE_LOOKBACK).max()
    x['slope_up'] = x['sma_s'] > prev_max
    x['slope_strength'] = x['sma_s'] - prev_max 
    
    # ATR & Candle Stats
    x['tr'] = np.maximum(x['high'] - x['low'], np.maximum(abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))))
    x['atr'] = x['tr'].rolling(ATR_PERIOD).mean()
    x['atr_pct'] = (x['atr'] / x['close']) * 100
    
    x['range'] = x['high'] - x['low']
    x['body'] = abs(x['close'] - x['open'])
    # Evita divisão por zero
    safe_range = x['range'].replace(0, np.nan)
    x['lower_wick'] = np.minimum(x['open'], x['close']) - x['low']
    x['lower_wick_pct'] = (x['lower_wick'] / safe_range) * 100
    x['clv'] = (x['close'] - x['low']) / safe_range
    x['ret_5_pct'] = x['close'].pct_change(5) * 100
    
    # RSI
    delta = x['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(RSI_PERIOD).mean() / down.rolling(RSI_PERIOD).mean()
    x['rsi'] = 100 - (100 / (1 + rs))
    
    # Contexto Topo
    roll_hi = x['high'].rolling(EXTREME_LOOKBACK).max()
    x['is_new_high'] = x['high'] >= roll_hi
    x['grp_hi'] = x['is_new_high'].cumsum()
    x['bars_since_new_high'] = x.groupby('grp_hi').cumcount()
    x['last_high_price'] = x['high'].where(x['is_new_high']).ffill()
    x['pullback_from_new_high_pct'] = (x['last_high_price'] - x['close']) / x['last_high_price'] * 100
    x['dist_to_sma80_pct'] = (x['close'] - x['sma_l']) / x['sma_l'] * 100
    
    rn_hi = x['high'].rolling(LOOKBACK_N).max()
    rn_lo = x['low'].rolling(LOOKBACK_N).min()
    rng_diff = (rn_hi - rn_lo).replace(0, np.nan)
    x['pos_in_range_n'] = (x['close'] - rn_lo) / rng_diff
    x['vol_z'] = (x['volume'] - x['volume'].rolling(20).mean()) / x['volume'].rolling(20).std()

    return x

# ================= SINAIS E BACKTEST =================
def check_signals(x, i):
    # PFR
    pfr = (x['low'].iloc[i] < x['low'].iloc[i-1]) and \
          (x['low'].iloc[i] < x['low'].iloc[i-2]) and \
          (x['close'].iloc[i] > x['close'].iloc[i-1])
          
    # DL
    dl = (x['low'].iloc[i] < x['low'].iloc[i-1]) and \
         (x['low'].iloc[i] < x['low'].iloc[i-2])
         
    # 8.2 (Fechou abaixo da mínima anterior)
    setup_82 = x['close'].iloc[i] < x['low'].iloc[i-1]
    
    # 8.3 (Fechou abaixo do ref i-2 duas vezes, sem ter ativado 8.2 antes)
    setup_83 = (x['close'].iloc[i] < x['close'].iloc[i-2]) and \
               (x['close'].iloc[i-1] < x['close'].iloc[i-2]) and \
               (x['close'].iloc[i-1] >= x['low'].iloc[i-2]) 
               
    return pfr, dl, setup_82, setup_83

def run_backtest(df, tf):
    try:
        x = add_indicators(df)
        tick = TICK_SIZE if TICK_SIZE > 0 else (df['close'].iloc[-1] * 0.0001)
        
        trades = []
        start_idx = max(SMA_LONG, 50)
        
        for i in range(start_idx, len(x) - MAX_ENTRY_WAIT_BARS - 5):
            # Filtro Global: Tendência + Inclinação
            if not (x['trend_up'].iloc[i] and x['slope_up'].iloc[i]):
                continue
                
            pfr, dl, s82, s83 = check_signals(x, i)
            
            active_setups = []
            if pfr and 'PFR' in SETUPS: active_setups.append('PFR')
            if dl and 'DL' in SETUPS and not pfr: active_setups.append('DL')
            if s82 and '8.2' in SETUPS: active_setups.append('8.2')
            if s83 and '8.3' in SETUPS: active_setups.append('8.3')
            
            if not active_setups: continue
            
            for setup in active_setups:
                entry_price = x['high'].iloc[i] + tick
                stop_price = x['low'].iloc[i] - tick
                
                filled = False
                fill_idx = -1
                
                # Lógica de Entrada (com deslocamento para 8.2/8.3)
                for w in range(1, MAX_ENTRY_WAIT_BARS + 1):
                    curr_idx = i + w
                    if curr_idx >= len(x): break
                    curr_bar = x.iloc[curr_idx]
                    
                    if curr_bar['high'] >= entry_price:
                        filled = True
                        fill_idx = curr_idx
                        break
                    
                    if setup in ['8.2', '8.3']:
                        if not x['slope_up'].iloc[curr_idx]: break
                        # Se fez máxima menor, desloca a entrada
                        if curr_bar['high'] < (entry_price - tick):
                            entry_price = curr_bar['high'] + tick
                            stop_price = min(stop_price, curr_bar['low'] - tick)
                    else:
                        # PFR/DL só espera 1 candle se configurado assim
                        if w >= 1 and MAX_ENTRY_WAIT_BARS == 1: break
                    
                if not filled: continue
                
                # Executou
                feat = {
                    'timeframe': tf, 'setup': setup,
                    'slope_strength': x['slope_strength'].iloc[i],
                    'bars_since_new_high': x['bars_since_new_high'].iloc[i],
                    'pullback_from_new_high_pct': x['pullback_from_new_high_pct'].iloc[i],
                    'dist_to_sma80_pct': x['dist_to_sma80_pct'].iloc[i],
                    'atr_pct': x['atr_pct'].iloc[i],
                    'clv': x['clv'].iloc[i],
                    'lower_wick_pct': x['lower_wick_pct'].iloc[i],
                    'pos_in_range_n': x['pos_in_range_n'].iloc[i],
                    'ret_5_pct': x['ret_5_pct'].iloc[i],
                    'vol_z': x['vol_z'].iloc[i]
                }
                
                for rr in RRS:
                    risk = entry_price - stop_price
                    target = entry_price + (risk * rr)
                    outcome = 'loss'
                    
                    for k in range(fill_idx, min(fill_idx + MAX_HOLD_BARS, len(x))):
                        curr = x.iloc[k]
                        hit_stop = curr['low'] <= stop_price
                        hit_target = curr['high'] >= target
                        
                        if hit_stop and hit_target:
                            outcome = AMBIGUOUS_POLICY
                            break
                        elif hit_stop:
                            outcome = 'loss'
                            break
                        elif hit_target:
                            outcome = 'win'
                            break
                    
                    feat[f"rr_{rr}"] = outcome
                
                trades.append(feat)
                
        return pd.DataFrame(trades)
    except Exception as e:
        log(f"Erro processando {tf}: {e}")
        return pd.DataFrame()

# --- ANÁLISE ---
def analyze_buckets(trades_df):
    if trades_df.empty: return pd.DataFrame()
    features = [
        'slope_strength', 'bars_since_new_high', 'pullback_from_new_high_pct',
        'dist_to_sma80_pct', 'atr_pct', 'clv', 'lower_wick_pct', 
        'pos_in_range_n', 'ret_5_pct', 'vol_z'
    ]
    report_rows = []
    
    for rr in RRS:
        rr_col = f"rr_{rr}"
        df_clean = trades_df[trades_df[rr_col].isin(['win', 'loss'])].copy()
        
        for (tf, setup), g in df_clean.groupby(['timeframe', 'setup']):
            total = len(g)
            if total < 15: continue
            
            wins = (g[rr_col] == 'win').sum()
            wr_base = wins / total
            
            report_rows.append({
                'TF': tf, 'Setup': setup, 'RR': rr, 
                'Feature': 'ALL', 'Bucket': 'ALL',
                'Trades': total, 'WinRate': wr_base,
                'Thr_Lo': np.nan, 'Thr_Hi': np.nan
            })
            
            for feat in features:
                try:
                    q25 = g[feat].quantile(0.25)
                    g_low = g[g[feat] <= q25]
                    if len(g_low) >= 10:
                        w_l = (g_low[rr_col] == 'win').sum()
                        report_rows.append({
                            'TF': tf, 'Setup': setup, 'RR': rr,
                            'Feature': feat, 'Bucket': 'Low25',
                            'Trades': len(g_low), 'WinRate': w_l/len(g_low),
                            'Thr_Lo': np.nan, 'Thr_Hi': q25
                        })
                    q75 = g[feat].quantile(0.75)
                    g_high = g[g[feat] >= q75]
                    if len(g_high) >= 10:
                        w_h = (g_high[rr_col] == 'win').sum()
                        report_rows.append({
                            'TF': tf, 'Setup': setup, 'RR': rr,
                            'Feature': feat, 'Bucket': 'High25',
                            'Trades': len(g_high), 'WinRate': w_h/len(g_high),
                            'Thr_Lo': q75, 'Thr_Hi': np.nan
                        })
                except: pass
    return pd.DataFrame(report_rows)

def main():
    log(f"--- Iniciando Backtest LONG ONLY para {SYMBOL} ---")
    
    # 1. Preparar pasta
    os.makedirs("results", exist_ok=True)
    
    # 2. Baixar Dados
    df_raw = fetch_history(SYMBOL, MAX_BARS_1H)
    if df_raw.empty:
        log("ERRO FATAL: Não foi possível baixar dados.")
        # Forçar criação de arquivo vazio para debug
        with open("results/ERROR_NO_DATA.txt", "w") as f: f.write("API falhou ou símbolo inválido.")
        return

    # 3. Processar TFs
    all_trades = []
    tf_map = {'1h': df_raw}
    if '2h' in TIMEFRAMES: tf_map['2h'] = resample(df_raw, '2h')
    if '4h' in TIMEFRAMES: tf_map['4h'] = resample(df_raw, '4h')
    if '1d' in TIMEFRAMES: tf_map['1d'] = resample(df_raw, '1D')
    if '1w' in TIMEFRAMES: tf_map['1w'] = resample(df_raw, 'W-SUN')
    
    for tf_name in TIMEFRAMES:
        if tf_name not in tf_map: continue
        log(f"Processando {tf_name}...")
        df_tf = tf_map[tf_name]
        trades = run_backtest(df_tf, tf_name)
        if not trades.empty:
            all_trades.append(trades)
        else:
            log(f"  > Sem trades em {tf_name}")
        
    # 4. Consolidar Resultados
    if not all_trades:
        log("Nenhum trade encontrado em nenhum timeframe.")
        final_df = pd.DataFrame(columns=['timeframe', 'setup'])
    else:
        final_df = pd.concat(all_trades, ignore_index=True)
        
    # 5. Salvar SEMPRE
    log("Salvando arquivos CSV...")
    final_df.to_csv(f"results/backtest_trades_{SYMBOL}.csv", index=False)
    
    # 6. Analisar e Gerar Markdown
    log("Gerando análise...")
    report = analyze_buckets(final_df)
    
    # Salvar Full Report
    report.to_csv(f"results/full_report_{SYMBOL}.csv", index=False)
    
    # Salvar Best Patterns (Markdown)
    with open(f"results/best_long_patterns_{SYMBOL}.md", "w") as f:
        f.write(f"# Top Long Setups (Winrate > 60%) - {SYMBOL}\n\n")
        f.write(f"Data: {datetime.now()}\n\n")
        
        if not report.empty:
            diamonds = report[report['WinRate'] >= 0.60].sort_values('WinRate', ascending=False)
            
            # Formatação para string
            diamonds_view = diamonds.copy()
            diamonds_view['WinRate'] = diamonds_view['WinRate'].apply(fmt_pct)
            
            try:
                f.write(tabulate(diamonds_view, headers="keys", tablefmt="pipe", showindex=False))
            except Exception as e:
                f.write(f"Erro ao tabular: {e}\n\n")
                f.write(diamonds_view.to_string())
        else:
            f.write("Nenhum setup atingiu > 60% de winrate ou não houve trades suficientes.\n")
            
    log("Concluído com sucesso.")

if __name__ == "__main__":
    main()
