import os
import itertools
import pandas as pd
import numpy as np
from tabulate import tabulate

# Configurações
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
INPUT_FILE = f"results/backtest_trades_{SYMBOL}.csv"
OUTPUT_FILE = f"results/pairwise_analysis_{SYMBOL}.csv"

# Parâmetros da Análise
N_BINS = 4          # 0 a 3 (Quartis: 0=Baixo, 3=Alto)
MIN_TRADES = 30     # Mínimo de trades para validar o padrão
MIN_WINRATE = 0.60  # Só mostra se acertar mais de 60%

def fmt_pct(val):
    try:
        return f"{val*100:.2f}%".replace(".", ",")
    except: return "-"

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Arquivo {INPUT_FILE} não encontrado.")
        return pd.DataFrame()
    return pd.read_csv(INPUT_FILE)

def get_numeric_features(df):
    # Lista de colunas que não são features numéricas
    ignore = [
        'timeframe', 'setup', 'side', 'signal_time', 'entry_time', 
        'entry_price', 'stop_price', 'fill_delay'
    ]
    # Identifica colunas de RR (resultado)
    rr_cols = [c for c in df.columns if c.startswith('rr_')]
    
    # O resto são features
    features = [c for c in df.columns if c not in ignore and c not in rr_cols]
    
    # Garante que são numéricas
    numeric_feats = []
    for f in features:
        try:
            pd.to_numeric(df[f], errors='raise')
            if df[f].nunique() > 1: # Ignora constantes
                numeric_feats.append(f)
        except: pass
        
    return numeric_feats, rr_cols

def analyze_pairwise():
    df = load_data()
    if df.empty: return

    features, rr_cols = get_numeric_features(df)
    results = []

    print(f"Analisando {len(features)} variáveis e seus cruzamentos...")

    # Loop por Timeframe e Setup (Para não misturar bananas com laranjas)
    for (tf, setup), group in df.groupby(['timeframe', 'setup']):
        print(f"Processando {tf} {setup}...")
        
        # 1. Pré-calcular Bins (Quartis) para esse grupo
        bins_df = pd.DataFrame(index=group.index)
        for feat in features:
            try:
                # Cria bins 0, 1, 2, 3 (qcut tenta dividir em quantidades iguais)
                bins_df[feat] = pd.qcut(group[feat], N_BINS, labels=False, duplicates='drop')
            except:
                continue # Pula se não der para binarizar (ex: muitos zeros)

        valid_feats = bins_df.columns.tolist()

        # 2. Loop por RR (Resultado)
        for rr_col in rr_cols:
            rr_val = rr_col.replace('rr_', '')
            
            # Pega apenas trades com resultado definido
            valid_trades = group[group[rr_col].isin(['win', 'loss'])].copy()
            if valid_trades.empty: continue
            
            # Adiciona info de win/loss
            # 1 = Win, 0 = Loss
            valid_trades['is_win'] = (valid_trades[rr_col] == 'win').astype(int)

            # 3. Cruzar Feature A x Feature B
            # itertools.combinations gera todos os pares possíveis sem repetição
            for feat_a, feat_b in itertools.combinations(valid_feats, 2):
                
                # Junta os bins com o resultado
                temp = valid_trades[['is_win']].copy()
                temp['bin_a'] = bins_df.loc[valid_trades.index, feat_a]
                temp['bin_b'] = bins_df.loc[valid_trades.index, feat_b]
                
                # Agrupa por Par de Bins
                stats = temp.groupby(['bin_a', 'bin_b'])['is_win'].agg(['count', 'sum'])
                stats.columns = ['trades', 'wins']
                
                # Filtra os resultados bons
                for (bin_a, bin_b), row in stats.iterrows():
                    trades_count = row['trades']
                    if trades_count < MIN_TRADES: continue
                    
                    wins = row['wins']
                    losses = trades_count - wins
                    win_rate = wins / trades_count
                    
                    if win_rate >= MIN_WINRATE:
                        results.append({
                            'timeframe': tf,
                            'setup': setup,
                            'rr': rr_val,
                            'feature_a': feat_a,
                            'feature_b': feat_b,
                            'bin_a': int(bin_a),
                            'bin_b': int(bin_b),
                            'trades': int(trades_count),
                            'wins': int(wins),
                            'losses': int(losses),
                            'win_rate': win_rate,
                            'win_rate_pct': fmt_pct(win_rate)
                        })

    # Salvar
    if results:
        final_df = pd.DataFrame(results)
        # Ordenar pelos melhores
        final_df = final_df.sort_values(['win_rate', 'trades'], ascending=[False, False])
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAnálise concluída! Salvo em: {OUTPUT_FILE}")
        
        # Mostra prévia
        print(tabulate(final_df.head(20), headers="keys", tablefmt="simple", showindex=False))
    else:
        print("Nenhum padrão encontrado com os filtros atuais.")

if __name__ == "__main__":
    analyze_pairwise()
