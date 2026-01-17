import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate

# Configuração Simplificada para Teste
BASE_URL = "https://contract.mexc.com/api/v1"
SYMBOL = "BTC_USDT"
TIMEFRAMES = ["1h", "4h", "1d"] # Reduzi para teste rápido
SETUPS = ["PFR", "DL", "8.2", "8.3"]

def main():
    print(">>> INICIANDO SCRIPT DE DIAGNÓSTICO <<<")
    
    # 1. Tenta criar pasta e arquivo vazio IMEDIATAMENTE
    try:
        os.makedirs("results", exist_ok=True)
        with open("results/debug_log.txt", "w") as f:
            f.write(f"Script iniciou em {time.ctime()}\n")
        print("Pasta results criada com sucesso.")
    except Exception as e:
        print(f"ERRO AO CRIAR PASTA: {e}")
        return

    # 2. Teste de Conexão Simples
    print(f"Tentando baixar 100 candles de {SYMBOL}...")
    url = f"{BASE_URL}/contract/kline/{SYMBOL}"
    params = {'interval': 'Min60', 'limit': 100}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {r.status_code}")
        data = r.json()
        
        if not data.get("data"):
            print("ERRO: Resposta da API veio sem dados:", data)
            with open("results/API_ERROR.txt", "w") as f: f.write(str(data))
            return
            
        candles = data["data"]
        print(f"Sucesso! Baixados {len(candles)} candles.")
        
        # Converte para DF só para testar pandas
        df = pd.DataFrame(candles)
        print("DataFrame criado. Colunas:", df.columns.tolist())
        
        # Salva um CSV de amostra
        df.to_csv(f"results/sample_data_{SYMBOL}.csv", index=False)
        print("Arquivo de amostra salvo.")
        
    except Exception as e:
        print(f"ERRO NA REQUISIÇÃO: {e}")
        with open("results/REQ_ERROR.txt", "w") as f: f.write(str(e))
        return

    # 3. Se chegou aqui, o ambiente funciona. 
    # Vamos rodar o backtest simplificado com os dados que baixamos (só 100 candles)
    # só para provar que a lógica não está quebrando.
    
    print("Rodando simulação dummy...")
    # ... (lógica mínima de trade aqui se quiser, mas o foco é ver se salva)
    
    dummy_results = pd.DataFrame({
        'timeframe': ['1h'], 'setup': ['TEST'], 'win_rate': ['100%']
    })
    dummy_results.to_csv("results/baseline_trades_BTC_USDT.csv", index=False)
    
    with open("results/baseline_summary_BTC_USDT.md", "w") as f:
        f.write("# Teste OK\nO ambiente está funcionando.")
        
    print(">>> FIM DO DIAGNÓSTICO. VERIFIQUE OS ARQUIVOS. <<<")

if __name__ == "__main__":
    main()
