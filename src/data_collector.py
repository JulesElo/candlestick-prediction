import requests
import pandas as pd
from datetime import datetime
import os

def fetch_daily_data(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Busca os dados históricos de um ativo financeiro na Binance.
    Retorna um DataFrame do Pandas contendo os dados diários.
    """
    # Endpoint oficial da Binance para buscar dados de candlesticks (klines)
    url = "https://api.binance.com/api/v3/klines"
    
    # Parâmetros que enviaremos para a API
    params = {
        "symbol": symbol,
        "interval": "1d", # '1d' significa 1 dia por vela (candlestick)
        "limit": limit    # 1000 é o limite máximo permitido pela Binance por requisição
    }
    
    # Realiza a requisição HTTP GET
    response = requests.get(url, params=params)
    
    # Levanta uma exceção (erro) caso a API não responda com sucesso (código 200)
    response.raise_for_status() 
    
    # Converte a resposta da API para o formato JSON (lista de listas)
    raw_data = response.json()
    
    # Nomes das colunas conforme especificado na documentação da API da Binance
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    # Cria o DataFrame com os dados brutos e as colunas apropriadas
    df = pd.DataFrame(raw_data, columns=columns)
    
    # O tempo fornecido pela Binance vem em milissegundos (timestamp). 
    # Aqui convertemos para um formato de data legível (YYYY-MM-DD).
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    
    # Seleciona apenas as colunas essenciais para o nosso projeto
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    
    # Renomeia a coluna 'open_time' para 'date' para facilitar a leitura posterior
    df.rename(columns={"open_time": "date"}, inplace=True)
    
    # Os valores de preço e volume chegam como texto (string). 
    # Precisamos convertê-los para números decimais (float) para cálculos e gráficos.
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    return df

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Salva o DataFrame em formato CSV na pasta raiz de dados brutos do projeto.
    """
    # Define o caminho da pasta onde o arquivo será salvo
    output_dir = os.path.join("..", "data", "raw")
    
    # Cria a pasta 'data/raw' automaticamente caso ela ainda não exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Junta o caminho da pasta com o nome do arquivo
    filepath = os.path.join(output_dir, filename)
    
    # Salva o arquivo CSV sem a coluna de índice numérico padrão do Pandas
    df.to_csv(filepath, index=False)
    print(f"Data successfully saved to: {filepath}")

# Bloco de execução principal do script
if __name__ == "__main__":
    # Define o par de moedas que queremos buscar (USDT e Real Brasileiro)
    target_symbol = "USDTBRL"
    
    print(f"Fetching historical data for {target_symbol}...")
    
    try:
        # Chama a função para buscar os dados
        historical_df = fetch_daily_data(symbol=target_symbol)
        
        # Imprime as 5 primeiras linhas no terminal para validação visual rápida
        print("\nData Preview:")
        print(historical_df.head())
        
        # Chama a função para salvar os dados no disco
        save_raw_data(historical_df, f"{target_symbol}_daily.csv")
        
    except Exception as e:
        print(f"An error occurred: {e}")