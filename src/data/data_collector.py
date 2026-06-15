import os

import pandas as pd
import requests

# Endpoint da Binance para buscar dados de candlesticks (klines) 
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def fetch_daily_data(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Busca os dados históricos de um ativo financeiro na Binance.

    Args:
        symbol (str): O símbolo do par de moedas a ser buscado (ex: 'USDTBRL').
        limit (int, optional): O número máximo de dias (candlesticks) a buscar. 
                               O limite máximo permitido pela API é 1000. Padrão é 1000.

    Returns:
        pd.DataFrame: DataFrame contendo as colunas essenciais: 
                      ['date', 'open', 'high', 'low', 'close', 'volume'].
    
    Raises:
        requests.exceptions.RequestException: Caso ocorra uma falha na comunicação HTTP com a API.
    """
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit
    }
    
    response = requests.get(BINANCE_API_URL, params=params)
    response.raise_for_status()
    
    # Converte a resposta da API para o formato JSON (lista de listas)
    raw_data = response.json()
    
    # Nomes das colunas conforme documentação da Binance
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    # Cria o DataFrame com os dados brutos e as colunas apropriadas
    df = pd.DataFrame(raw_data, columns=columns)
    
    # O tempo fornecido pela Binance vem em milissegundos (timestamp). 
    # Aqui é convertido para um formato de data legível (YYYY-MM-DD).
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    
    # Seleciona apenas as colunas essenciais para o projeto
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    
    # Renomeia a coluna 'open_time' para 'date' para facilitar a leitura posterior
    df.rename(columns={"open_time": "date"}, inplace=True)
    
    # Os valores de preço e volume chegam como texto (string). 
    # Aqui é convertido para números decimais (float) para cálculos e gráficos.
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    return df

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Salva o DataFrame processado em formato CSV na pasta raiz de dados brutos.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados do ativo financeiro.
        filename (str): O nome do arquivo com a extensão (ex: 'USDTBRL_daily.csv').
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

if __name__ == "__main__":
    target_symbol = "USDTBRL"
    
    print(f"Fetching historical data for {target_symbol}...")
    
    try:
        historical_df = fetch_daily_data(symbol=target_symbol)
        
        print("\nData Preview:")
        print(historical_df.head())
        
        save_raw_data(historical_df, f"{target_symbol}_daily.csv")
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred while fetching from Binance: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")