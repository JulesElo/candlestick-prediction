import os
import sys
import pandas as pd
import requests

# Ajuste no caminho do sistema para permitir a importação do config estando dentro da pasta data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def fetch_daily_data(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Busca os dados históricos de um ativo financeiro na API da Binance.

    Args:
        symbol (str): O símbolo do par de moedas a ser buscado (ex: 'USDTBRL').
        limit (int, optional): O número máximo de dias (candlesticks) a buscar. 
                               O limite máximo permitido pela API é 1000. Padrão é 1000.

    Returns:
        pd.DataFrame: DataFrame contendo as colunas essenciais do mercado: 
                      ['date', 'open', 'high', 'low', 'close', 'volume'].
    
    Raises:
        requests.exceptions.RequestException: Caso ocorra uma falha na comunicação HTTP.
    """
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit
    }
    
    response = requests.get(BINANCE_API_URL, params=params)
    response.raise_for_status()
    
    raw_data = response.json()
    
    columns = [
        "date", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    df = pd.DataFrame(raw_data, columns=columns)
    
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    return df

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Salva o DataFrame processado em formato CSV, utilizando o caminho
    centralizado nas configurações do projeto.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados do ativo financeiro.
        filename (str): O nome do arquivo com a extensão (ex: 'USDTBRL_daily.csv').
    """
    # Importa o diretório diretamente do arquivo de configuração
    output_dir = CONFIG.paths.raw_data_dir
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Dados gravados com sucesso em: {filepath}")

if __name__ == "__main__":
    target_symbol = "USDTBRL"
    print(f"Iniciando a busca de dados históricos para {target_symbol}...")
    
    try:
        historical_df = fetch_daily_data(symbol=target_symbol)
        print("\nVisualização dos dados carregados:")
        print(historical_df.head())
        
        save_raw_data(historical_df, f"{target_symbol}_daily.csv")
    except requests.exceptions.RequestException as e:
        print(f"Erro na conexão com a API da Binance: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")