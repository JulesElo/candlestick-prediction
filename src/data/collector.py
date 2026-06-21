import pandas as pd
import requests

from src.config import CONFIG

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def fetch_daily_data(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Busca dados históricos de candlesticks diários na API da Binance.

    Args:
        symbol (str): Par de moedas (ex: 'USDTBRL').
        limit (int): Número máximo de registros (candlesticks) a buscar.

    Returns:
        pd.DataFrame: DataFrame com colunas date, open, high, low, close e volume.
    """
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit
    }
    
    response = requests.get(BINANCE_API_URL, params=params, timeout=10)
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
    Salva o DataFrame em formato CSV no diretório configurado.

    Args:
        df (pd.DataFrame): Dados financeiros processados.
        filename (str): Nome do arquivo de saída.
    """
    output_dir = CONFIG.paths.raw_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    df.to_csv(filepath, index=False)
    print(f"Dados gravados em: {filepath}")

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