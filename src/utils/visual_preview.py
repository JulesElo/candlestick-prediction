import os
import sys
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from typing import Optional

# Ajuste no caminho do sistema para importar o config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def generate_preview_image(
    window_size: int = 30,
    dpi: int = 100, 
    csv_filename: str = "USDTBRL_daily.csv"
) -> None:
    """
    Gera uma imagem de pré-visualização de um gráfico de candlestick 
    utilizando a resolução oficial do projeto definida no config.py.

    Args:
        window_size (int, optional): O tamanho da janela em dias. Padrão é 30.
        dpi (int, optional): Densidade de pixels por polegada. Padrão é 100.
        csv_filename (str, optional): Nome do arquivo CSV bruto.
    """
    csv_path = os.path.join(CONFIG.paths.raw_data_dir, csv_filename)

    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except FileNotFoundError:
        print(f"Erro: Arquivo {csv_path} não encontrado. Execute o coletor de dados primeiro.")
        return

    # Pega uma janela aleatória de dias (ex: do dia 50 ao dia 50 + window_size)
    window_df = df.iloc[50:50+window_size]

    market_colors = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    custom_style = mpf.make_mpf_style(marketcolors=market_colors, facecolor='black', edgecolor='black', figcolor='black', gridstyle='', y_on_right=False)
    
    # Busca a resolução oficial do projeto
    resolution = CONFIG.model.image_size
    fig_width = resolution / dpi
    
    # Salva na pasta de experimentos para não misturar com o dataset oficial
    os.makedirs(CONFIG.paths.experiments_dir, exist_ok=True)
    filepath = os.path.join(CONFIG.paths.experiments_dir, f"preview_{resolution}x{resolution}.png")
    
    print(f"Gerando imagem de pré-visualização em {resolution}x{resolution} pixels...")
    
    fig = plt.figure(figsize=(fig_width, fig_width), dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    
    mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
    fig.savefig(filepath, dpi=dpi, facecolor='black')
    plt.close(fig)
    
    print(f"Preview salvo com sucesso em: {filepath}")

if __name__ == "__main__":
    generate_preview_image()