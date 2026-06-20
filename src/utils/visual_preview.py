import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

# Adiciona a raiz do projeto ao sys.path para importação do módulo src
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG

def generate_preview_image(
    window_size: int = 30,
    dpi: int = 100, 
    csv_filename: str = "USDTBRL_daily.csv"
) -> None:
    """
    Gera uma imagem de pré-visualização de um gráfico de candlestick.

    Args:
        window_size (int): Tamanho da janela em dias.
        dpi (int): Densidade de pixels por polegada.
        csv_filename (str): Nome do arquivo CSV de entrada.
    """
    csv_path = CONFIG.paths.raw_data_dir / csv_filename

    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except FileNotFoundError:
        print(f"Erro: Arquivo {csv_path} não encontrado.")
        return

    if len(df) < 50 + window_size:
        print("Erro: Dados insuficientes para gerar a pré-visualização.")
        return

    window_df = df.iloc[50:50+window_size]

    market_colors = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    custom_style = mpf.make_mpf_style(marketcolors=market_colors, facecolor='black', edgecolor='black', figcolor='black', gridstyle='', y_on_right=False)
    
    resolution = CONFIG.model.image_size
    fig_width = resolution / dpi
    
    output_dir = CONFIG.paths.experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"preview_{resolution}x{resolution}.png"
    
    fig = plt.figure(figsize=(fig_width, fig_width), dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    
    mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
    fig.savefig(filepath, dpi=dpi, facecolor='black')
    plt.close(fig)
    
    print(f"Preview salvo em: {filepath}")

if __name__ == "__main__":
    generate_preview_image()