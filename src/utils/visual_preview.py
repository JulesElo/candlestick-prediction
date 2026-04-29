import os
from typing import Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def generate_preview_image(
    width_pixels: int = 100, 
    height_pixels: int = 100, 
    dpi: int = 100, 
    csv_path: Optional[str] = None
) -> None:
    """
    Gera uma imagem de pré-visualização de um gráfico de candlestick 
    para validar a resolução e o formato visual antes de gerar o dataset completo.

    Args:
        width_pixels (int, optional): A largura exata da imagem em pixels. Padrão é 100.
        height_pixels (int, optional): A altura exata da imagem em pixels. Padrão é 100.
        dpi (int, optional): Densidade de pixels por polegada (Dots Per Inch). Padrão é 100.
        csv_path (str, optional): Caminho para o arquivo CSV. Se None, tenta buscar no caminho padrão.
    """
    if csv_path is None:
        csv_path = os.path.join("..", "data", "raw", "USDTBRL_daily.csv")

    try:
        # Lê os dados
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_path}' não foi encontrado. Execute o data_collector.py primeiro.")
        return
    
    # Pega uma janela qualquer de 30 dias (ex: do dia 50 ao dia 80)
    window_df = df.iloc[50:80]

    # Configuração visual nativa do mplfinance
    market_colors = mpf.make_marketcolors(
        up='green', down='red', edge='inherit', wick='inherit', volume='in'
    )
    custom_style = mpf.make_mpf_style(
        marketcolors=market_colors, facecolor='black', edgecolor='black', 
        figcolor='black', gridstyle='', y_on_right=False
    )
    
    final_filename = f"grafico_preview_{width_pixels}x{height_pixels}.png"
    print(f"Gerando imagem nativa de {width_pixels}x{height_pixels} pixels...")
    
    # Calcula o tamanho da figura em polegadas (Inches = Pixels / DPI)
    fig_width = width_pixels / dpi
    fig_height = height_pixels / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black')
    
    # Cria o eixo ocupando 100% da figura [esquerda, base, largura, altura]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    
    mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
    
    fig.savefig(final_filename, dpi=dpi, facecolor='black')
    plt.close(fig)
    
    print(f"Sucesso! Imagem salva como '{final_filename}'.")

if __name__ == "__main__":
    # ==========================================
    # PAINEL DE TESTE RÁPIDO
    # ==========================================
    TEST_WIDTH = 100
    TEST_HEIGHT = 100
    
    generate_preview_image(width_pixels=TEST_WIDTH, height_pixels=TEST_HEIGHT)