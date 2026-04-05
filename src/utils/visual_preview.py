import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os

def test_high_resolution_image():
    # 1. ESCOLHA A RESOLUÇÃO EXATA AQUI (Pixels)
    LARGURA_PIXELS = 100
    ALTURA_PIXELS = 100
    DPI = 100
    
    csv_path = os.path.join("..", "..", "data", "raw", "USDTBRL_daily.csv")
    
    # Lê os dados
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Pega uma janela qualquer de 30 dias (ex: do dia 50 ao dia 80)
    window_df = df.iloc[50:80]
    
    market_colors = mpf.make_marketcolors(
        up='green', down='red', edge='inherit', wick='inherit', volume='in'
    )
    custom_style = mpf.make_mpf_style(
        marketcolors=market_colors, facecolor='black', edgecolor='black', 
        figcolor='black', gridstyle='', y_on_right=False
    )
    
    final_filename = f"grafico_teste_{LARGURA_PIXELS}x{ALTURA_PIXELS}.png"
    
    print(f"Gerando imagem nativa de {LARGURA_PIXELS}x{ALTURA_PIXELS} pixels...")
    
    # Calcula o tamanho da figura em polegadas (Inches = Pixels / DPI)
    fig_width = LARGURA_PIXELS / DPI
    fig_height = ALTURA_PIXELS / DPI
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=DPI, facecolor='black')
    
    # Cria o eixo ocupando 100% da figura [esquerda, base, largura, altura]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    
    mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
    
    fig.savefig(final_filename, dpi=DPI, facecolor='black')
    plt.close(fig)
    
    print(f"Sucesso! Imagem salva como '{final_filename}'.")

if __name__ == "__main__":
    test_high_resolution_image()