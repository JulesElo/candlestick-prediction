import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm

# Ajuste no caminho do sistema para permitir a importação do config estando dentro da pasta data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def create_output_directory() -> str:
    """
    Cria o diretório de saída para as imagens processadas, 
    utilizando o caminho centralizado no config.
    """
    output_dir = CONFIG.paths.processed_data_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_candlestick_images(
    filename: str = "USDTBRL_daily.csv", 
    window_size: int = 30
) -> None:
    """
    Gera as imagens de candlestick a partir do CSV de dados brutos.
    As imagens são salvas em um único diretório com o rótulo da classe 
    embutido no nome do arquivo para garantir a ordenação cronológica.

    Args:
        filename (str, optional): O nome do arquivo CSV bruto. Padrão é "USDTBRL_daily.csv".
        window_size (int, optional): O tamanho da janela deslizante em dias. Padrão é 30.
        
    Raises:
        FileNotFoundError: Caso o arquivo CSV não exista na pasta raw.
    """
    csv_path = os.path.join(CONFIG.paths.raw_data_dir, filename)
    output_dir = create_output_directory()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo de dados brutos não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Configurações de estilo do gráfico financeiro (Fundo preto, candles verdes e vermelhos)
    market_colors = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    custom_style = mpf.make_mpf_style(marketcolors=market_colors, facecolor='black', edgecolor='black', figcolor='black', gridstyle='', y_on_right=False)

    # Lógica dinâmica para converter tamanho de imagem para DPI do Matplotlib
    dpi = 100
    fig_size = CONFIG.model.image_size / dpi
    
    total_images = len(df) - window_size - 1
    print(f"Total de imagens a gerar: {total_images} | Resolução Alvo: {CONFIG.model.image_size}x{CONFIG.model.image_size} pixels")
    
    for i in tqdm(range(total_images)):
        window_df = df.iloc[i : i + window_size]
        close_day_current = window_df.iloc[-1]['close']
        target_day_close = df.iloc[i + window_size]['close']
        
        label = "up" if target_day_close > close_day_current else "down"
            
        start_date = window_df.index[0].strftime("%Y%m%d")
        end_date = window_df.index[-1].strftime("%Y%m%d")
        
        # Rótulo acoplado ao nome para o TimeSeriesSplit funcionar sem quebrar a ordem
        img_filename = f"{start_date}_to_{end_date}_{label}.png"
        filepath = os.path.join(output_dir, img_filename)
        
        fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi, facecolor='black')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        
        mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
        fig.savefig(filepath, dpi=dpi, facecolor='black')
        plt.close(fig)

if __name__ == "__main__":
    print("Iniciando conversão de séries temporais para imagens...")
    try:
        generate_candlestick_images()
        print("\nProcesso de geração de imagens concluído com sucesso!")
    except Exception as e:
        print(f"\nOcorreu um erro durante o processamento: {e}")