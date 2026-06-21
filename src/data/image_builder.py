import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm

from src.config import CONFIG

def generate_candlestick_images(
    filename: str = "USDTBRL_daily.csv", 
    window_size: int = 30
) -> None:
    """
    Gera imagens de candlestick a partir do CSV de dados brutos utilizando uma janela deslizante.

    Args:
        filename (str): Nome do arquivo CSV bruto.
        window_size (int): Tamanho da janela deslizante em dias.
    """
    csv_path = CONFIG.paths.raw_data_dir / filename
    output_dir = CONFIG.paths.processed_data_dir
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    market_colors = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    custom_style = mpf.make_mpf_style(marketcolors=market_colors, facecolor='black', edgecolor='black', figcolor='black', gridstyle='', y_on_right=False)

    # Converte a resolução em pixels para polegadas (inches), exigência do matplotlib
    dpi = 100
    fig_size = CONFIG.model.image_size / dpi
    
    # O limite superior do loop evita o erro de índice (IndexError) ao buscar o dia alvo
    total_images = len(df) - window_size - 1
    print(f"Total de imagens a gerar: {total_images} | Resolução: {CONFIG.model.image_size}x{CONFIG.model.image_size}")
    
    for i in tqdm(range(total_images)):
        # Fatiamento (slice) do DataFrame para obter os dados da janela temporal atual
        window_df = df.iloc[i : i + window_size]
        
        # O alvo da predição é o dia imediatamente posterior ao fim da janela
        close_day_current = window_df.iloc[-1]['close']
        target_day_close = df.iloc[i + window_size]['close']
        
        label = "up" if target_day_close > close_day_current else "down"
            
        start_date = window_df.index[0].strftime("%Y%m%d")
        end_date = window_df.index[-1].strftime("%Y%m%d")
        
        img_filename = f"{start_date}_to_{end_date}_{label}.png"
        filepath = output_dir / img_filename
        
        # Renderização da imagem sem eixos e margens
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
        print("\nProcesso concluído.")
    except Exception as e:
        print(f"\nErro no processamento: {e}")