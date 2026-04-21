import pandas as pd
import os
import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_directories():
    """
    Cria a estrutura completa de pastas para treino e teste.
    """
    base_dir = os.path.join("..", "images")
    folders = [
        os.path.join(base_dir, "train", "up"),
        os.path.join(base_dir, "train", "down"),
        os.path.join(base_dir, "test", "up"),
        os.path.join(base_dir, "test", "down")
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return base_dir

def generate_candlestick_images(csv_path: str, window_size: int = 30, train_split: float = 0.8):
    """
    Gera imagens nativas do matplotlib ocupando 100% da tela em 224x224 pixels.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    base_dir = create_directories()
    
    market_colors = mpf.make_marketcolors(
        up='green', down='red', edge='inherit', wick='inherit', volume='in'
    )
    custom_style = mpf.make_mpf_style(
        marketcolors=market_colors, facecolor='black', edgecolor='black', 
        figcolor='black', gridstyle='', y_on_right=False
    )

    total_images = len(df) - window_size - 1
    split_index = int(total_images * train_split)
    
    # Define a resolução (2.24 inches * 100 dpi = 224 pixels)
    FIG_SIZE = 0.5
    DPI = 100
    
    print(f"Total de imagens: {total_images} | Target: 224x224 pixels (Nativo)")
    
    for i in tqdm(range(total_images)):
        window_df = df.iloc[i : i + window_size]
        
        close_day_30 = window_df.iloc[-1]['close']
        target_day_close = df.iloc[i + window_size]['close']
        
        label = "up" if target_day_close > close_day_30 else "down"
        dataset_type = "train" if i < split_index else "test"
            
        start_date = window_df.index[0].strftime("%Y%m%d")
        end_date = window_df.index[-1].strftime("%Y%m%d")
        filename = f"{start_date}_to_{end_date}.png"
        filepath = os.path.join(base_dir, dataset_type, label, filename)
        
        # 1. Cria a figura vazia com o tamanho exato
        fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI, facecolor='black')
        
        # 2. Cria o eixo ocupando toda a figura [esquerda, base, largura, altura] -> 100%
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off() # Remove réguas e números
        
        # 3. Manda o mplfinance desenhar no nosso eixo (ax=ax)
        mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
        
        # 4. Salva a figura (sem o bbox_inches='tight' para não deformar a resolução)
        fig.savefig(filepath, dpi=DPI, facecolor='black')
        
        # 5. Fecha a figura para liberar a memória RAM do computador
        plt.close(fig)

if __name__ == "__main__":
    raw_data_path = os.path.join("..", "data", "raw", "USDTBRL_daily.csv")
    
    try:
        generate_candlestick_images(raw_data_path, window_size=20, train_split=0.8)
        print("\nProcesso concluído com sucesso! Todas as imagens estão nativamente em 224x224.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")