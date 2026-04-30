import os

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from tqdm import tqdm

def create_directories(base_dir: str = os.path.join("..", "images")) -> str:
    """
    Cria a estrutura completa de pastas (Treino e Teste / Up e Down) 
    para armazenar o dataset de imagens geradas.

    Args:
        base_dir (str, optional): O caminho raiz onde a pasta 'images' será criada. 
                                  Padrão é '../images'.

    Returns:
        str: O caminho absoluto ou relativo da pasta base criada.
    """
    folders = [
        os.path.join(base_dir, "train", "up"),
        os.path.join(base_dir, "train", "down"),
        os.path.join(base_dir, "test", "up"),
        os.path.join(base_dir, "test", "down")
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    return base_dir

def generate_candlestick_images(
    csv_path: str, 
    window_size: int = 30, 
    train_split: float = 0.8,
    fig_size: float = 2.24,
    dpi: int = 100
) -> None:
    """
    Lê os dados históricos em CSV e gera gráficos de Candlestick em formato de imagem,
    rotulando-os como 'up' (alta) ou 'down' (baixa) com base no fechamento futuro.

    As imagens são geradas nativamente sem bordas ou eixos, ocupando 100% do canvas,
    otimizadas para leitura por Redes Neurais Convolucionais (CNNs).

    Args:
        csv_path (str): O caminho para o arquivo CSV contendo os dados brutos.
        window_size (int, optional): A quantidade de dias em cada imagem. Padrão é 30.
        train_split (float, optional): A proporção de imagens para treino (0 a 1.0). Padrão é 0.8 (80%).
        fig_size (float, optional): O tamanho da figura em polegadas. Padrão é 2.24.
        dpi (int, optional): A densidade de pixels por polegada. Padrão é 100.
                             (A resolução final é fig_size * dpi. Ex: 2.24 * 100 = 224x224 pixels).
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    base_dir = create_directories()
    
    # Configuração de cores e estilo visual focado apenas nos candles, sem grid
    market_colors = mpf.make_marketcolors(
        up='green', down='red', edge='inherit', wick='inherit', volume='in'
    )
    custom_style = mpf.make_mpf_style(
        marketcolors=market_colors, facecolor='black', edgecolor='black', 
        figcolor='black', gridstyle='', y_on_right=False
    )

    total_images = len(df) - window_size - 1
    split_index = int(total_images * train_split)
    
    resolution = int(fig_size * dpi)
    print(f"Total de imagens a gerar: {total_images} | Resolução Alvo: {resolution}x{resolution} pixels")
    
    for i in tqdm(range(total_images)):
        window_df = df.iloc[i : i + window_size]
        
        close_day_current = window_df.iloc[-1]['close']
        target_day_close = df.iloc[i + window_size]['close']
        
        label = "up" if target_day_close > close_day_current else "down"
        dataset_type = "train" if i < split_index else "test"
            
        start_date = window_df.index[0].strftime("%Y%m%d")
        end_date = window_df.index[-1].strftime("%Y%m%d")
        filename = f"{start_date}_to_{end_date}.png"
        filepath = os.path.join(base_dir, dataset_type, label, filename)
        
        # 1. Cria a figura vazia com o tamanho exato
        fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi, facecolor='black')
        
        # 2. Cria o eixo ocupando toda a figura [esquerda, base, largura, altura] -> 100%
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off() # Remove réguas e números
        
        # 3. Manda o mplfinance desenhar no eixo (ax=ax)
        mpf.plot(window_df, type='candle', style=custom_style, ax=ax)
        
        # 4. Salva a figura (sem o bbox_inches='tight' para não deformar a resolução)
        fig.savefig(filepath, dpi=dpi, facecolor='black')

        # 5. Fecha a figura para liberar a memória RAM do computador
        plt.close(fig)

if __name__ == "__main__":
    raw_data_path = os.path.join("..", "data", "raw", "USDTBRL_daily.csv")
    
    try:
        # A resolução padrão da função já está configurada para gerar 224x224 (2.24 * 100)
        generate_candlestick_images(csv_path=raw_data_path, window_size=30, train_split=0.8, fig_size=1.0, dpi=100)
        print("\nProcesso concluído com sucesso!")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{raw_data_path}' não foi encontrado. Execute o data_collector.py primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")