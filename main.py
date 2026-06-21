import argparse
from src.data.collector import fetch_daily_data, save_raw_data
from src.data.image_builder import generate_candlestick_images
from src.utils.normalization import calculate_normalization_params
from src.training.trainer import train_walk_forward

def collect_data():
    """Executa o download dos dados da Binance."""
    target_symbol = "USDTBRL"
    print(f"Iniciando coleta para {target_symbol}...")
    df = fetch_daily_data(symbol=target_symbol)
    save_raw_data(df, f"{target_symbol}_daily.csv")

def build_images():
    """Executa a conversão dos dados CSV em imagens de candlestick."""
    print("Iniciando geração de imagens...")
    generate_candlestick_images()

def calculate_norm():
    """Calcula os parâmetros de normalização do dataset."""
    print("Calculando média e desvio padrão do dataset...")
    mean, std = calculate_normalization_params()
    print(f"\nValores para o config.py:\nmean = {mean}\nstd = {std}")

def run_training():
    """Inicia o pipeline de treinamento Walk-Forward."""
    print("Iniciando pipeline de treinamento...")
    train_walk_forward()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Previsão de Candlesticks")
    parser.add_argument(
        "--action", 
        type=str, 
        required=True, 
        choices=["collect", "generate", "normalize", "train", "all"],
        help="Ação a ser executada pelo pipeline."
    )
    
    args = parser.parse_args()

    try:
        if args.action == "collect":
            collect_data()
        elif args.action == "generate":
            build_images()
        elif args.action == "normalize":
            calculate_norm()
        elif args.action == "train":
            run_training()
        elif args.action == "all":
            collect_data()
            build_images()
            run_training()
    except Exception as e:
        print(f"\nErro fatal durante a execução ({args.action}): {e}")