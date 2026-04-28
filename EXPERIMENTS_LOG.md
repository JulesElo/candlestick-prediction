# Log de Experimentos - Predição de Ativos (USDT/BRL)

| ID | Data | Janela | Imagem | Épocas | Batch | Acurácia Treino | Acurácia Teste | Observações |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | 21/04/2025 | 20 dias | 50x50 | 10 | 32 | 52.87% | 45.92% | Baseline do artigo. Treino muito rápido (CPU), mas o modelo não convergiu (Loss estagnado em 0.69). Acurácia de teste indica chute aleatório. O volume de 1.000 dias do USDT/BRL parece insuficiente para extrair padrões nesta resolução. |
| **EXP-02** | 21/04/2025 | 20 dias | 50x50 | 50 | 32 | 81.48% | 55.10% | O modelo convergiu no treino a partir da época 19, reduzindo o Loss para 0.41. Contudo, a diferença entre Treino (81%) e Teste (55%) indica forte Overfitting. O modelo memorizou os dados, mas teve dificuldade de generalizar. |
| **EXP-03** | 21/04/2025 | 20 dias | 50x50 | 50 | 32 | 58.62% | 46.94% | Adição de LR Decay (Step=10, Gamma=0.5). O freio prematuro na taxa de aprendizado impediu a convergência. O modelo "congelou" o aprendizado antes de achar o padrão, piorando consideravelmente em relação ao EXP-02. |