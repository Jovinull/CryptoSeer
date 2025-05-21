# Cryptoseer

Previsão de preços de criptomoedas utilizando redes neurais recorrentes (LSTM, GRU, Dense), com indicadores técnicos e tuning automático de hiperparâmetros.

## Visão Geral

Este projeto tem como objetivo prever o preço futuro de criptomoedas usando dados históricos e técnicas modernas de aprendizado profundo. Ele utiliza uma pipeline robusta que abrange desde coleta e pré-processamento de dados até visualização, avaliação e previsão recursiva.

## Tecnologias

* Python 3.12.6
* TensorFlow & Keras
* Scikit-learn
* yFinance
* Matplotlib / Pandas
* Modelos: LSTM, GRU, Dense
* Hiperparâmetros otimizados automaticamente via tuning

---

## Estrutura do Projeto

```
├── config.py
├── data/
│   ├── loader.py
│   └── preprocessing.py
├── model/
│   ├── architecture.py
│   └── train.py
├── utils/
│   ├── metrics.py
│   └── visualization.py
├── tuning.py
├── requirements.txt
├── main.py
└── results/ (gerado automaticamente)
```

---

## Configuração

1. Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

2. Certifique-se de que está usando **Python 3.12.6**.

---

## Execução

Para rodar a previsão principal:

```bash
python main.py
```

O script irá:

* Baixar dados de uma criptomoeda (ex: DOGE-USD)
* Pré-processar com indicadores técnicos (RSI, MACD, Bandas de Bollinger, etc.)
* Treinar ou carregar modelo com base nos melhores hiperparâmetros encontrados
* Fazer previsões, avaliar e plotar resultados
* Executar previsão futura e recursiva
* Comparar com baseline ingênuo (shift de 1 dia)

---

## Tuning de Hiperparâmetros

Para rodar o tuning (demorado):

```bash
python tuning.py
```

O tuning testa combinações de:

* Tipo de modelo (LSTM, GRU, Dense)
* Número de camadas, unidades, dropout
* Otimizadores e learning rates
* Janela de entrada (`prediction_days`) e horizonte futuro (`future_day`)
* Tipo de normalização (`MinMax` ou `Standard`)

Os resultados são salvos automaticamente em `results/tuning_results_*.csv`.

---

## Visualizações

* **plot\_predictions**: gráfico real vs. previsto
* **plot\_residuals**: resíduos da previsão
* **Previsão recursiva**: extensão da previsão para múltiplos dias no futuro

---

## Métricas de Avaliação

São usadas:

* **MSE (Erro Quadrático Médio)**
* **MAE (Erro Absoluto Médio)**
* **R² (Coeficiente de Determinação)**

---

## Observações Técnicas

* Cache de dados com persistência em disco (`data_cache/`)
* Uso de `MinMaxScaler` ou `StandardScaler`
* Suporte a modelos bidirecionais e otimização com Nadam, Adam ou RMSprop
* Detecção e uso automático do melhor `tuning` com base em `score_final`

---

## Parâmetros Padrão

Configurados em `config.py`:

* Criptomoeda: `DOGE-USD`
* Período de treino: desde `2020-01-01`
* Testes a partir de `2022-01-01`
* Modelo salvo: `model_DOGE.h5`

---

## Exemplo de Saída

```
Avaliação do modelo:
 - MSE: 0.0021
 - MAE: 0.0365
 - R²: 0.8942

Previsão de preço para os próximos 30 dias: $0.0884

Previsão recursiva para os próximos 15 dias:
Dia +1: $0.0881
Dia +2: $0.0887
...
```

---

## Licença

Este projeto é livre para uso educacional e pesquisa. Para fins comerciais, entre em contato.

