# Modelagem de Machine Learning para Previsão de Sucesso de Projetos

Este diretório contém os scripts, modelos e dados históricos utilizados para o desenvolvimento, treinamento e avaliação do modelo de machine learning do case técnico.

## Estrutura do Diretório

- `train_model.py`: Script principal de treinamento do modelo.
- `evaluate_model.py`: Script para avaliação do modelo (opcional).
- `model.pkl`: Modelo treinado salvo (Random Forest).
- `data/`: Base histórica de projetos (`historico_projetos.csv`) utilizada para treinamento e validação.
- `notebooks/`: Notebooks de EDA e modelagem.

## Como Executar

1. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

2. Execute o treinamento do modelo:
```bash
python train_model.py
```

3. (Opcional) Avalie o modelo:
```bash
python evaluate_model.py
```

4. Utilize os notebooks para análise exploratória e experimentação:
  - `notebooks/eda.ipynb`: Análise exploratória dos dados
  - `notebooks/modelagem.ipynb`: Pipeline de modelagem e validação

---
## Escolha e Desempenho do Modelo

Foram avaliados modelos como Regressão Logística, Random Forest, XGBoost e CatBoost. O modelo final escolhido foi o Random Forest, considerando desempenho, robustez e interpretabilidade.

- **F1-Score médio em validação cruzada: 0.9046**
- **F1-Score real no teste: 0.7315**
- **Recall: 0.779**
- **Threshold ajustado automaticamente para 0.3000**, maximizando o F1-score real para **0.8097**

O Random Forest se destacou por sua simplicidade, robustez e melhor explicação das features, sendo adequado para ambientes reais que exigem rastreabilidade e auditabilidade.

A análise de erros confirmou que o modelo aprende padrões relevantes, mesmo em regiões de maior ambiguidade dos dados.
