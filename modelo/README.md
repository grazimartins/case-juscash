# Modelagem de Machine Learning para Previsão de Sucesso de Projetos

Este diretório contém os scripts, modelos, testes automatizados e dados históricos utilizados para o desenvolvimento, validação e entrega da solução de machine learning do case técnico.

## 📁 Estrutura do Diretório

- `train_model.py`: Script principal de treinamento do modelo, engenharia de features, validação cruzada e ajuste de threshold.
- `api_predicao.py`: API FastAPI que expõe o modelo para predição.
- `tests/`: Testes unitários para o pipeline de modelagem e testes automatizados para a API.
- `data/`: Contém o arquivo `historico_projetos.csv` usado no treinamento e nas predições.
- `artifacts/` (separado ou implícito): Contém artefatos gerados, como o modelo `.pkl`, imagens SHAP etc.
- `notebooks/`: (opcional) Notebooks utilizados para exploração, validação e prototipagem.



## 🚀 Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Treine o modelo:
```bash
python train_model.py
```

3. (Opcional) Rode os testes:
```bash
pytest -v
```

4. Execute a API para predições:
```bash
uvicorn api_predicao:app --reload
```

## Testes Automatizados

Estão incluídos testes para:
- Carregamento do modelo e estrutura do pipeline
- Predição correta com threshold
- Cobertura da API FastAPI para `/predict` com payloads válidos e inválidos

Cobertura de testes via `pytest-cov` disponível com:
```bash
pytest --cov=modelo --cov-report=term --cov-report=html
```

## Escolha e Desempenho do Modelo

Foram avaliados Regressão Logística, Random Forest, XGBoost e CatBoost. O modelo final escolhido foi:

- **Modelo:** Random Forest Classifier
- **F1-Score (validação cruzada):** 0.9046
- **F1-Score (teste):** 0.7315
- **Recall:** 0.779
- **Threshold ajustado:** 0.3000
- **F1 ajustado real:** 0.8097

O Random Forest foi selecionado por seu desempenho consistente, interpretabilidade via importância de features e compatibilidade com produção e auditoria.

A interpretabilidade foi complementada com gráficos SHAP salvos automaticamente durante o treinamento (`shap_summary_bar.png`, `shap_summary_beeswarm.png`).

---

Para mais detalhes, consulte os notebooks ou os testes incluídos neste diretório.
