# ⚙️ Modelagem de Machine Learning para Previsão de Sucesso de Projetos

Este diretório contém os scripts, modelos, testes automatizados e dados históricos utilizados para o desenvolvimento, validação e entrega da solução de machine learning proposta no case técnico.

## 🔷 Estrutura do Diretório

- `train_model.py`: Script principal de treinamento, engenharia de features, validação cruzada, interpretabilidade (SHAP) e salvamento do modelo.
- `api_predicao.py`: API FastAPI para realizar predições com o modelo treinado.
- `test_train_model.py`: Testes unitários para o pipeline de machine learning.
- `test_api_predicao.py`: Testes automatizados para a API REST.
- `data/`: Contém a base `historico_projetos.csv` utilizada no treinamento e validação.
- `artifacts/`: Inclui artefatos gerados, como modelo serializado (`.pkl`), imagens SHAP e features salvas.
- `notebooks/`: (opcional) Notebooks utilizados para exploração de dados e prototipação.
- `requirements.txt`: Lista de dependências necessárias para rodar a aplicação.
- `requirements-dev.txt`: Dependências adicionais para testes e desenvolvimento.


## ⚙️ Execução Local

1. Instale as dependências principais:
```bash
pip install -r requirements.txt
```

2. Para desenvolvimento e testes, instale também:
```bash
pip install -r requirements-dev.txt
```

3. Treine o modelo:
```bash
python train_model.py
```

4. Rode os testes:
```bash
pytest -v
```

5. Execute a API:
```bash
uvicorn api_predicao:app --reload
```

## 🔹 Testes Automatizados

Os testes cobrem:
- Estrutura e carregamento do modelo (`train_model.py`)
- Predição com limiar ajustado
- API REST: respostas com payloads válidos e inválidos

Para gerar relatório de cobertura:
```bash
pytest --cov=modelo --cov-report=term --cov-report=html
```

## 🔵 Modelo Final Selecionado

Modelos avaliados: Regressão Logística, Random Forest, XGBoost e CatBoost. O escolhido foi:

- **Modelo:** Random Forest Classifier
- **F1-Score (validação cruzada):** 0.9046
- **F1-Score (teste):** 0.7315
- **Recall:** 0.779
- **Threshold ajustado:** 0.3000
- **F1 com threshold ajustado:** 0.8097

Critérios de escolha: desempenho robusto, interpretabilidade, estabilidade e compatibilidade com produção.

Interpretabilidade via gráficos SHAP:
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`

---

Para detalhes adicionais, consulte os notebooks ou os scripts contidos neste diretório.
