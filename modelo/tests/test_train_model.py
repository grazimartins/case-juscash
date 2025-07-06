# test_train_model.py - Testes automatizados para pipeline de modelagem (com fix de nomes do scaler)

import pytest
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_optimized.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'model_features.txt')

@pytest.fixture(scope="module")
def modelo_carregado():
    assert os.path.exists(MODEL_PATH), "Arquivo do modelo não encontrado."
    model_data = joblib.load(MODEL_PATH)
    return model_data

@pytest.fixture(scope="module")
def dados_exemplo():
    return pd.DataFrame([{
        'orcamento': 80000,
        'duracao': 60,
        'orcamento_por_membro': 10000,
        'duracao_por_membro': 10,
        'duracao_x_equipe': 240,
        'equipe_extrema': 0,
        'orcamento_por_recurso': 25000,
        'recursos_por_membro': 2.5,
        'orcamento_baixo': 0,
        'duracao_curta': 0,
        'duracao_por_recurso': 20,
        'orcamento_por_duracao': 1333.33,
        'complexidade': 3,
        'nota_cliente': 4.2,
        'idade': 35,
        'projetos_concluidos': 8,
        'horas_trabalhadas': 40,
        'nivel_experiencia': 3
    }])

def test_modelo_estrutura(modelo_carregado):
    assert 'model' in modelo_carregado
    assert 'scaler' in modelo_carregado
    assert 'threshold' in modelo_carregado
    assert 'selected_features' in modelo_carregado
    assert isinstance(modelo_carregado['model'], ClassifierMixin)
    assert isinstance(modelo_carregado['scaler'], StandardScaler)
    assert isinstance(modelo_carregado['selected_features'], list)

def test_predicao_simples(modelo_carregado, dados_exemplo):
    modelo = modelo_carregado['model']
    scaler = modelo_carregado['scaler']
    threshold = modelo_carregado['threshold']
    selected_features = modelo_carregado['selected_features']

    # Separar corretamente as features normalizadas e binárias
    numerical_features = scaler.feature_names_in_.tolist()
    binary_features = [f for f in selected_features if f not in numerical_features]

    X_num = dados_exemplo[numerical_features]
    X_bin = dados_exemplo[binary_features].astype(int)

    dados_scaled = scaler.transform(X_num)
    X_input = np.hstack([dados_scaled, X_bin.values])

    prob = modelo.predict_proba(X_input)[0][1]
    pred = int(prob >= threshold)

    assert 0.0 <= prob <= 1.0
    assert pred in [0, 1]
    print(f"Probabilidade prevista: {prob:.4f} | Predição final: {pred}")

def test_features_compatibilidade(modelo_carregado, dados_exemplo):
    selected_features = modelo_carregado['selected_features']
    assert all([col in dados_exemplo.columns for col in selected_features]), "Algumas features esperadas não estão no input."

def test_threshold_valido(modelo_carregado):
    t = modelo_carregado['threshold']
    assert 0.0 <= t <= 1.0, "Threshold fora do intervalo esperado."
