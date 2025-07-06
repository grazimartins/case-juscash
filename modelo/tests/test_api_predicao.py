# test_api_predicao.py - Testes automatizados da API FastAPI

import pytest
from fastapi.testclient import TestClient
from api_predicao import app

client = TestClient(app)

# Dados de exemplo com user_id existente (ajuste se necessário)
exemplo_payload = {
    "user_id": 1,
    "valor_projeto": 80000,
    "duracao_dias": 60,
    "complexidade": 3,
    "nota_cliente": 4.2
}

def test_predict_status_code():
    response = client.post("/predict", json=exemplo_payload)
    assert response.status_code == 200

def test_predict_estrutura_retorno():
    response = client.post("/predict", json=exemplo_payload)
    json_data = response.json()
    assert "probabilidade" in json_data
    assert "predicao" in json_data
    assert "threshold_usado" in json_data
    assert isinstance(json_data["probabilidade"], float)
    assert json_data["predicao"] in [0, 1]

# Teste com usuário que não existe na base
invalido_payload = exemplo_payload.copy()
invalido_payload["user_id"] = 99999999

def test_usuario_nao_encontrado():
    response = client.post("/predict", json=invalido_payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Histórico do usuário não encontrado"

# Teste com payload incompleto
def test_payload_invalido():
    response = client.post("/predict", json={"user_id": 1})
    assert response.status_code == 422  # erro de validação do FastAPI