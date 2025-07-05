
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os


class ProjetoPayload(BaseModel):
    user_id: int
    valor_projeto: float
    duracao_dias: float
    complexidade: float
    nota_cliente: float

app = FastAPI()

# Features esperadas pelo modelo treinado
MODEL_FEATURES = [
    'media_sucesso', 'projetos_total', 'media_valor', 'media_duracao',
    'media_complexidade', 'media_nota_cliente',
    'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia',
    'valor_projeto', 'duracao_dias', 'complexidade', 'nota_cliente'
]

def carregar_modelo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_optimized.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def calcular_historico_usuario(user_id, historico):
    # Calcula agregados do histórico do usuário
    user_hist = historico[historico['user_id'] == user_id]
    if user_hist.empty:
        return None
    return {
        'media_sucesso': user_hist['sucesso'].mean(),
        'projetos_total': user_hist['sucesso'].count(),
        'media_valor': user_hist['valor_projeto'].mean(),
        'media_duracao': user_hist['duracao_dias'].mean(),
        'media_complexidade': user_hist['complexidade'].mean(),
        'media_nota_cliente': user_hist['nota_cliente'].mean(),
        'idade': user_hist['idade'].iloc[-1],
        'projetos_concluidos': user_hist['projetos_concluidos'].iloc[-1],
        'horas_trabalhadas': user_hist['horas_trabalhadas'].iloc[-1],
        'nivel_experiencia': user_hist['nivel_experiencia'].iloc[-1],
    }

@app.post("/predict")
def predict(payload: ProjetoPayload):
    payload = payload.dict()
    try:
        # Carregar modelo
        model = carregar_modelo()
        # Carregar histórico de projetos
        base_dir = os.path.dirname(os.path.abspath(__file__))
        historico_path = os.path.join(base_dir, 'data', 'historico_projetos.csv')
        import pandas as pd
        historico = pd.read_csv(historico_path)

        # Corrigir nomes de colunas para compatibilidade
        historico = historico.rename(columns={
            'id_usuario': 'user_id',
            'orcamento': 'valor_projeto',
            'duracao': 'duracao_dias',
        })
        # Adicionar colunas fictícias se necessário
        for col in ['complexidade', 'nota_cliente', 'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia']:
            if col not in historico.columns:
                if col == 'complexidade':
                    historico[col] = np.random.randint(1, 6, size=len(historico))
                elif col == 'nota_cliente':
                    historico[col] = np.random.uniform(1, 5, size=len(historico))
                elif col == 'idade':
                    historico[col] = np.random.randint(20, 60, size=len(historico))
                elif col == 'projetos_concluidos':
                    historico[col] = np.random.randint(1, 20, size=len(historico))
                elif col == 'horas_trabalhadas':
                    historico[col] = np.random.randint(10, 60, size=len(historico))
                elif col == 'nivel_experiencia':
                    historico[col] = np.random.randint(1, 6, size=len(historico))

        user_id = payload.get('user_id')
        if user_id is None:
            raise HTTPException(status_code=400, detail='user_id é obrigatório')
        hist = calcular_historico_usuario(user_id, historico)
        if hist is None:
            raise HTTPException(status_code=404, detail='Histórico do usuário não encontrado')

        # Dados do novo projeto
        novo = {
            'valor_projeto': float(payload.get('valor_projeto', 0)),
            'duracao_dias': float(payload.get('duracao_dias', 0)),
            'complexidade': float(payload.get('complexidade', 0)),
            'nota_cliente': float(payload.get('nota_cliente', 0)),
        }

        # Montar vetor de entrada
        features = {**hist, **novo}
        X = np.array([[features[f] for f in MODEL_FEATURES]])
        prob = model.predict_proba(X)[0][1]
        return {"probabilidade": float(prob)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
