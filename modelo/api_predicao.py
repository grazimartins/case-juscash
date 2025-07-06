from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

class ProjetoPayload(BaseModel):
    user_id: int
    valor_projeto: float
    duracao_dias: float
    complexidade: float
    nota_cliente: float

app = FastAPI()

def carregar_modelo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "artifacts/model.pkl")
    model_data = joblib.load(model_path)
    return model_data

def calcular_historico_usuario(user_id, historico):
    user_hist = historico[historico['user_id'] == user_id]
    if user_hist.empty:
        return None
    return {
        'orcamento': user_hist['valor_projeto'].mean(),
        'duracao': user_hist['duracao_dias'].mean(),
        'orcamento_por_membro': user_hist['orcamento_por_membro'].mean(),
        'duracao_por_membro': user_hist['duracao_por_membro'].mean(),
        'duracao_x_equipe': user_hist['duracao_x_equipe'].mean(),
        'orcamento_por_recurso': user_hist['orcamento_por_recurso'].mean(),
        'recursos_por_membro': user_hist['recursos_por_membro'].mean(),
        'duracao_por_recurso': user_hist['duracao_por_recurso'].mean(),
        'orcamento_por_duracao': user_hist['orcamento_por_duracao'].mean(),
        'complexidade': user_hist['complexidade'].mean(),
        'nota_cliente': user_hist['nota_cliente'].mean(),
        'idade': user_hist['idade'].iloc[-1],
        'projetos_concluidos': user_hist['projetos_concluidos'].iloc[-1],
        'horas_trabalhadas': user_hist['horas_trabalhadas'].iloc[-1],
        'nivel_experiencia': user_hist['nivel_experiencia'].iloc[-1],
        # Calcular as features binárias
        'duracao_curta': int(user_hist['duracao_dias'].mean() < 30),
        'orcamento_baixo': int(user_hist['valor_projeto'].mean() < 50000),  # ajuste o valor conforme seu contexto
        'equipe_extrema': int(user_hist['duracao_x_equipe'].mean() > 100)  # ajuste também conforme contexto
    }

@app.post("/predict")
def predict(payload: ProjetoPayload):
    try:
        model_data = carregar_modelo()
        model = model_data['model']
        scaler = model_data['scaler']
        selected_features = model_data['selected_features']
        threshold = model_data.get('threshold', 0.5)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        historico_path = os.path.join(base_dir, 'data', 'historico_projetos.csv')
        historico = pd.read_csv(historico_path)

        # Renomear colunas do histórico para manter consistência
        historico = historico.rename(columns={
            'id_usuario': 'user_id',
            'orcamento': 'valor_projeto',
            'duracao': 'duracao_dias',
        })

        # Preencher colunas que podem faltar com valores default (exemplo aleatório ou zero)
        for col in ['orcamento_por_membro', 'duracao_por_membro', 'duracao_x_equipe', 
                    'orcamento_por_recurso', 'recursos_por_membro', 'duracao_por_recurso', 
                    'orcamento_por_duracao', 'complexidade', 'nota_cliente', 'idade', 
                    'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia']:
            if col not in historico.columns:
                historico[col] = 0

        user_id = payload.user_id
        hist = calcular_historico_usuario(user_id, historico)
        if hist is None:
            raise HTTPException(status_code=404, detail='Histórico do usuário não encontrado')

        # Dados do novo projeto
        novo = {
            'orcamento': payload.valor_projeto,
            'duracao': payload.duracao_dias,
            'complexidade': payload.complexidade,
            'nota_cliente': payload.nota_cliente,
            # Vamos precisar dessas binárias para o novo projeto
            'duracao_curta': int(payload.duracao_dias < 30),
            'orcamento_baixo': int(payload.valor_projeto < 50000),  # ajuste o valor conforme contexto
            'equipe_extrema': 0,  # aqui você pode definir uma regra ou manter 0 se não aplicável no novo projeto
        }

        # Juntar histórico com dados do novo projeto, preferindo dados do novo projeto para as features correspondentes
        features = {**hist, **novo}

        # Garantir que todas as features que o modelo espera estejam no dict
        for col in selected_features:
            if col not in features:
                features[col] = 0

        # Criar DataFrame com a ordem certa das colunas
        X_df = pd.DataFrame([features], columns=selected_features)

        # Separar features numéricas e binárias conforme treinamento
        numerical_features = [
            'orcamento', 'duracao', 'orcamento_por_membro', 'duracao_por_membro',
            'duracao_x_equipe', 'orcamento_por_recurso', 'recursos_por_membro',
            'duracao_por_recurso', 'orcamento_por_duracao', 'complexidade',
            'nota_cliente', 'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia'
        ]
        binary_features = ['duracao_curta', 'orcamento_baixo', 'equipe_extrema']

        # Aplicar scaler só nas numéricas
        X_num = X_df[numerical_features].astype(float)
        X_bin = X_df[binary_features].astype(int)

        X_num_scaled = scaler.transform(X_num)

        # Reunir arrays para formar o input final
        X_final = np.hstack([X_num_scaled, X_bin.values])
        X_final_df = pd.DataFrame(X_final, columns=numerical_features + binary_features)

        # Garantir ordem conforme o esperado pelo modelo
        X_final_df = X_final_df[selected_features]

        prob = model.predict_proba(X_final_df)[0][1]
        pred = int(prob >= threshold)

        return {
            "probabilidade": float(prob),
            "predicao": pred,
            "threshold_usado": threshold
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
