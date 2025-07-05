
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Caminhos dos arquivos
INPUT_PROJETOS = 'modelo/data/historico_projetos.csv'
MODEL_PATH = 'modelo/model.pkl'


# Buscar modelo salvo
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_PATH}")
model_artifacts = joblib.load(MODEL_PATH)
print(f"Modelo carregado de: {MODEL_PATH}")

model = model_artifacts['model']
scaler = model_artifacts['scaler']
selected_features = model_artifacts['selected_features']
threshold = model_artifacts.get('threshold', 0.5)

# Carregar dados
df = pd.read_csv(INPUT_PROJETOS)

# Garantir que todas as features estejam presentes
for col in selected_features:
    if col not in df.columns:
        df[col] = 0

X = df[selected_features].fillna(0).copy()

# Normalizar variáveis numéricas (apenas as que foram normalizadas no treino)
numeric_cols = [
    'orcamento', 'duracao', 'orcamento_por_membro', 'duracao_por_membro',
    'duracao_x_equipe', 'orcamento_por_recurso', 'recursos_por_membro',
    'duracao_por_recurso', 'orcamento_por_duracao',
    'complexidade', 'nota_cliente', 'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia'
]
for col in numeric_cols:
    if col in X.columns:
        X[col] = X[col].astype(float)
        # Normaliza apenas se o scaler foi treinado com essa coluna
        if hasattr(scaler, 'mean_') and len(scaler.mean_) == len(numeric_cols):
            X.loc[:, numeric_cols] = scaler.transform(X[numeric_cols])
            break

# Target
y = df['sucesso'] if 'sucesso' in df.columns else None

# Predição
if hasattr(model, 'predict_proba'):
    y_scores = model.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
else:
    y_pred = model.predict(X)

# Avaliação
if y is not None:
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y, y_pred))
else:
    print("Coluna 'sucesso' não encontrada. Apenas predições geradas.")

# Salvar predições (opcional)
df['sucesso_predito'] = y_pred
df.to_csv('modelo/data/historico_projetos_com_predicao.csv', index=False)
print("Predições salvas em data/historico_projetos_com_predicao.csv")
