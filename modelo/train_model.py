# train_model.py - Versao final com engenharia completa de features, selecao de modelos e interpretabilidade com SHAP

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTEENN
import shap
import joblib
import matplotlib.pyplot as plt

# Caminhos
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'historico_projetos.csv'
MODEL_PATH = BASE_DIR / 'model_optimized.pkl'
RANDOM_STATE = 42

# Carregamento
print(f"Lendo base: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Engenharia de features
if 'recursos_disponiveis' in df.columns:
    df = pd.get_dummies(df, columns=['recursos_disponiveis'], prefix='recursos')

df['numero_membros_equipe'] = df.get('numero_membros_equipe', 8)
df['orcamento_por_membro'] = df['orcamento'] / df['numero_membros_equipe']
df['duracao_por_membro'] = df['duracao'] / df['numero_membros_equipe']
df['duracao_x_equipe'] = df['duracao'] * df['numero_membros_equipe']
df['equipe_extrema'] = df['numero_membros_equipe'].apply(lambda x: 1 if x <= 2 or x >= 12 else 0)
df['orcamento_por_recurso'] = df['orcamento'] / (df.get('recursos_Baixo',0) + 2*df.get('recursos_Médio',0) + 3*df.get('recursos_Alto',0) + 1e-10)
df['recursos_por_membro'] = (df.get('recursos_Baixo',0) + 2*df.get('recursos_Médio',0) + 3*df.get('recursos_Alto',0)) / df['numero_membros_equipe']
df['orcamento_baixo'] = (df['orcamento'] <= df['orcamento'].quantile(0.25)).astype(int)
df['duracao_curta'] = (df['duracao'] <= df['duracao'].quantile(0.25)).astype(int)
df['duracao_por_recurso'] = df['duracao'] / (df.get('recursos_Baixo',0) + 2*df.get('recursos_Médio',0) + 3*df.get('recursos_Alto',0) + 1e-10)
df['orcamento_por_duracao'] = df['orcamento'] / (df['duracao'] + 1e-10)

# Features finais
features = [
    'orcamento', 'duracao', 'orcamento_por_membro', 'duracao_por_membro', 'duracao_x_equipe',
    'equipe_extrema', 'orcamento_por_recurso', 'recursos_por_membro', 'orcamento_baixo', 'duracao_curta',
    'duracao_por_recurso', 'orcamento_por_duracao',
    'complexidade', 'nota_cliente', 'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia'
]
X = df[features].fillna(0).astype(float)
y = df['sucesso']

# Normalização
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[X.columns] = scaler.fit_transform(X[X.columns])

# Split e balanceamento
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
smote_enn = SMOTEENN(sampling_strategy=0.7, random_state=RANDOM_STATE)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# Modelos e grids
modelos = {
    'Regressão Logística': LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
}
param_grids = {
    'Regressão Logística': [
        {'C': [10, 50, 100], 'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
        {'C': [10, 50, 100], 'solver': ['saga'], 'penalty': ['l1', 'l2']},
        {'C': [10, 50, 100], 'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [0.1, 0.5]}
    ],
    'Random Forest': {
        'n_estimators': [200, 300], 'max_depth': [10, 20], 'min_samples_split': [2],
        'min_samples_leaf': [2], 'max_features': ['sqrt']
    },
    'XGBoost': {
        'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [1, 2], 'subsample': [0.9], 'colsample_bytree': [0.9]
    },
    'CatBoost': {
        'iterations': [100, 200], 'depth': [6], 'l2_leaf_reg': [3], 'learning_rate': [0.05]
    }
}

melhor_modelo, melhor_nome, melhor_f1, melhor_threshold = None, '', 0, 0.3

for nome, modelo in modelos.items():
    print(f"\nTreinando modelo: {nome}")
    grid = GridSearchCV(modelo, param_grids[nome], cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring='f1', n_jobs=-1)
    grid.fit(X_resampled, y_resampled)
    best_model = grid.best_estimator_
    y_scores = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    idx = f1_scores.argmax()
    threshold = max(0.3, thresholds[idx])
    y_pred = (y_scores >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    print(f"{nome} - Limiar ótimo: {threshold:.2f} - F1: {f1:.4f}")

    if f1 > melhor_f1:
        melhor_modelo, melhor_nome, melhor_f1, melhor_threshold = best_model, nome, f1, threshold

print(f"\nMelhor modelo: {melhor_nome} | F1 = {melhor_f1:.4f} | Limiar = {melhor_threshold:.2f}")
y_final_pred = (melhor_modelo.predict_proba(X_test)[:, 1] >= melhor_threshold).astype(int)
print(classification_report(y_test, y_final_pred))

# SHAP explicabilidade
try:
    if melhor_nome in ['XGBoost', 'CatBoost', 'Random Forest']:
        explainer = shap.Explainer(melhor_modelo, X_test)
        shap_values = explainer(X_test)
        print("\nGerando visualizações SHAP...")
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "shap_summary_bar.png")
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "shap_summary_beeswarm.png")
        print("Visualizações SHAP salvas.")
except Exception as e:
    print(f"Erro ao gerar SHAP: {e}")

# Salvar
joblib.dump({
    'model': melhor_modelo,
    'scaler': scaler,
    'threshold': melhor_threshold,
    'selected_features': features
}, MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")

with open(BASE_DIR / 'model_features.txt', 'w') as f:
    f.write('\n'.join(features))
