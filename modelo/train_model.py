import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
import pickle
from pathlib import Path


# Caminhos
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'historico_projetos.csv'
MODEL_PATH = BASE_DIR / 'model_optimized.pkl'

# 1. Carregar base histórica de projetos
print(f"Lendo base de projetos: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Renomear colunas para o padrão esperado pelo modelo
df = df.rename(columns={
    'id_usuario': 'user_id',
    'orcamento': 'valor_projeto',
    'duracao': 'duracao_dias',
})

# Adicionar colunas fictícias para compatibilidade com o modelo
for col in ['complexidade', 'nota_cliente', 'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia']:
    if col not in df.columns:
        if col == 'complexidade':
            df[col] = np.random.randint(1, 6, size=len(df))
        elif col == 'nota_cliente':
            df[col] = np.random.uniform(1, 5, size=len(df))
        elif col == 'idade':
            df[col] = np.random.randint(20, 60, size=len(df))
        elif col == 'projetos_concluidos':
            df[col] = np.random.randint(1, 20, size=len(df))
        elif col == 'horas_trabalhadas':
            df[col] = np.random.randint(10, 60, size=len(df))
        elif col == 'nivel_experiencia':
            df[col] = np.random.randint(1, 6, size=len(df))

# 2. Agregar variáveis históricas por usuário
agg = df.groupby('user_id').agg(
    media_sucesso=('sucesso', 'mean'),
    projetos_total=('sucesso', 'count'),
    media_valor=('valor_projeto', 'mean'),
    media_duracao=('duracao_dias', 'mean'),
    media_complexidade=('complexidade', 'mean'),
    media_nota_cliente=('nota_cliente', 'mean'),
    idade=('idade', 'last'),
    projetos_concluidos=('projetos_concluidos', 'last'),
    horas_trabalhadas=('horas_trabalhadas', 'last'),
    nivel_experiencia=('nivel_experiencia', 'last')
).reset_index()

# 3. Para cada projeto, combinar histórico do usuário + dados do novo projeto
projetos = df.copy()
projetos = projetos.merge(agg, on='user_id', suffixes=('', '_hist'))

# 4. Features de entrada: histórico + dados do novo projeto
features = [
    'media_sucesso', 'projetos_total', 'media_valor', 'media_duracao',
    'media_complexidade', 'media_nota_cliente',
    'idade', 'projetos_concluidos', 'horas_trabalhadas', 'nivel_experiencia',
    'valor_projeto', 'duracao_dias', 'complexidade', 'nota_cliente'
]
X = projetos[features]
y = projetos['sucesso']


# 5. Treinar modelo conforme pipeline do notebook (Random Forest com ajuste de threshold e scaler)

# Normalização das features numéricas
numeric_cols = ['media_valor', 'media_duracao', 'media_complexidade', 'media_nota_cliente',
               'idade', 'horas_trabalhadas', 'valor_projeto', 'duracao_dias', 'complexidade', 'nota_cliente']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=2, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Ajuste de threshold ótimo para F1
if hasattr(model, 'predict_proba'):
    y_scores = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = max(0.3, thresholds[optimal_idx])
    print(f"Melhor limiar para F1: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
else:
    optimal_threshold = 0.5

# Avaliação
y_pred = (model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred))

# Salvar modelo, scaler, threshold e features
joblib.dump({
    'model': model,
    'scaler': scaler,
    'threshold': optimal_threshold,
    'selected_features': features
}, MODEL_PATH)
print(f"Modelo completo salvo em {MODEL_PATH}")

# Salvar features usadas
with open(BASE_DIR / 'model_features.txt', 'w') as f:
    f.write('\n'.join(features))
print("Features salvas em model_features.txt")
