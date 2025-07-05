import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# 5. Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Salvar modelo
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"Modelo salvo em {MODEL_PATH}")

# 8. Salvar features usadas
with open(BASE_DIR / 'model_features.txt', 'w') as f:
    f.write('\n'.join(features))
print("Features salvas em model_features.txt")
