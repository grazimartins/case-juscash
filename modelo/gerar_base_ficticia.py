import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Parâmetros
total_usuarios = 30
projetos_por_usuario = np.random.randint(5, 20, total_usuarios)

projetos = []


for user_id in range(1, total_usuarios + 1):
    nome = f"Usuário {user_id}"
    idade = np.random.randint(20, 60)
    projetos_concluidos = np.random.randint(1, 20)
    horas_trabalhadas = np.random.randint(10, 60)
    nivel_experiencia = np.random.randint(1, 6)
    for i in range(projetos_por_usuario[user_id-1]):
        projeto_id = f"{user_id}_{i+1}"
        orcamento = np.random.randint(1000, 20000)
        duracao = np.random.randint(1, 24)  # meses
        sucesso = np.random.binomial(1, 0.7)  # 70% de chance de sucesso
        data_inicio = pd.Timestamp('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1000), unit='D')
        complexidade = np.random.randint(1, 6)
        nota_cliente = np.random.uniform(1, 5)
        projetos.append({
            "id_usuario": user_id,
            "nome_usuario": nome,
            "projeto_id": projeto_id,
            "orcamento": orcamento,
            "duracao": duracao,
            "sucesso": sucesso,
            "data_inicio": data_inicio,
            "complexidade": complexidade,
            "nota_cliente": nota_cliente,
            "idade": idade,
            "projetos_concluidos": projetos_concluidos,
            "horas_trabalhadas": horas_trabalhadas,
            "nivel_experiencia": nivel_experiencia
        })

# Salva o DataFrame

historico = pd.DataFrame(projetos)
output_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'historico_projetos.csv')
historico.to_csv(output_path, index=False)
print(f"Base fictícia salva em {output_path}")
