from fastapi import FastAPI, HTTPException
import pandas as pd
import os

def buscar_usuario_por_id(user_id: int):
    """Função utilitária para buscar usuário por ID, para uso interno do chatbot."""
    import pandas as pd
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "usuarios.csv")
    df = pd.read_csv(csv_path)
    usuario = df[df['id_usuario'] == int(user_id)]
    if usuario.empty:
        return None
    row = usuario.iloc[0]
    return {
        "nome": str(row["nome"]),
        "idade": int(row["anos_experiencia"]),
        "projetos_concluidos": int(row["projetos_passados"]),
        "horas_trabalhadas": 40,
        "nivel_experiencia": min(int(row["anos_experiencia"] // 5 + 1), 5)
    }


app = FastAPI()

@app.get("/usuarios/{user_id}")
def get_usuario(user_id: int):
    try:
        # Caminho relativo para rodar de qualquer lugar
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "data", "usuarios.csv")
        df = pd.read_csv(csv_path)
        usuario = df[df['id_usuario'] == user_id]
        if usuario.empty:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        row = usuario.iloc[0]
        return {
            "nome": str(row["nome"]),
            "idade": int(row["anos_experiencia"]),
            "projetos_concluidos": int(row["projetos_passados"]),
            "horas_trabalhadas": 40,  # Valor fictício, ajuste conforme necessário
            "nivel_experiencia": min(int(row["anos_experiencia"] // 5 + 1), 5)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
