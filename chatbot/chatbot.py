import pandas as pd
from pathlib import Path
import joblib
import streamlit as st
import requests
from api_usuario import buscar_usuario_por_id

# ===== CAMINHOS =====
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "usuarios.csv"
MODEL_PATH = BASE_DIR.parent / "modelo" / "artifacts"/  "model.pkl"

# ===== CSS =====
def apply_custom_css():
    st.markdown("""<style>
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-size: 16px;
            color: #4B5563;
            background-color: #F9FAFB;
        }
        h2 { color: #1E3A8A; font-weight: 600; }
        .stChatMessage {
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            max-width: 80%;
        }
        .chatMessage.human {
            background-color: #FFFFFF;
            border-left: 4px solid #D1D5DB;
            margin-left: auto;
        }
        .chatMessage.ai {
            background-color: #F3F4F6;
            border-left: 4px solid #D1D5DB;
            margin-right: auto;
        }
        .stChatInput > div > div > input {
            border: 1px solid #D1D5DB;
            border-radius: 8px;
            background-color: #FFFFFF;
        }
        .stButton > button {
            background-color: #1E3A8A;
            color: #FFFFFF;
            font-weight: 500;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            border: none;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #1E3A8A;
            transform: translateY(-1px);
        }
        .css-1v3fvcr {
            background-color: #FFFFFF;
            border-right: 1px solid #D1D5DB;
        }
        hr {
            border: 0;
            height: 1px;
            background: #D1D5DB;
            margin: 1.5em 0;
        }
    </style>""", unsafe_allow_html=True)

# ===== CARREGAMENTO =====
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Arquivo usuarios.csv não encontrado em {DATA_PATH}")
        return None

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            conteudo = joblib.load(file)
            return conteudo['model']
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# ===== API DE PREDIÇÃO =====
def predict_success_api(user_id, novo_projeto):
    url = "http://localhost:8000/predict"
    payload = {"user_id": user_id}
    payload.update(novo_projeto)
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("probabilidade", None)
    except Exception:
        return None

# ===== GERA RECOMENDAÇÃO =====
def make_recommendations(prob, novo_projeto, df):
    mensagens = []
    chance_text = f"Com base nos dados fornecidos e no seu histórico de projetos, o seu projeto tem <strong>{prob * 100:.0f}%</strong> de chance de ser bem-sucedido."
    mensagens.append(chance_text)

    if df is not None and 'valor_projeto' in df.columns:
        media_orcamento = df['valor_projeto'].mean()
        orcamento = novo_projeto.get("valor_projeto")
        if orcamento and orcamento < media_orcamento:
            mensagens.append(f"Seu orçamento está abaixo da média dos projetos de sucesso (média: R$ {media_orcamento:.2f}). Considere ajustar o orçamento.")

    feature_names = {
        "valor_projeto": "Valor do projeto",
        "duracao_dias": "Duração em dias",
        "complexidade": "Complexidade",
        "nota_cliente": "Nota do cliente"
    }

    model = load_model()
    if model:
        try:
            import numpy as np
            feature_importances = model.feature_importances_
            linhas = []
            for idx, key in enumerate(novo_projeto):
                nome_amigavel = feature_names.get(key, key)
                impacto = feature_importances[idx] * 100
                if impacto > 0:
                    linhas.append(f"\n - {nome_amigavel} apresenta impacto {impacto:.0f}%. Considere otimizar item para melhorar os resultados.\n")
            if linhas:
                mensagens.append("Recomendações:\n" + "\n".join(linhas))
        except:
            pass

    return mensagens

# ===== ESTADO =====
def initialize_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.user_id = None
        st.session_state.user_input = []
        st.session_state.messages = [
            {"type": "ai", "content": "Bem-vindo ao <strong>Chat de Predição de Sucesso</strong>! Digite seu ID de usuário."}
        ]

# ===== CHAT =====
def chat_app():
    df = load_data()
    model = load_model()
    if model is None:
        return

    container = st.container()
    for msg in st.session_state.messages:
        with container.chat_message(msg["type"]):
            st.markdown(f"<div class='chatMessage {msg['type']}'>{msg['content']}</div>", unsafe_allow_html=True)

    questions = [
        "Qual o valor do novo projeto? (Ex.: 15000)",
        "Qual a duração do novo projeto em dias? (Ex.: 30)",
        "Qual a complexidade do novo projeto? (1 a 5)",
        "Qual a nota do cliente esperada? (1 a 5)",
    ]

    nova_mensagem = st.chat_input(" Digite sua resposta...")
    if nova_mensagem:
        st.session_state.messages.append({"type": "human", "content": nova_mensagem})
        try:
            if st.session_state.step == 0:
                st.session_state.user_id = nova_mensagem
                user_data = buscar_usuario_por_id(st.session_state.user_id)
                if user_data is None:
                    st.session_state.messages.append({"type": "ai", "content": "Usuário não encontrado. Digite um ID válido ou 'sim' para reiniciar."})
                    st.session_state.step = len(questions) + 1
                    return
                nome = user_data.get('nome', '')
                st.session_state.messages.append({"type": "ai", "content": f"Olá, {nome}!"})
                st.session_state.user_input = []
                st.session_state.messages.append({"type": "ai", "content": questions[0]})
                st.session_state.step = 1
            elif st.session_state.step <= len(questions):
                value = float(nova_mensagem)
                st.session_state.user_input.append(value)
                if st.session_state.step < len(questions):
                    st.session_state.messages.append({"type": "ai", "content": questions[st.session_state.step]})
                    st.session_state.step += 1
                else:
                    novo_projeto = {
                        "valor_projeto": st.session_state.user_input[0],
                        "duracao_dias": st.session_state.user_input[1],
                        "complexidade": st.session_state.user_input[2],
                        "nota_cliente": st.session_state.user_input[3],
                    }
                    with container.chat_message("ai"):
                        st.markdown("<div class='chatMessage ai'>⏳ Calculando predição...</div>", unsafe_allow_html=True)
                    prob = predict_success_api(st.session_state.user_id, novo_projeto)
                    if prob is not None:
                        recomendacoes = make_recommendations(prob, novo_projeto, df)
                        for rec in recomendacoes:
                            st.session_state.messages.append({"type": "ai", "content": rec})
                    else:
                        st.session_state.messages.append({"type": "ai", "content": "Erro ao obter predição da API."})
                    st.session_state.messages.append({"type": "ai", "content": "Deseja fazer outra predição? Digite 'sim' para reiniciar."})
                    st.session_state.step += 1
            elif st.session_state.step == len(questions) + 1 and nova_mensagem.lower() == 'sim':
                st.session_state.clear()
                initialize_session_state()
            else:
                st.session_state.messages.append({"type": "ai", "content": "Digite 'sim' para reiniciar ou feche o chat."})
        except ValueError:
            st.session_state.messages.append({"type": "ai", "content": "Por favor, insira um número válido."})
        st.rerun()

# ===== MAIN =====
def main():
    st.set_page_config(page_title="Chat de Predição de Sucesso", layout="wide")
    apply_custom_css()
    st.markdown("<h2 style='color:#1E3A8A;'> Bem-vindo ao <strong>Chat de Predição de Sucesso</strong>!</h2><hr>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h3 style='color:#1E3A8A;'>⚙️ Configurações</h3>", unsafe_allow_html=True)
        if st.button("Reiniciar Chat", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    initialize_session_state()
    chat_app()

if __name__ == "__main__":
    main()
