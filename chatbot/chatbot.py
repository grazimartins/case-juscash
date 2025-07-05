import pandas as pd
from pathlib import Path
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import streamlit as st
from api_usuario import buscar_usuario_por_id

# Configuração de caminhos
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "usuarios.csv"
MODEL_PATH = BASE_DIR.parent / "modelo" / "model.pkl"

def apply_custom_css():
    """Aplica CSS elegante com tons de azul escuro, branco e cinza."""
    st.markdown("""
        <style>
            /* Fonte e cores */
            html, body, [class*="css"] {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 16px;
                color: #4B5563; /* Cinza escuro para texto */
                background-color: #F9FAFB; /* Cinza claro para fundo */
            }

            /* Cabeçalho */
            h2 {
                color: #1E3A8A;
                font-weight: 600;
            }

            /* Chat container */
            .stChatMessage {
                border-radius: 10px;
                padding: 12px 16px;
                margin-bottom: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                max-width: 80%;
            }

            .chatMessage.human {
                background-color: #FFFFFF; /* Branco */
                border-left: 4px solid #D1D5DB;
                margin-left: auto;
            }

            .chatMessage.ai {
                background-color: #F3F4F6; /* Cinza claro */
                border-left: 4px solid #D1D5DB;
                margin-right: auto;
            }

            /* Campo de entrada do chat */
            .stChatInput > div > div > input {
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                background-color: #FFFFFF;
            }

            /* Botões */
            .stButton > button {
                background-color: #1E3A8A; /* Azul escuro */
                color: #FFFFFF; /* Branco */
                font-weight: 500;
                border-radius: 8px;
                padding: 0.5em 1.5em;
                border: none;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }

            .stButton > button:hover {
                background-color: #1E3A8A;  /* Azul escuro */
                transform: translateY(-1px);
            }

            /* Barra lateral */
            .css-1v3fvcr {
                background-color: #FFFFFF;
                border-right: 1px solid #D1D5DB;
            }

            /* Linha divisória */
            hr {
                border: 0;
                height: 1px;
                background: #D1D5DB;
                margin: 1.5em 0;
            }
        </style>
    """, unsafe_allow_html=True)

def load_data():
    """Carrega os dados do usuário do arquivo CSV."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Arquivo usuarios.csv não encontrado em {DATA_PATH}")
        return None

def load_model():
    """Carrega o modelo."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            conteudo = joblib.load(MODEL_PATH)
            model = conteudo['model']
        return model
    except FileNotFoundError:
        st.error(f"Modelo 'model.pkl' não encontrado em {MODEL_PATH}")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def call_llm(prompt):
    """Placeholder para chamar o LLM (substitua pela API real)."""
    return f"[LLM Simulado] Resposta para: {prompt}\nRecomendações personalizadas baseadas na entrada."

def get_user_history(df, user_id):
    """Consulta o histórico do usuário no CSV."""
    if not user_id:
        return None
    user_data = df[df['user_id'] == user_id]
    if user_data.empty:
        st.error(f"Usuário {user_id} não encontrado no histórico.")
        return None
    return user_data

import requests
def predict_success_api(user_id, novo_projeto):
    """Chama a API de predição com user_id e dados do novo projeto."""
    url = "http://localhost:8000/predict"
    payload = {"user_id": user_id}
    payload.update(novo_projeto)
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        prob = response.json().get("probabilidade", None)
        return prob
    except Exception as e:
        return None

def make_recommendations(prob, feature_importance):
    """Gera recomendações usando o LLM."""
    max_feature = max(feature_importance, key=lambda k: abs(feature_importance[k]))
    max_value = feature_importance[max_feature]
    prompt = f"""
    A probabilidade de sucesso de um usuário é {prob * 100:.2f}%. 
    A feature mais importante é '{max_feature}' com impacto {max_value:.4f}.
    Gere recomendações personalizadas em português para melhorar o sucesso do usuário,
    focando em ações práticas relacionadas à feature mais importante.
    Se a probabilidade for menor que 50%, sugira melhorias. Se for maior, incentive a continuidade.
    Responda em até 3 frases, em tom motivador e claro.
    """
    return call_llm(prompt)

def initialize_session_state():
    """Inicializa o estado da sessão."""
    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.user_id = None
        st.session_state.user_input = []
        st.session_state.messages = [
            {"type": "ai", "content": "Bem-vindo ao <strong>Chat de Predição de Sucesso</strong>! Digite seu ID de usuário."}
        ]

def chat_app():
    """Gerencia a interface de chat."""
    df = load_data()
    model = load_model()
    if model is None:
        return

    container = st.container()
    for mensagem in st.session_state.messages:
        with container.chat_message(mensagem["type"]):
            css_class = f"chatMessage {mensagem['type']}"
            st.markdown(f"<div class='{css_class}'>{mensagem['content']}</div>", unsafe_allow_html=True)

    # As perguntas agora serão baseadas nos campos do CSV, exceto user_id
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
                    st.session_state.messages.append({"type": "ai", "content": "Usuário não encontrado via API. Digite um ID válido ou 'sim' para reiniciar."})
                    st.session_state.step = len(questions) + 1
                    return
                nome_usuario = user_data.get('nome', None)
                if nome_usuario:
                    st.session_state.messages.append({"type": "ai", "content": f"Olá, {nome_usuario}!"})
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
                    # Montar dicionário do novo projeto
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
                        # Exemplo de recomendação baseada na probabilidade
                        mensagem = f"Com base nos dados fornecidos e no seu histórico de projetos, o seu projeto tem <strong>{prob * 100:.0f}%</strong> de chance de ser bem-sucedido."
                        # Exemplo de análise simples de orçamento
                        orcamento = novo_projeto.get("valor_projeto", None)
                        if orcamento is not None and df is not None:
                            media_orcamento = df['valor_projeto'].mean() if 'valor_projeto' in df.columns else None
                            if media_orcamento and orcamento < media_orcamento:
                                mensagem += f"<br>Seu orçamento está abaixo da média dos projetos de sucesso (média: R$ {media_orcamento:.2f}). Considere ajustar o orçamento."
                        st.session_state.messages.append({"type": "ai", "content": mensagem})
                    else:
                        st.session_state.messages.append({"type": "ai", "content": "Erro ao obter predição da API."})
                    with container.chat_message("ai"):
                        st.markdown("<div class='chatMessage ai'>Deseja fazer outra predição? Digite 'sim' para reiniciar.</div>", unsafe_allow_html=True)
                    st.session_state.step += 1
            elif st.session_state.step == len(questions) + 1 and nova_mensagem.lower() == 'sim':
                st.session_state.step = 0
                st.session_state.user_id = None
                st.session_state.user_input = []
                st.session_state.messages = [{"type": "ai", "content": " Bem-vindo ao <strong>Chat de Predição de Sucesso</strong>! Digite seu ID de usuário."}]
            else:
                st.session_state.messages.append({"type": "ai", "content": "Resposta inválida. Digite 'sim' para reiniciar ou feche o chat."})
        except ValueError:
            st.session_state.messages.append({"type": "ai", "content": "Por favor, insira um valor numérico válido."})
        st.rerun()

def main():
    st.set_page_config(page_title="Chat de Predição de Sucesso", page_icon="", layout="wide")
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