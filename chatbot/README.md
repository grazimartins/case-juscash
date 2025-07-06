# ⚙️ Chatbot para Previsão de Sucesso de Projetos

Este diretório contém o chatbot desenvolvido para coletar informações de novos projetos e consultar uma API de Machine Learning (implementada em FastAPI), prevendo a chance de sucesso com base no histórico do usuário.

## 🔷 Estrutura do Diretório

- `chatbot_app.py`: Aplicação principal em Streamlit.
- `chatbot_logic.py`: Lógica de coleta de dados e integração com a API de predição (FastAPI).
- `data/usuarios.csv`: Base de usuários para simulação e personalização.

## ⚙️ Como Executar

1. Instale as dependências necessárias:
```bash
pip install streamlit pandas requests
```

2. Execute o chatbot:
```bash
streamlit run chatbot_app.py
```

---
## 🔹 Funcionamento

O chatbot solicita dados do projeto e do usuário, envia as informações para a API de Machine Learning e exibe a previsão de sucesso de forma interativa e amigável. O histórico do usuário pode ser utilizado para personalizar a resposta.

---
## 🔵 Observações

- Certifique-se de que a API de predição (FastAPI) esteja em execução para o chatbot funcionar corretamente.
- O fluxo pode ser adaptado para integração com diferentes modelos ou bases de usuários.