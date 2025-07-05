
# Chatbot - Previsão de Sucesso de Projetos

Este chatbot coleta informações sobre um novo projeto e consulta uma API de Machine Learning para prever a chance de sucesso, com base também no histórico do usuário.

## Como usar

1. Estrutura dos arquivos do Chatbot
```bash
chatbot/
├── chatbot_app.py           # app Streamlit
├── chatbot_logic.py         # coleta de dados, chamada API
├── data/usuarios.csv        # base simples de usuários
├── README.md
```

2. Instale as dependências:
```bash
pip install streamlit pandas requests
```

3. Execute o chatbot:
```bash
streamlit run chatbot_app.py
```