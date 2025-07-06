# 🔷 Previsão de Sucesso de Projetos

Este repositório apresenta uma solução completa para previsão de sucesso de projetos, combinando modelagem de machine learning, API de predição (FastAPI) e chatbot interativo (Streamlit) em uma arquitetura modular, flexível e documentada.

## 🔷 Estrutura de Diretórios

```bash
chatbot_projeto_sucesso/
├── modelo/        # Código, scripts e dados do modelo de machine learning
├── chatbot/       # Código do chatbot e integração com a API
```

## 🔷 Visão Geral

O projeto é composto por dois principais módulos:

- **modelo/**: Pipeline de modelagem em Python, scripts de treinamento e avaliação, dados históricos e modelo final salvo (Random Forest). O código é organizado, reprodutível e documentado, com métricas como F1-score, recall e acurácia reportadas e justificadas.
- **chatbot/**: Chatbot em Streamlit, responsável por coletar dados do usuário/projeto e consultar a API de predição (FastAPI). O sistema utiliza uma base de usuários para personalizar as respostas, sendo modular e de fácil adaptação.

## ⚙️ Como Executar

Consulte os READMEs específicos em cada subdiretório para instruções detalhadas de instalação, execução e exemplos de uso.



## 🔷 Objetivo

Automatizar a previsão de sucesso de projetos, facilitando a tomada de decisão e a análise de novos casos por meio de um chatbot, integração com modelos de machine learning e com foco em explicabilidade e inovação.