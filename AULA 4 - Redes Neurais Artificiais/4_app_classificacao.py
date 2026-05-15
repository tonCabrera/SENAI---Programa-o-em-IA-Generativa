import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. Configuração da Interface Científica
st.set_page_config(page_title="Classificador de Desempenho", layout="centered")
st.title("📊 Classificação de Status Acadêmico via Regressão Logística")
st.write("Predição do status de aprovação (Aprovado/Reprovado) com base no histórico de faltas.")

# 2. Dataset Estruturado
alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]
})

# 3. Modelagem de Classificação Binária
# X: Feature (Faltas) -> Formato Bidimensional
# y: Target/Classe (Resultado) -> Vetor Unidimensional
X = alunos[['faltas']]
y = alunos['resultado']

modelo = LogisticRegression()
modelo.fit(X, y)

# 4. Painel de Controle de Entrada (Sidebar)
st.sidebar.header("Variável de Entrada")
faltas_input = st.sidebar.slider(
    "Selecione a quantidade de faltas:",
    min_value=0,
    max_value=12,
    value=3,
    step=1
)

# 5. Inferência do Modelo (Classe e Probabilidade)
X_novo = np.array([[faltas_input]])
classe_predita = modelo.predict(X_novo)[0]
# O método predict_proba retorna a probabilidade para ambas as classes [P(0), P(1)]
probabilidades = modelo.predict_proba(X_novo)[0]

# Mapeamento do output numérico para rótulo textual
status = "APROVADO" if classe_predita == 1 else "REPROVADO"
cor_status = "green" if classe_predita == 1 else "red"

# Exibição de Métricas na Interface
col1, col2 = st.columns(2)
col1.markdown(f"Status Predito: <h3 style='color:{cor_status}; margin:0;'>{status}</h3>", unsafe_allow_html=True)
col2.metric("Confiança da Predição", f"{probabilidades[classe_predita]*100:.1f}%")

# 6. Construção da Curva Sigmoide e Visualização Gráfica
fig, ax = plt.subplots(figsize=(10, 5))

# Plot das observações reais (0 ou 1)
ax.scatter(alunos['faltas'], alunos['resultado'], color='blue', s=100, label='Dados Reais', zorder=3)

# Geração do espaço amostral para desenhar a curva contínua de probabilidade
X_espaco = np.linspace(0, 12, 300).reshape(-1, 1)
# Extraímos apenas a probabilidade da classe 1 (Aprovado)
probabilidade_aprovacao = modelo.predict_proba(X_espaco)[:, 1]

# Plot da função logística (Sigmoide)
ax.plot(X_espaco, probabilidade_aprovacao, color='purple', linestyle='-', linewidth=2, label='Curva de Probabilidade (Sigmoide)')

# Linha de Fronteira de Decisão (Decision Boundary onde P = 0.5)
# Matematicamente: onde o modelo divide as classes
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Fronteira de Decisão (P=0.5)')

# Ponto dinâmico calculado com a entrada do usuário
ax.scatter(faltas_input, probabilidade_aprovacao[np.abs(X_espaco - faltas_input).argmin()], 
           color=cor_status, s=180, marker='X', label='Instância Atual', zorder=4)

# Configurações Estéticas do Eixo
ax.set_xlabel("Número de Faltas")
ax.set_ylabel("Probabilidade de Aprovação")
ax.set_xlim(-0.5, 12)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower left')

st.pyplot(fig)