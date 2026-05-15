import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da página e estilo visual
st.set_page_config(page_title="Análise de Fadiga Gamer", layout="centered")
st.title("Predição de Nível de Cansaço em Gamers")
st.write("Estudo analítico do impacto do tempo de jogo contínuo na fadiga percebida.")

# 1. Definição do Dataset Proposto (ID: jlwm1z)
gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# 2. Processamento e Ajuste do Modelo Linear
# X (Variável independente / Feature): Matriz 2D
# y (Variável dependente / Target): Vetor 1D
X = gamer[['horas_jogo']]
y = gamer['cansaco']

modelo = LinearRegression()
modelo.fit(X, y)

# 3. Interface de Controle (Sidebar)
st.sidebar.header("Variável de Entrada")
horas_digitadas = st.sidebar.slider(
    label="Selecione a quantidade de horas jogadas:",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

# 4. Cálculo da Inferência em Tempo Real
X_novo = np.array([[horas_digitadas]])
predicao_bruta = modelo.predict(X_novo)[0]

# Limitação lógica da escala de cansaço para o intervalo [0, 10]
nivel_cansaco_predito = max(0.0, min(10.0, predicao_bruta))

# Exibição do resultado inferido
st.metric(label="Nível de Cansaço Estimado (0 a 10)", value=f"{nivel_cansaco_predito:.2f}")

# 5. Construção e Renderização da Representação Gráfica
fig, ax = plt.subplots(figsize=(8, 5))

# Scatter plot dos dados amostrais (fatos observados)
ax.scatter(gamer['horas_jogo'], gamer['cansaco'], color='purple', s=80, label='Amostras Reais', zorder=3)

# Geração da reta de regressão contínua
X_vetor_linha = np.linspace(0, 12, 100).reshape(-1, 1)
y_vetor_linha = modelo.predict(X_vetor_linha)
ax.plot(X_vetor_linha, y_vetor_linha, color='darkgray', linestyle='--', linewidth=1.5, label='Reta de Melhor Ajuste')

# Destaque do ponto dinâmico de predição solicitado pelo usuário
ax.scatter(horas_digitadas, nivel_cansaco_predito, color='green', s=180, marker='*', label='Interpolação/Predição', zorder=4)

# Formatação das propriedades do gráfico (Padrão Acadêmico)
ax.set_xlabel("Horas de Jogo Contínuas")
ax.set_ylabel("Índice de Cansaço Percebido")
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper left')

# Renderização final na interface web
st.pyplot(fig)