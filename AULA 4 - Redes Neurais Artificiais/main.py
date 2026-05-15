import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da página acadêmica
st.set_page_config(page_title="Análise de Regressão Linear", layout="centered")
st.title("Predição de Desempenho Acadêmico")
st.write("Interface analítica para previsão de notas baseada em horas de estudo.")

# 1. Definição do Dataset Core
dados = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# 2. Pipeline de Machine Learning
X = dados[['horas']]
y = dados['notas']

modelo = LinearRegression()
modelo.fit(X, y)

# 3. Interface do Usuário (Sidebar para Entrada de Dados)
st.sidebar.header("Parâmetros de Entrada")
horas_inseridas = st.sidebar.slider(
    label="Selecione a quantidade de horas estudadas:",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.5
)

# 4. Predição em Tempo Real
X_novo = np.array([[horas_inseridas]])
nota_prevista = modelo.predict(X_novo)[0]
# Garantir que a nota permaneça dentro dos limites acadêmicos convencionais [0, 10]
nota_final = max(0.0, min(10.0, nota_prevista))

# Exibição dos Resultados da Predição
st.metric(label="Nota Estimada", value=f"{nota_final:.2f}")

# 5. Geração da Representação Gráfica
fig, ax = plt.subplots(figsize=(8, 5))

# Plot dos dados históricos (pontos reais)
ax.scatter(dados['horas'], dados['notas'], color='blue', label='Dados Reais (Histórico)', zorder=3)

# Plot da linha de regressão (tendência matemática)
X_linha = np.linspace(0, 12, 100).reshape(-1, 1)
y_linha = modelo.predict(X_linha)
ax.plot(X_linha, y_linha, color='gray', linestyle='--', label='Reta de Regressão')

# Plot do ponto dinâmico estimado pelo usuário
ax.scatter(horas_inseridas, nota_final, color='red', s=150, marker='X', label='Predição Atual', zorder=4)

# Customização do Gráfico
ax.set_xlabel("Horas de Estudo")
ax.set_ylabel("Nota Obtida")
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

# Renderização do gráfico no Streamlit
st.pyplot(fig)