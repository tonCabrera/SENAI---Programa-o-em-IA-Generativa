import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Configuração da Interface Analítica
st.set_page_config(page_title="Preditor de Energia por Cafeína", layout="centered")
st.title("⚡ IA: Previsão de Energia baseada em Cafés Tomados")
st.write("Abordagem por Regressão Linear para mensurar o impacto do consumo de café nos níveis de energia.")

# 2. Definição do Dataset Amostral
cafe = pd.DataFrame({
    'xicaras': [1, 2, 3, 4, 5],
    'energia': [2, 4, 6, 8, 10]
})

# 3. Modelagem e Ajuste Matemática (Scikit-Learn)
# X: Feature / Variável Independente (Estrutura bidimensional/Matriz)
# y: Target / Variável Dependente (Vetor 1D)
X = cafe[['xicaras']]
y = cafe['energia']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Camada de Entrada de Dados (Interface Reativa)
st.sidebar.header("Variável de Entrada")
xicaras_input = st.sidebar.slider(
    label="Selecione a quantidade de xícaras de café:",
    min_value=0.0,
    max_value=7.0,
    value=3.0,
    step=0.5
)

# 5. Pipeline de Inferência
X_novo = np.array([[xicaras_input]])
energia_predita = modelo.predict(X_novo)[0]

# Tratamento de limites para adequação à escala de domínio de negócio [0, 12]
energia_final = max(0.0, energia_predita)

# Exibição dos Indicadores na UI
col1, col2 = st.columns(2)
col1.metric("Consumo Selecionado", f"{xicaras_input} xícaras")
col2.metric("Nível de Energia Estimado", f"{energia_final:.1f}")

# 6. Geração da Representação Gráfica (Matplotlib)
fig, ax = plt.subplots(figsize=(9, 5))

# Plot das observações reais coletadas (Dados Históricos)
ax.scatter(cafe['xicaras'], cafe['energia'], color='#6F4E37', s=120, label='Dados Históricos', zorder=3)

# Geração e plot da linha de tendência linear (hiperplano de melhor ajuste)
X_espaco = np.linspace(0, 7, 100).reshape(-1, 1)
y_espaco = modelo.predict(X_espaco)
ax.plot(X_espaco, y_espaco, color='#D4A373', linestyle='--', linewidth=2, label='Função de Regressão Linear')

# Destaque do ponto dinâmico de predição solicitado pelo usuário
ax.scatter(xicaras_input, energia_final, color='#FF4B4B', s=180, marker='X', edgecolors='black', label='Predição Atual', zorder=4)

# Normatização estética do gráfico para padrões acadêmicos
ax.set_xlabel("Quantidade de Xícaras")
ax.set_ylabel("Nível de Energia")
ax.set_xlim(-0.2, 7.2)
ax.set_ylim(-0.5, 15)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderização final na aplicação Streamlit
st.pyplot(fig)

# Validação do ajuste do modelo
r2_score = modelo.score(X, y)
st.caption(f"**Nota Estatística**: O Coeficiente de Determinação (R²) é {r2_score:.4f}, refletindo um ajuste determinístico e linear perfeito.")