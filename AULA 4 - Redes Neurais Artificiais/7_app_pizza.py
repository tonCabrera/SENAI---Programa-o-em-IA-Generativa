import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Configuração da Interface Gráfica
st.set_page_config(page_title="Preditor de Preço de Pizzas", layout="centered")
st.title("🍕 IA: Precificação de Pizzas por Tamanho")
st.write("Análise de Regressão Linear para estimar o valor econômico com base no diâmetro (cm).")

# 2. Dataset Base
pizza = pd.DataFrame({
    'tamanho': [20, 25, 30, 35, 40],
    'preco': [20, 30, 40, 50, 60]
})

# 3. Modelagem e Ajuste do Modelo (Scikit-Learn)
# X: Feature / Variável Independente (Matriz 2D)
# y: Target / Variável Dependente (Vetor 1D)
X = pizza[['tamanho']]
y = pizza['preco']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Painel de Controle de Entrada (Sidebar)
st.sidebar.header("Parâmetros do Produto")
tamanho_input = st.sidebar.slider(
    label="Selecione o tamanho da pizza (Diâmetro em cm):",
    min_value=15.0,
    max_value=50.0,
    value=33.0,
    step=1.0
)

# 5. Pipeline de Inferência em Tempo Real
X_novo = np.array([[tamanho_input]])
preco_predito = modelo.predict(X_novo)[0]

# Garantir que o preço não seja negativo devido à extrapolação matemática
preco_final = max(0.0, preco_predito)

# Exibição de Métricas na Interface
col1, col2 = st.columns(2)
col1.metric("Tamanho Escolhido", f"{tamanho_input} cm")
col2.metric("Preço Estimado", f"R$ {preco_final:.2f}")

# 6. Geração da Representação Gráfica (Matplotlib)
fig, ax = plt.subplots(figsize=(9, 5))

# Plot dos dados reais (amostragem histórica)
ax.scatter(pizza['tamanho'], pizza['preco'], color='#E63946', s=120, label='Dados Históricos (Pizzaria)', zorder=3)

# Geração e plot da linha de tendência linear (hiperplano de regressão)
X_espaco = np.linspace(15, 50, 100).reshape(-1, 1)
y_espaco = modelo.predict(X_espaco)
ax.plot(X_espaco, y_espaco, color='#1D3557', linestyle='--', linewidth=2, label='Modelo de Regressão Linear')

# Destaque do ponto gerado dinamicamente via inferência do usuário
ax.scatter(tamanho_input, preco_final, color='#FFB703', s=180, marker='D', edgecolors='black', label='Predição Atual', zorder=4)

# Propriedades estéticas e normatização acadêmica do gráfico
ax.set_xlabel("Tamanho da Pizza (Diâmetro em cm)")
ax.set_ylabel("Preço de Venda (R$)")
ax.set_xlim(13, 52)
ax.set_ylim(0, 90)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderização final na aplicação Streamlit
st.pyplot(fig)

# Validação do ajuste do modelo
r2_score = modelo.score(X, y)
st.caption(f"**Métrica Estatística**: O Coeficiente de Determinação (R²) é {r2_score:.4f}, indicando um ajuste perfeito aos dados fornecidos.")