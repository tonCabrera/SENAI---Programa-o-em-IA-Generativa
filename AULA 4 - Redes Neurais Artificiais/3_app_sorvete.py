import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Configuração de Estilo e Cabeçalho
st.set_page_config(page_title="Preditor de Vendas de Sorvete", layout="centered")
st.title("🍦 IA: Previsão de Vendas por Temperatura")
st.markdown("""
Esta aplicação utiliza um modelo de **Regressão Linear** para prever o volume de vendas 
com base na oscilação térmica.
""")

# 2. Dataset Base
sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# 3. Modelagem Matemática (Scikit-Learn)
# X: Features (Temperatura) - Necessário formato 2D [[]]
# y: Target (Vendas)
X = sorvete[['temperatura']]
y = sorvete['vendas']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Interface do Desenvolvedor (Input via Slider)
st.sidebar.header("Parâmetros do Modelo")
temp_input = st.sidebar.slider(
    "Selecione a Temperatura (°C):",
    min_value=15.0,
    max_value=45.0,
    value=25.0,
    step=0.5
)

# 5. Inferência e Resultados
venda_predita = modelo.predict(np.array([[temp_input]]))[0]

col1, col2 = st.columns(2)
col1.metric("Temperatura Selecionada", f"{temp_input}°C")
col2.metric("Vendas Estimadas", f"{max(0, int(venda_predita))} unidades")

# 6. Representação Gráfica
fig, ax = plt.subplots(figsize=(10, 6))

# Plot dos dados históricos
ax.scatter(sorvete['temperatura'], sorvete['vendas'], color='#005088', s=100, label='Dados Históricos', zorder=3)

# Reta de Tendência
X_plot = np.linspace(15, 45, 100).reshape(-1, 1)
y_plot = modelo.predict(X_plot)
ax.plot(X_plot, y_plot, color='#11caa0', linestyle='--', linewidth=2, label='Modelo (Regressão Linear)')

# Ponto da Predição Atual
ax.scatter(temp_input, venda_predita, color='red', s=200, marker='*', label='Ponto de Predição', zorder=4)

# Estética Acadêmica
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Quantidade de Vendas")
ax.set_title("Correlação Temperatura vs. Vendas")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

st.pyplot(fig)

# Comentário Didático
st.info(f"""
**Nota Didática:** O modelo calculou um coeficiente de determinação (R²) de {modelo.score(X, y):.2f}, 
o que indica uma correlação extremamente forte entre o calor e o consumo do produto.
""")
