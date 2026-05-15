import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Inicialização da Interface Gráfica
st.set_page_config(page_title="Análise de Bem-estar Animal", layout="centered")
st.title("🐾 Preditor de Felicidade Canina")
st.write("Abordagem estatística linear para correlacionar a frequência de passeios ao nível de felicidade estimulado.")

# 2. Definição do Dataset Amostral
pets = pd.DataFrame({
    'passeios': [1, 2, 3, 4, 5],
    'felicidade': [2, 4, 5, 8, 10]
})

# 3. Modelagem e Ajuste Matemática (Scikit-Learn)
# X: Variável independente / Feature (Matriz 2D)
# y: Variável dependente / Target (Vetor 1D)
X = pets[['passeios']]
y = pets['felicidade']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Camada de Entrada de Dados (Interface Reativa)
st.sidebar.header("Parâmetros de Entrada")
passeios_input = st.sidebar.slider(
    label="Selecione a quantidade de passeios semanais:",
    min_value=0.0,
    max_value=7.0,
    value=3.0,
    step=0.5
)

# 5. Pipeline de Inferência
X_novo = np.array([[passeios_input]])
felicidade_predita = modelo.predict(X_novo)[0]

# Tratamento lógico de limites (escala convencional de 0 a 10)
felicidade_final = max(0.0, min(10.0, felicidade_predita))

# Exibição de Resultados da Inferência
st.metric(label="Índice de Felicidade Estimado", value=f"{felicidade_final:.2f} / 10")

# 6. Geração da Representação Gráfica (Matplotlib)
fig, ax = plt.subplots(figsize=(9, 5))

# Plot do histórico amostral (dados reais)
ax.scatter(pets['passeios'], pets['felicidade'], color='#FF9F1C', s=120, label='Dados Históricos', zorder=3)

# Geração e plot da linha de tendência linear
X_espaco = np.linspace(0, 7, 100).reshape(-1, 1)
y_espaco = modelo.predict(X_espaco)
ax.plot(X_espaco, y_espaco, color='#2EC4B6', linestyle='--', linewidth=2, label='Reta de Regressão')

# Destaque do ponto gerado dinamicamente pelo usuário
ax.scatter(passeios_input, felicidade_final, color='#E71D36', s=180, marker='o', edgecolors='black', label='Predição Atual', zorder=4)

# Formatação e propriedades estéticas do gráfico
ax.set_xlabel("Frequência de Passeios")
ax.set_ylabel("Índice de Felicidade")
ax.set_xlim(0, 7)
ax.set_ylim(0, 11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderização final na aplicação Streamlit
st.pyplot(fig)

# Validação estatística didática
r2_score = modelo.score(X, y)
st.caption(f"**Nota Técnica**: O Coeficiente de Determinação (R²) deste modelo é {r2_score:.4f}, evidenciando forte ajuste linear.")