import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Configuração da Interface Gráfica
st.set_page_config(page_title="Detector de Filme Bom", layout="centered")
st.title("🎬 IA: Detector de Filme Bom")
st.write("Análise de Regressão Linear para estimar a nota de um filme com base em sua duração (minutos).")

# 2. Definição do Dataset Base
filmes = pd.DataFrame({
    'duracao': [80, 90, 100, 110, 120],
    'nota': [4, 5, 7, 8, 9]
})

# 3. Modelagem e Ajuste do Modelo (Scikit-Learn)
# X: Feature / Variável Independente (Matriz 2D obrigatória no scikit-learn)
# y: Target / Variável Dependente (Vetor 1D)
X = filmes[['duracao']]
y = filmes['nota']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Painel de Controle de Entrada (Sidebar)
st.sidebar.header("Parâmetros do Filme")
duracao_input = st.sidebar.slider(
    label="Selecione a duração do filme (em minutos):",
    min_value=60,
    max_value=180,
    value=105,
    step=5
)

# 5. Pipeline de Inferência em Tempo Real
X_novo = np.array([[duracao_input]])
nota_predita = modelo.predict(X_novo)[0]

# Tratamento lógico de limites para manter a coerência da escala acadêmica/cinematográfica [0, 10]
nota_final = max(0.0, min(10.0, nota_predita))

# Exibição de Métricas na Interface
col1, col2 = st.columns(2)
col1.metric("Duração Avaliada", f"{duracao_input} min")
col2.metric("Nota Estimada pelo Modelo", f"{nota_final:.1f} / 10")

# 6. Geração da Representação Gráfica (Matplotlib)
fig, ax = plt.subplots(figsize=(9, 5))

# Plot dos dados reais (amostragem histórica)
ax.scatter(filmes['duracao'], filmes['nota'], color='#E50914', s=120, label='Dados Reais (Histórico)', zorder=3)

# Geração e plot da linha de tendência linear
X_espaco = np.linspace(60, 180, 100).reshape(-1, 1)
y_espaco = modelo.predict(X_espaco)
ax.plot(X_espaco, y_espaco, color='#221F1F', linestyle='--', linewidth=2, label='Reta de Regressão')

# Destaque do ponto gerado dinamicamente via inferência do usuário
ax.scatter(duracao_input, nota_final, color='#00B4D8', s=180, marker='s', edgecolors='black', label='Predição Atual', zorder=4)

# Propriedades estéticas e normatização do gráfico
ax.set_xlabel("Duração (Minutos)")
ax.set_ylabel("Nota do Filme")
ax.set_xlim(55, 185)
ax.set_ylim(0, 11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderização final na aplicação Streamlit
st.pyplot(fig)

# Validação do ajuste matemático
r2_score = modelo.score(X, y)
st.caption(f"**Métrica Estatística**: Coeficiente de Determinação (R²) de {r2_score:.4f}. Indica forte correlação linear no intervalo amostrado.")