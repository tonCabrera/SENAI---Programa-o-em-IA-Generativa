import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Configuração da Interface Analítica
st.set_page_config(page_title="Preditor de Sucesso Musical", layout="centered")
st.title("🎵 IA: Previsão de Potencial Viral de Músicas")
st.write("Ajuste linear estatístico para estimar o índice de viralização com base no BPM (Beats Per Minute).")

# 2. Dataset de Entrada (Corrigido para consistência sintática: 'musica')
musica = pd.DataFrame({
    'bpm': [80, 90, 100, 120, 140],
    'viral': [1, 2, 4, 7, 10]
})

# 3. Pipeline de Treinamento (Scikit-Learn)
# X: Vetor de Features (Estrutura bidimensional/Matriz exigida pelo estimador)
# y: Vetor de Target (Variável alvo contínua)
X = musica[['bpm']]
y = musica['viral']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Interface de Controle Operacional (Sidebar)
st.sidebar.header("Atributos da Faixa")
bpm_input = st.sidebar.slider(
    label="Defina o andamento da música (BPM):",
    min_value=60,
    max_value=200,
    value=110,
    step=5
)

# 5. Inferência Estatística
X_novo = np.array([[bpm_input]])
predicao_bruta = modelo.predict(X_novo)[0]

# Tratamento lógico de restrição de domínio (escala de viralização mapeada de 0 a 10)
viral_final = max(0.0, min(10.0, predicao_bruta))

# Exibição dos Indicadores na UI
col1, col2 = st.columns(2)
col1.metric("Andamento Definido", f"{bpm_input} BPM")
col2.metric("Potencial Viral Estimado", f"{viral_final:.2f} / 10")

# 6. Camada de Visualização Gráfica (Matplotlib)
fig, ax = plt.subplots(figsize=(9, 5))

# Plot das observações reais coletadas
ax.scatter(musica['bpm'], musica['viral'], color='#FF007F', s=120, label='Dados Amostrais (Faixas Reais)', zorder=3)

# Construção da reta de melhor ajuste (hiperplano de regressão)
X_espaco = np.linspace(60, 200, 100).reshape(-1, 1)
y_espaco = modelo.predict(X_espaco)
ax.plot(X_espaco, y_espaco, color='#1A1A2E', linestyle='--', linewidth=2, label='Função de Regressão Linear')

# Destaque vetorial da consulta interativa do usuário
ax.scatter(bpm_input, viral_final, color='#00F5D4', s=200, marker='*', edgecolors='black', label='Ponto de Inferência', zorder=4)

# Normatização estética do gráfico para publicações
ax.set_xlabel("Batidas por Minuto (BPM)")
ax.set_ylabel("Índice de Potencial Viral")
ax.set_xlim(55, 205)
ax.set_ylim(-0.5, 11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderização na Web
st.pyplot(fig)

# Métrica de Performance Acadêmica
r2_score = modelo.score(X, y)
st.caption(f"**Métrica Estatística**: O coeficiente de determinação (R²) do ajuste é {r2_score:.4f}, demonstrando alto grau de explicação da variância pelo modelo linear no intervalo amostrado.")