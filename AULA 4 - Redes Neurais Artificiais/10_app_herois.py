import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# 1. Configuração da Interface Analítica
st.set_page_config(page_title="Rede Neural dos Super-Heróis", layout="centered")
st.title("🦸‍♂️ IA: Rede Neural dos Super-Heróis")
st.write("Classificação binária (Fraco vs. Forte) utilizando um Perceptron Multicamadas (MLP).")

# 2. Dataset Base
# 0 = Fraco | 1 = Forte
herois = pd.DataFrame({
    'forca': [1, 2, 3, 7, 8, 10],
    'heroi': [0, 0, 0, 1, 1, 1]
})

# 3. Modelagem e Treinamento da Rede Neural
# X: Matriz de Features (Bidimensional)
# y: Vetor de Labels (Unidimensional)
X = herois[['forca']]
y = herois['heroi']

# Inicialização do Classificador MLP
# Definimos uma arquitetura simples com 1 camada oculta contendo 5 neurônios
# Definimos um random_state fixo para garantir a reprodutibilidade dos pesos iniciais
modelo_nn = MLPClassifier(
    hidden_layer_sizes=(5,), 
    activation='logistic', 
    solver='lbfgs', 
    max_iter=500, 
    random_state=42
)
modelo_nn.fit(X, y)

# 4. Painel de Entrada de Dados (Sidebar)
st.sidebar.header("Atributos do Herói")
forca_input = st.sidebar.slider(
    label="Selecione o nível de força do herói:",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

# 5. Pipeline de Inferência e Cálculo de Probabilidades
X_novo = np.array([[forca_input]])
classe_predita = modelo_nn.predict(X_novo)[0]
probabilidades = modelo_nn.predict_proba(X_novo)[0]

# Mapeamento semântico dos outputs
resultado_texto = "FORTE 💪" if classe_predita == 1 else "FRACO 📉"
cor_resultado = "#2ECC71" if classe_predita == 1 else "#E74C3C"

# Exibição de Métricas na Interface
col1, col2 = st.columns(2)
col1.markdown(f"Classificação: <h3 style='color:{cor_resultado}; margin:0;'>{resultado_texto}</h3>", unsafe_allow_html=True)
col2.metric("Confiança da Rede Neural", f"{probabilidades[classe_predita]*100:.1f}%")

# 6. Geração da Fronteira de Decisão Gráfica (Matplotlib)
fig, ax = plt.subplots(figsize=(10, 5))

# Plot dos dados amostrais (histórico dos heróis)
ax.scatter(herois['forca'], herois['heroi'], color='#34495E', s=120, label='Dados Amostrais', zorder=3)

# Geração de espaço contínuo para plotar a curva de decisão da rede neural
X_espaco = np.linspace(0, 12, 300).reshape(-1, 1)
# Probabilidade do herói pertencer à classe 1 (Forte)
probabilidade_forte = modelo_nn.predict_proba(X_espaco)[:, 1]

# Plot da curva de ativação da rede
ax.plot(X_espaco, probabilidade_forte, color='#9B59B6', linestyle='-', linewidth=2.5, label='Curva de Probabilidade da Rede')

# Linha divisória de decisão (Limiar P = 0.5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Fronteira de Decisão (P=0.5)')

# Ponto dinâmico gerado pela inferência atual do usuário
ponto_prob = modelo_nn.predict_proba(X_novo)[0][1]
ax.scatter(forca_input, ponto_prob, color=cor_resultado, s=180, marker='X', edgecolors='black', label='Instância Atual', zorder=4)

# Ajustes estéticos padrão acadêmico
ax.set_xlabel("Nível de Força")
ax.set_ylabel("Probabilidade de ser Classificado como Forte")
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower right')

# Renderização final na aplicação Streamlit
st.pyplot(fig)

# Métrica de Performance do Ajuste
acuracia = modelo_nn.score(X, y)
st.caption(f"**Nota de Arquitetura**: Rede Neural ajustada com acurácia de {acuracia*100:.0f}% sobre o conjunto de dados fornecido.")