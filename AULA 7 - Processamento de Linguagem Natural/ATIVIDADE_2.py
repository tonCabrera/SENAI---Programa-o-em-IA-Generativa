import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

# Garante os downloads necessários do NLTK de forma segura
@st.cache_resource
def carregar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao baixar recursos do NLTK: {e}")

carregar_recursos_nltk()

# Configuração da página do Streamlit
st.set_page_config(page_title="Contador de Frequência", page_icon="📊")

st.title("📊 Análise de Frequência de Palavras")
st.subheader("Identifique os termos mais comuns nas avaliações dos clientes")

# Texto de exemplo simulando avaliações de clientes
exemplo_avaliacoes = """
O produto é excelente, entrega muito rápida e o produto veio muito bem embalado. 
Adorei o atendimento, porém o produto veio com uma pequena marca. 
No geral, o atendimento foi excelente e a entrega surpreendeu.
"""

# Input do usuário (Área de texto para colar os dados)
texto_entrada = st.text_area(
    "Cole as avaliações dos clientes aqui:",
    value=exemplo_avaliacoes.strip(),
    height=150
)

# Filtro extra: Permitir ignorar maiúsculas/minúsculas para não duplicar contagens (ex: "O" e "o")
ignorar_case = st.checkbox("Ignorar diferença entre Maiúsculas e Minúsculas", value=True)

# Botão para disparar o processamento
if st.button("Analisar Frequência"):
    if texto_entrada.strip():
        
        # 1. Ajusta o texto conforme a escolha do usuário
        texto_processado = texto_entrada.lower() if ignorar_case else texto_entrada
        
        # 2. Tokeniza o texto em palavras utilizando NLTK
        todas_palavras = word_tokenize(texto_processado, language='portuguese')
        
        # 3. Limpeza simples: Remove pontuações isoladas para focar apenas em termos textuais
        palavras_limpas = [palavra for palavra in todas_palavras if palavra.isalnum()]
        
        if palavras_limpas:
            # 4. Conta a frequência utilizando a classe Counter nativa do Python
            contador_frequencia = Counter(palavras_limpas)
            
            # 5. Transforma o resultado em um DataFrame do Pandas para exibição rica no Streamlit
            df_frequencia = pd.DataFrame(
                contador_frequencia.items(), 
                columns=['Palavra', 'Frequência']
            ).sort_values(by='Frequência', ascending=False).reset_index(drop=True)
            
            st.success("Análise concluída!")
            
            # Exibição dos dados estruturados
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Tabela de Frequência:**")
                st.dataframe(df_frequencia, use_container_width=True)
                
            with col2:
                st.write("**Gráfico dos Termos Mais Citados:**")
                # Exibe um gráfico de barras nativo do Streamlit limitando aos 10 termos mais frequentes
                st.bar_chart(data=df_frequencia.head(10), x='Palavra', y='Frequência')
                
        else:
            st.warning("O texto inserido não contém palavras válidas para análise pós-filtragem.")
    else:
        st.warning("Por favor, digite ou cole dados textuais para iniciar a contagem.")