import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Garante o download seguro e com cache dos recursos do NLTK
@st.cache_resource
def carregar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao baixar recursos do NLTK: {e}")

carregar_recursos_nltk()

# Configuração da página do Streamlit
st.set_page_config(page_title="Análise de Satisfação", page_icon="🛍️")

st.title("🛍️ Analisador Automático de Avaliações")
st.subheader("Identifique a satisfação de clientes sem a necessidade de leitura manual")

# 1. Criação dos dicionários léxicos de sentimento (Satisfação vs Insatisfação)
DICIONARIO_SATISFEITO = {"excelente", "bom", "ótimo", "otimo", "maravilhoso", "perfeito", "adorei", "amei", "recomendo", "rápido"}
DICIONARIO_INSATISFEITO = {"ruim", "péssimo", "pessimo", "horrível", "horrivel", "defeito", "quebrou", "lento", "pior", "odiei", "insatisfeito"}

# Exemplo padrão de avaliação mista/positiva para o usuário testar
exemplo_avaliacao = "O produto chegou muito rápido e o design é perfeito! Porém, achei o cabo um pouco curto, mas no geral o funcionamento é muito bom."

# Input do usuário: Texto da avaliação do produto
avaliacao_texto = st.text_area(
    "Insira a avaliação do produto para analisar o nível de satisfação:",
    value=exemplo_avaliacao,
    height=120
)

# Botão para disparar o pipeline de análise
if st.button("Analisar Nível de Satisfação"):
    if avaliacao_texto.strip():
        
        # 2. Tokenização: Isola palavras de pontuações de forma precisa com NLTK
        tokens_originais = word_tokenize(avaliacao_texto, language='portuguese')
        
        # Contadores de pontuação e listas de rastreamento
        pontos_positivos = 0
        pontos_negativos = 0
        termos_positivos = []
        termos_negativos = []
        
        # 3. Lógica Condicional Combinada: Varre os tokens normalizados (minúsculos)
        for token in tokens_originais:
            palavra = token.lower()
            
            if palavra in DICIONARIO_SATISFEITO:
                pontos_positivos += 1
                termos_positivos.append(palavra)
            elif palavra in DICIONARIO_INSATISFEITO:
                pontos_negativos += 1
                termos_negativos.append(palavra)
        
        # 4. Cálculo do Saldo de Satisfação (Métrica Final)
        saldo_final = pontos_positivos - pontos_negativos
        
        st.write("---")
        st.write("### 📊 Relatório de Classificação")
        
        # 5. Regras Condicionais Finais para determinar o estado de Satisfação
        if saldo_final > 0:
            st.success("😊 **Resultado: CLIENTE SATISFEITO**")
            st.write("A avaliação apresenta um tom predominante de aprovação e satisfação com o produto.")
            
        elif saldo_final < 0:
            st.error("😢 **Resultado: CLIENTE INSATISFEITO**")
            st.write("Atenção! Os termos utilizados indicam frustração ou problemas com a experiência de compra.")
            
        else:
            st.info("😐 **Resultado: SENTIMENTO NEUTRO OU MISTO**")
            st.write("O cliente manteve uma postura neutra ou equilibrou perfeitamente os pontos positivos e negativos.")
            
        # Exibição detalhada para fins de auditoria do analista
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Termos de Satisfação", value=pontos_positivos)
            if termos_positivos:
                st.caption(f"Palavras identificadas: {', '.join(set(termos_positivos))}")
                
        with col2:
            st.metric(label="Termos de Insatisfação", value=pontos_negativos)
            if termos_negativos:
                st.caption(f"Palavras identificadas: {', '.join(set(termos_negativos))}")
                
    else:
        st.warning("Por favor, digite uma avaliação para rodar o modelo condicional.")