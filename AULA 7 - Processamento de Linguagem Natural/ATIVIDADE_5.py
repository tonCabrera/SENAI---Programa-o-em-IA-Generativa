import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Garante o download seguro dos recursos de tokenização do NLTK
@st.cache_resource
def carregar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao baixar recursos do NLTK: {e}")

carregar_recursos_nltk()

# Configuração da página do Streamlit
st.set_page_config(page_title="Classificador de Sentimento", page_icon="🧠")

st.title("🧠 Classificador de Sentimento por Regras")
st.subheader("Análise rápida de comentários de marketing usando palavras-chave")

# 1. Definição dos nossos dicionários de termos (Léxico Manual)
PALAVRAS_POSITIVAS = {"bom", "ótimo", "otimo", "excelente", "maravilhoso", "amei", "adorei", "perfeito", "rápido", "recomendo"}
PALAVRAS_NEGATIVAS = {"ruim", "péssimo", "pessimo", "horrível", "horrivel", "odiei", "lento", "caro", "defeito", "quebrou", "pior"}

# Exemplo padrão para teste da equipe de marketing
exemplo_comentario = "O produto é excelente e muito rápido, mas o frete achei um pouco caro."

# Input do usuário: Comentário do cliente
comentario = st.text_area(
    "Insira o comentário do cliente para classificação:",
    value=exemplo_comentario,
    height=120
)

# Botão para executar a classificação
if st.button("Classificar Sentimento"):
    if comentario.strip():
        
        # 2. Tokenização com NLTK para isolar as palavras corretamente
        tokens = word_tokenize(comentario, language='portuguese')
        
        # Listas para guardar quais termos foram localizados no texto
        positivas_encontradas = []
        negativas_encontradas = []
        
        # 3. Varredura e contagem baseada nas regras condicionais
        for token in tokens:
            palavra_formatada = token.lower() # Padroniza para minúsculo
            
            if palavra_formatada in PALAVRAS_POSITIVAS:
                positivas_encontradas.append(palavra_formatada)
            elif palavra_formatada in PALAVRAS_NEGATIVAS:
                negativas_encontradas.append(palavra_formatada)
        
        # 4. Cálculo do Score Final (Saldo de sentimentos)
        score = len(positivas_encontradas) - len(negativas_encontradas)
        
        st.write("---")
        st.write("### 📊 Resultado da Análise")
        
        # 5. Regra Condicional Final para definir a classificação
        if score > 0:
            st.success(f"😊 **Sentimento Geral: POSITIVO** (Score: +{score})")
            st.write("O comentário possui mais termos elogiosos do que críticos.")
        elif score < 0:
            st.error(f"😢 **Sentimento Geral: NEGATIVO** (Score: {score})")
            st.write("Atenção! O cliente utilizou termos que indicam insatisfação.")
        else:
            st.info(f"😐 **Sentimento Geral: NEUTRO** (Score: {score})")
            st.write("O comentário é neutro ou possui um equilíbrio exato entre elogios e críticas.")
            
        # Exibição detalhada dos termos encontrados para a equipe de marketing
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Termos Positivos identificados:**")
            st.write(positivas_encontradas if positivas_encontradas else "Nenhum")
        with col2:
            st.markdown("**Termos Negativos identificados:**")
            st.write(negativas_encontradas if negativas_encontradas else "Nenhum")
            
    else:
        st.warning("Por favor, digite um comentário antes de classificar.")