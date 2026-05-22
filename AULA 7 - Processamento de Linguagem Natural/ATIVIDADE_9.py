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
st.set_page_config(page_title="Normalizador de Texto", page_icon="🧹")

st.title("🧹 Normalizador e Limpador de Textos")
st.subheader("Prepare dados brutos para sistemas de IA removendo ruídos e pontuações")

# Exemplo de texto "sujo" com pontuações variadas e caixas misturadas
exemplo_sujo = "ATENÇÃO!!! O produto (que comprei online) chegou... quebrado?! Entrei em contato com o suporte, mas... nada."

# Input do usuário: Texto bruto para limpeza
texto_bruto = st.text_area(
    "Insira o texto bruto que deseja normalizar:",
    value=exemplo_sujo,
    height=120
)

# Botão para executar o pipeline de limpeza
if st.button("Executar Limpeza e Normalização"):
    if texto_bruto.strip():
        
        # 1. Tokenização: O NLTK separa pontuações e caracteres especiais das palavras de forma cirúrgica
        tokens_brutos = word_tokenize(texto_bruto, language='portuguese')
        
        # 2. Normalização e Remoção de Pontuação: 
        # Convertemos para minúsculo (.lower()) e filtramos apenas caracteres alfanuméricos (.isalnum())
        tokens_limpos = [
            token.lower() for token in tokens_brutos 
            if token.isalnum()
        ]
        
        # 3. Reconstrução: Junta as palavras limpas em uma string única espaçada
        texto_resultado = " ".join(tokens_limpos)
        
        st.success("Pipeline de processamento aplicado!")
        st.write("---")
        
        # Exibição dos resultados em painéis comparativos
        col_esq, col_dir = st.columns(2)
        
        with col_esq:
            st.markdown("### 📄 Texto Original")
            st.info(texto_bruto)
            
            # Mostra a contagem de elementos originais
            st.caption(f"Contagem original: {len(tokens_brutos)} elementos (incluindo pontuações).")
            
        with col_dir:
            st.markdown("### ✨ Texto Normalizado e Limpo")
            if texto_resultado:
                st.success(texto_resultado)
                # Mostra a contagem de palavras puras restantes
                st.caption(f"Contagem pós-limpeza: {len(tokens_limpos)} palavras relevantes.")
            else:
                st.warning("Nenhuma palavra restou após a filtragem de pontuações.")
                
        # Área técnica expandível para inspecionar os tokens resultantes
        with st.expander("🔍 Visualizar lista de tokens limpos (Array para IA)"):
            st.write(tokens_limpos)
            
    else:
        st.warning("Por favor, digite algum texto antes de iniciar a limpeza.")