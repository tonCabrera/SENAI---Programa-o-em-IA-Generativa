import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Garante o download seguro dos recursos necessários para tokenização e stopwords
@st.cache_resource
def carregar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        st.error(f"Erro ao baixar recursos do NLTK: {e}")

carregar_recursos_nltk()

# Configuração da página do Streamlit
st.set_page_config(page_title="Filtro de Stopwords", page_icon="🧹")

st.title("🧹 Limpeza de Texto: Remoção de Stopwords")
st.subheader("Remova palavras vazias para revelar as palavras que realmente importam")

# Texto de exemplo padrão para o usuário testar
exemplo_texto = "O aplicativo é muito bom para o gerenciamento de tarefas do dia a dia, mas eu acho que a interface poderia ser mais limpa para os usuários."

# Input do usuário: Área de texto
texto_original = st.text_area(
    "Digite ou cole o texto que deseja limpar:",
    value=exemplo_texto,
    height=120
)

# Botão para executar a limpeza
if st.button("Remover Stopwords"):
    if texto_original.strip():
        
        # 1. Carrega a lista oficial de stopwords em português do NLTK
        palavras_vazias = set(stopwords.words('portuguese'))
        
        # 2. Tokeniza o texto original em palavras individuais
        tokens_originais = word_tokenize(texto_original, language='portuguese')
        
        # 3. Aplica o filtro: mantém a palavra apenas se ela NÃO estiver na lista de stopwords
        # Também removemos pontuações avulsas com .isalnum() para limpar ainda mais o texto
        tokens_filtrados = [
            token for token in tokens_originais 
            if token.lower() not in palavras_vazias and token.isalnum()
        ]
        
        # 4. Reconstrói o texto limpo juntando as palavras restantes
        texto_limpo = " ".join(tokens_filtrados)
        
        # Cálculos de eficiência da limpeza
        total_antes = len(tokens_originais)
        total_depois = len(tokens_filtrados)
        reducao = ((total_antes - total_depois) / total_antes) * 100
        
        st.success("Texto limpo com sucesso!")
        
        # Exibição de métricas comparativas
        col1, col2, col3 = st.columns(3)
        col1.metric("Palavras Originais", total_antes)
        col2.metric("Palavras Após Filtro", total_depois)
        col3.metric("Redução de Volume", f"{reducao:.1f}%")
        
        # Exibição dos textos lado a lado para comparação clara
        st.write("---")
        col_esq, col_dir = st.columns(2)
        
        with col_esq:
            st.markdown("### 📄 Texto Original")
            st.info(texto_original)
            
        with col_dir:
            st.markdown("### ✨ Texto Filtrado (Apenas palavras-chave)")
            st.success(texto_limpo if texto_limpo else "Nenhuma palavra sobrou após o filtro!")
            
        # Spoiler de Dev: Mostrando os bastidores do NLTK
        with st.expander("🔍 Ver exemplos de Stopwords que foram removidas neste processo"):
            stopwords_encontradas = set([t.lower() for t in tokens_originais if t.lower() in palavras_vazias])
            st.write(list(stopwords_encontradas))
            
    else:
        st.warning("Por favor, digite algum texto para aplicar o filtro.")