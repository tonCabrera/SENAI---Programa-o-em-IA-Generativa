import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Garante o download seguro dos recursos necessários para a tokenização
@st.cache_resource
def carregar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao baixar recursos do NLTK: {e}")

carregar_recursos_nltk()

# Configuração da página do Streamlit
st.set_page_config(page_title="Tokenizador de Mensagens", page_icon="🔍")

st.title("🔍 Tokenizador de Mensagens de Clientes")
st.subheader("Transforme textos longos em palavras individuais para análise")

# Texto de exemplo padrão para facilitar o teste do usuário
exemplo_padrao = "Olá! Gostaria de saber o status do meu pedido #1024. O prazo de entrega já venceu e o suporte ainda não me respondeu. Aguardo retorno urgente, obrigado."

# Input do usuário: Área de texto para colar a mensagem do cliente
mensagem_cliente = st.text_area(
    "Cole a mensagem do cliente abaixo:",
    value=exemplo_padrao,
    height=150
)

# Botão para executar a tokenização
if st.button("Processar e Separar Palavras"):
    if mensagem_cliente.strip():
        # Processo de Tokenização por palavras utilizando o NLTK
        tokens = word_tokenize(mensagem_cliente, language='portuguese')
        
        # Filtra opcionalmente para contar apenas o total de elementos gerados
        total_tokens = len(tokens)
        
        # Exibição dos resultados em métricas e formato scannable
        st.success("Texto processado com sucesso!")
        
        # Cria duas colunas para organizar as métricas e o resultado
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Total de Tokens/Palavras", value=total_tokens)
            st.info("💡 Pontuações e caracteres especiais também são contados como tokens individuais na análise de PLN.")
            
        with col2:
            st.write("**Lista de Palavras Extraídas:**")
            # Exibe os tokens em um formato limpo e estruturado
            st.write(tokens)
            
            # Exibição alternativa em formato de chips de texto para melhor visualização
            st.write("**Visualização Rápida:**")
            st.caption(" | ".join(tokens))
    else:
        st.warning("Por favor, insira ou cole um texto antes de processar.")