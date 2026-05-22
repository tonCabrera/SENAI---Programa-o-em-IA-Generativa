import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Garante o download dos recursos de tokenização de forma segura e cacheada
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao carregar recursos do NLTK: {e}")

setup_nltk()

# Configurações de interface do Streamlit
st.set_page_config(page_title="Roteador de Chatbot", page_icon="🤖")

st.title("🤖 Chatbot: Roteador de Chamados")
st.markdown("""
Esta aplicação identifica palavras-chave em mensagens de clientes para direcionar o atendimento 
ao setor correto de forma automática.
""")

# Campo de entrada para o usuário
user_input = st.text_input("Olá! Sou o assistente virtual. Como posso te ajudar hoje?", 
                           placeholder="Ex: Preciso de ajuda com meu pagamento.")

# Botão para processar a mensagem
if st.button("Enviar Mensagem"):
    if user_input.strip():
        # 1. Tokenização: Transforma a frase em uma lista de palavras isoladas
        # O uso do NLTK garante que pontuações não fiquem 'coladas' nas palavras
        tokens = word_tokenize(user_input, language='portuguese')
        
        # 2. Normalização: Converte todas as palavras para minúsculo para comparação precisa
        tokens_normalizados = [t.lower() for t in tokens]
        
        # Variáveis de controle para a lógica de decisão
        setor_destino = None
        gatilho_encontrado = None

        # 3. Regras de Decisão baseadas em palavras-chave
        if "cancelar" in tokens_normalizados:
            setor_destino = "Setor de Retenção e Cancelamentos"
            gatilho_encontrado = "cancelar"
            cor_box = "warning" # Amarelo/Laranja
            
        elif "erro" in tokens_normalizados:
            setor_destino = "Suporte Técnico"
            gatilho_encontrado = "erro"
            cor_box = "error" # Vermelho
            
        elif "pagamento" in tokens_normalizados:
            setor_destino = "Departamento Financeiro"
            gatilho_encontrado = "pagamento"
            cor_box = "info" # Azul
            
        else:
            setor_destino = "Atendimento Geral (Transbordo Humano)"
            cor_box = "success" # Verde

        # 4. Exibição do feedback visual para o usuário
        st.write("---")
        if gatilho_encontrado:
            # Caso algum gatilho específico tenha sido acionado
            feedback_func = getattr(st, cor_box)
            feedback_func(f"### Direcionando para: {setor_destino}")
            st.markdown(f"**Identificamos a necessidade:** `{gatilho_encontrado}`")
            st.info("Aguarde um momento enquanto conectamos você com um especialista deste setor.")
        else:
            # Caso caia no fluxo de atendimento geral
            st.success(f"### Direcionando para: {setor_destino}")
            st.markdown("Não identificamos uma categoria específica, mas não se preocupe!")
            st.write("Um de nossos atendentes irá te auxiliar em instantes.")
            
    else:
        st.warning("Por favor, digite uma mensagem para que eu possa te ajudar.")

# Rodapé simples
st.caption("Protótipo de Roteamento de Mensagens via NLTK & Streamlit.")