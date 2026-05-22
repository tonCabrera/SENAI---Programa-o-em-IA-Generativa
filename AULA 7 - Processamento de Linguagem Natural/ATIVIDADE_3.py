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
st.set_page_config(page_title="Triagem de Suporte", page_icon="🚨")

st.title("🚨 Sistema de Triagem e Priorização de Suporte")
st.subheader("Detecte mensagens críticas automaticamente através de regras de PLN")

# 1. Definição do conjunto (set) de palavras-chave críticas para busca otimizada
PALAVRAS_CRITICAS = {
    "ruim", "péssimo", "pessimo", "erro", "horrível", "horrivel", 
    "defeito", "cancelar", "pior", "atraso", "reclamação", "reclamacao"
}

# Exemplo padrão contendo gatilhos para teste inicial
exemplo_critico = "O sistema apresentou um erro terrível na finalização do pagamento. O serviço está muito ruim e pretendo cancelar minha assinatura imediatamente se não resolverem."

# Input do usuário: Mensagem do cliente a ser avaliada
mensagem_usuario = st.text_area(
    "Digite ou cole a mensagem do cliente para análise:",
    value=exemplo_critico,
    height=150
)

# Botão para executar a análise de prioridade
if st.button("Analisar Prioridade do Chamado"):
    if mensagem_usuario.strip():
        
        # 2. Tokenização do texto utilizando o NLTK para isolar as palavras das pontuações
        tokens = word_tokenize(mensagem_usuario, language='portuguese')
        
        # 3. Varredura condicional convertendo os tokens para minúsculo
        # Captura as palavras críticas encontradas mantendo apenas uma ocorrência de cada (set intersection)
        palavras_detectadas = {token.lower() for token in tokens if token.lower() in PALAVRAS_CRITICAS}
        
        # 4. Regra Condicional para definição da prioridade com base nos gatilhos encontrados
        if palavras_detectadas:
            # ALTA PRIORIDADE: Mensagem contém termos críticos
            st.error("⚠️ **ALTA PRIORIDADE DETECTADA**")
            st.write("Esta mensagem foi classificada como urgente e deve ser enviada para a fila de prioridade máxima.")
            
            # Exibe quais gatilhos acionaram o alerta
            st.markdown(f"**Gatilhos encontrados:** {', '.join([f'`{p}`' for p in palavras_detectadas])}")
            
            # 5. Destaca visualmente as palavras críticas no texto original
            texto_destacado = mensagem_usuario
            for palavra in palavras_detectadas:
                # Substituição simples para fins de exibição em Markdown (adiciona negrito e caixa alta)
                import re
                padrao = re.compile(re.escape(palavra), re.IGNORECASE)
                texto_destacado = padrao.sub(f"💥**{palavra.upper()}**💥", texto_destacado)
                
            st.write("---")
            st.write("**Visualização do Texto com Destaques:**")
            st.write(texto_destacado)
            
        else:
            # PRIORIDADE NORMAL: Nenhuma palavra do dicionário crítico foi localizada
            st.success("✅ **PRIORIDADE NORMAL**")
            st.write("Nenhum termo de alta criticidade foi identificado. O chamado pode seguir o fluxo padrão de atendimento.")
            
    else:
        st.warning("Por favor, insira o texto de uma mensagem para realizar a triagem.")