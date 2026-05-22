import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# Garante o download seguro dos recursos necessários para a tokenização do NLTK
@st.cache_resource
def carregar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao baixar recursos do NLTK: {e}")

carregar_recursos_nltk()

# Configuração da página do Streamlit
st.set_page_config(page_title="Classificador de Setor", page_icon="📁")

st.title("📁 Classificador Automático de Chamados")
st.subheader("Direcione mensagens automaticamente para Suporte Técnico ou Financeiro")

# 1. Definição das palavras-chave específicas de cada setor (Léxico de Categorias)
TERMOS_TECNICOS = {"sistema", "erro", "bug", "senha", "login", "trava", "lento", "computador", "aplicativo", "app", "site"}
TERMOS_FINANCEIROS = {"boleto", "pagamento", "fatura", "cobrança", "cobranca", "cartão", "cartao", "reembolso", "preço", "valor", "pago"}

# Mensagem de exemplo simulando um problema financeiro
exemplo_chamado = "Não consigo baixar a fatura do meu plano e o boleto está com o valor errado."

# Input do usuário: Texto da mensagem do cliente
mensagem_entrada = st.text_area(
    "Digite ou cole a mensagem enviada pelo cliente:",
    value=exemplo_chamado,
    height=120
)

# Botão para processar e classificar
if st.button("Classificar Mensagem"):
    if mensagem_entrada.strip():
        
        # 2. Tokenização com NLTK para isolar as palavras da pontuação
        tokens = word_tokenize(mensagem_entrada, language='portuguese')
        
        # Inicializa os contadores (regras condicionais baseadas em acúmulo de pontos)
        pontos_tecnico = 0
        pontos_financeiro = 0
        
        # Listas para rastrear quais termos acionaram os pontos
        palavras_tech_encontradas = []
        palavras_fin_encontradas = []
        
        # 3. Regras condicionais simples varrendo os tokens do texto (normalizados em minúsculas)
        for token in tokens:
            palavra = token.lower()
            
            if palavra in TERMOS_TECNICOS:
                pontos_tecnico += 1
                palavras_tech_encontradas.append(palavra)
            elif palavra in TERMOS_FINANCEIROS:
                pontos_financeiro += 1
                palavras_fin_encontradas.append(palavra)
        
        st.write("---")
        st.write("### 🏷️ Resultado da Classificação")
        
        # 4. Decisão condicional final baseada no score de cada setor
        if pontos_tecnico > pontos_financeiro:
            st.warning("🛠️ Encaminhado para: **SUPORTE TÉCNICO**")
            st.write("A mensagem possui forte viés relacionado a problemas em sistemas, usabilidade ou infraestrutura.")
            
        elif pontos_financeiro > pontos_tecnico:
            st.success("💰 Encaminhado para: **FINANCEIRO**")
            st.write("A mensagem possui forte viés relacionado a faturamento, meios de pagamento ou cobranças.")
            
        else:
            # Caso haja empate (ex: 1 termo técnico e 1 financeiro) ou nenhum termo seja encontrado
            st.info("❓ Encaminhado para: **ATENDIMENTO GERAL / TRIAGEM HUMANA**")
            st.write("Não foi possível determinar um padrão claro ou houve um equilíbrio exato entre os setores.")
            
        # Exibição analítica dos termos encontrados para fins de auditoria interna
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Score Técnico", value=pontos_tecnico)
            if palavras_tech_encontradas:
                st.caption(f"Termos: {', '.join(set(palavras_tech_encontradas))}")
        with col2:
            st.metric(label="Score Financeiro", value=pontos_financeiro)
            if palavras_fin_encontradas:
                st.caption(f"Termos: {', '.join(set(palavras_fin_encontradas))}")
                
    else:
        st.warning("Por favor, digite uma mensagem para executar a classificação por regras.")