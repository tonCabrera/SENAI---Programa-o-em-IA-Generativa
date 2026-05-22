import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

# Garante o download seguro e performático dos recursos do NLTK em produção
@st.cache_resource
def inicializar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.error(f"Erro ao inicializar o NLTK: {e}")

inicializar_recursos_nltk()

# Configuração de Layout e Identidade da Página
st.set_page_config(page_title="Mineração de Texto - Reclamações", page_icon="📊", layout="wide")

st.title("📊 Painel de Mineração de Texto (Text Mining)")
st.subheader("Auditoria automatizada e inteligência de produto sobre feedbacks de clientes")

# Texto de exemplo longo e realista simulando o compilado de reclamações extraídas do banco
exemplo_reclamacoes = """
O aplicativo está apresentando muita lentidão na tela de carregamento. O sistema simplesmente trava.
Ontem sofri com um travamento severo ao tentar fechar uma compra, gerou um bug no carrinho.
Sinto que o app ficou pesado. Esse bug do carrinho gerou muito atraso no meu pedido.
A empresa precisa corrigir esse travamento urgente, a lentidão está insuportável e o atraso no suporte só piora.
Mais um bug encontrado: lentidão extrema ao atualizar o perfil. O suporte técnico está em atraso.
"""

# Input: Área de texto para o analista colar o bloco consolidado de reclamações
texto_entrada = st.text_area(
    "Insira ou cole o bloco de reclamações dos clientes para auditoria:",
    value=exemplo_reclamacoes.strip(),
    height=200
)

# Botão de processamento do pipeline de dados
if st.button("Minerar Texto e Gerar Insights"):
    if texto_entrada.strip():
        
        # 1. Pipeline de PLN: Tokenização através do NLTK
        tokens_brutos = word_tokenize(texto_entrada, language='portuguese')
        
        # 2. Pipeline de Limpeza: Normalização (lowercase) e remoção de pontuações/caracteres (isalnum)
        palavras_limpas = [
            token.lower() for token in tokens_brutos 
            if token.isalnum()
        ]
        
        if palavras_limpas:
            # 3. Engenharia de Recursos: Contagem estatística eficiente com Counter
            frequencia_absoluta = Counter(palavras_limpas)
            
            # 4. Estruturação de Dados: Modelagem em DataFrame do Pandas
            df_frequencia = pd.DataFrame(
                frequencia_absoluta.items(), 
                columns=['Palavra', 'Frequência']
            ).sort_values(by='Frequência', ascending=False).reset_index(drop=True)
            
            st.success("🎯 Mineração de dados textuais concluída com sucesso!")
            st.write("---")
            
            # Divisão da tela em colunas para exibição rica de métricas e gráficos
            col_tabela, col_grafico = st.columns([1, 2])
            
            with col_tabela:
                st.markdown("### 📋 Tabela Estatística de Termos")
                st.caption("Visão detalhada ordenada por recorrência absoluta.")
                # Renderiza a tabela otimizada ocupando a largura total do container
                st.dataframe(df_frequencia, use_container_width=True)
                
            with col_grafico:
                st.markdown("### 📈 Top 10 Termos Mais Citados (Gargalos de Produto)")
                st.caption("Gráfico analítico focado nos principais focos de atrito relatados.")
                
                # Seleciona o Top 10 e plota o gráfico nativo do Streamlit
                top_10 = df_frequencia.head(10)
                st.bar_chart(data=top_10, x='Palavra', y='Frequência', use_container_width=True)
                
            # Métricas rápidas de Auditoria de Dados no rodapé da página
            st.write("---")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total de Tokens Isolados", len(tokens_brutos))
            col_m2.metric("Palavras Válidas (Pós-Filtro)", len(palavras_limpas))
            col_m3.metric("Vocabulário Único Encontrado", len(frequencia_absoluta))
            
        else:
            st.error("O texto inserido não contém palavras válidas para processamento após a filtragem de pontuações.")
    else:
        st.warning("Por favor, preencha a área de texto com dados para iniciar a extração de frequência.")