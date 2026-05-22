import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from deep_translator import GoogleTranslator

# Garante o download do dicionário do VADER
nltk.download('vader_lexicon')

# Inicializa o analisador e o tradutor (PT para EN)
sia = SentimentIntensityAnalyzer()
translator = GoogleTranslator(source='pt', target='en')

# Configuração simples da página
st.title("Análise de Sentimentos com NLTK")
st.subheader("Digite em português e descubra o sentimento")

# Input do usuário (Prompt agora em PT)
user_text = st.text_input("Digite sua frase aqui:", placeholder="Ex: Eu adorei esse aplicativo!")

# Botão para executar
if st.button("Analisar Sentimento"):
    if user_text:
        # Traduz o texto de PT para EN antes da análise
        texto_traduzido = translator.translate(user_text)
        
        # Faz a análise no texto traduzido
        scores = sia.polarity_scores(texto_traduzido)
        compound = scores['compound']
        
        # Define a lógica do sentimento baseado no score 'compound'
        if compound >= 0.05:
            resultado = "😊 Positivo"
        elif compound <= -0.05:
            resultado = "😢 Negativo"
        else:
            resultado = "😐 Neutro"
            
        # Mostra o resultado na tela
        st.write(f"**Resultado:** O sentimento predominante é {resultado}")
    else:
        st.write("Por favor, digite algo antes de analisar!")