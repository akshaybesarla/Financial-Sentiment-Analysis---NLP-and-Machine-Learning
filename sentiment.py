import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the serialised model and vectorizer (must be in the same folder)
model      = joblib.load("RFC.pkl")
vectorizer = joblib.load("tfidf.pkl")
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[""…]', '', text)
    text = re.sub(r'\n', '', text)
    tokens    = word_tokenize(text)
    tokens    = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def app():
    st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="💹")
    st.title('💹 Financial Sentiment Analysis')
    st.write('Enter a financial statement and the model will classify its sentiment.')

    user_input = st.text_input('Enter your sentence:')

    if user_input:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(vectorizer.transform([preprocessed_input]))

        if prediction == 1:
            st.success('✅  Positive Sentiment')
        elif prediction == -1:
            st.error('🔴  Negative Sentiment')
        else:
            st.info('⚪  Neutral Sentiment')

if __name__ == '__main__':
    app()
