import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from tensorflow.keras import backend as K

# Clear previous sessions to avoid Keras state issues
K.clear_session()

# Load IMDb word index dictionary once
word_index = imdb.get_word_index()

# ✅ Cache model load to avoid reinitializing on rerun
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return load_model('SimpleRNN_IMDB_Model.h5')

model = load_sentiment_model()

# Title and intro
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter a movie review and the model will predict if the sentiment is **Positive** or **Negative**.")

# Function to preprocess input
def preprocess_input(sentence, word_index, maxlen=500):
    if isinstance(sentence, str):
        sentence = [sentence]
    processed_sequences = []

    for sent in sentence:
        tokens = sent.lower().split()
        sequence = [word_index.get(word, 2) for word in tokens]  # 2 = <UNK>
        processed_sequences.append(sequence)

    return pad_sequences(processed_sequences, maxlen=maxlen, padding='post', truncating='post')

# Function to predict sentiment
def prediction(text):
    processed_text = preprocess_input(text, word_index)
    predict = model.predict(processed_text)[0][0]
    return "🌟 Positive" if predict > 0.5 else "😞 Negative"

# User input
user_input = st.text_area("📝 Enter your movie review here:", height=150)

# Prediction trigger
if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        result = prediction(user_input)
        st.success(f"**Prediction:** {result}")
