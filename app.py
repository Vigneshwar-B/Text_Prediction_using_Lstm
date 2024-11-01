import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        /* Background styling */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1526401281622-ff5e7fdc4fe3");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        /* Container styling */
        .container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
            max-width: 600px;
            margin: auto;
        }
        /* Title styling */
        h1 {
            color: #6a1b9a;
            text-align: center;
            font-family: 'Arial', sans-serif;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        /* Button styling */
        .stButton>button {
            color: white;
            background-color: #6a1b9a;
            font-size: 1rem;
            padding: 10px;
            border-radius: 5px;
        }
        /* Output styling */
        .output {
            font-size: 1.2rem;
            color: #333;
            text-align: center;
            margin-top: 1rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Container for app content
with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # Title of the app
    st.title("Next Word Prediction with LSTM")

    # User input
    input_text = st.text_input("Enter the sequence of words:", "To be or not to")

    # Predict next word on button click
    if st.button("Predict Next Word"):
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.markdown(f"<p class='output'>Next word: <b>{next_word}</b></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='output'>Next word not found in vocabulary.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
