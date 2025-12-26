import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


## load the model
model=load_model('simple_rnn_imbd.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter the movie review to classifiy it as positive or negative.')

## user input

user_input=st.text_area('Movie review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    #make prd
    prediction=model.predict(preprocess_input)
    sentiment='Postive' if prediction>0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'prediction score: {prediction[0][0]}')
else:
    st.write("please enter a moview review.")

