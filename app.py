import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download punkt if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()
stop_words = ENGLISH_STOP_WORDS

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# Load trained vectorizer & model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title(" Email / SMS Spam Classifier")

input_email = st.text_area("Enter the email or SMS message")

if st.button("Predict"):
    if input_email.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_email = transform_text(input_email)
        vector_input = tfidf.transform([transformed_email])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error(" Spam Message")
        else:
            st.success(" Not Spam")
