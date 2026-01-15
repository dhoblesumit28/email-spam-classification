import pandas as pd
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB

# Download tokenizer
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

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Apply text transformation
df['text'] = df['text'].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model and Vectorizer trained & saved successfully!")

