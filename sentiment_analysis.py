import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
import numpy as np

# Load dataset
data = pd.read_csv("movie_reviews.csv")  # Replace with your file path

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# Data Preprocessing Function
def preprocess_text(text):
    # Tokenization, stopword removal, stemming, and lemmatization
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Apply preprocessing
data["processed_review"] = data["review"].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
data["encoded_sentiment"] = label_encoder.fit_transform(data["sentiment"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(data["processed_review"], data["encoded_sentiment"], test_size=0.2, random_state=42)

### Feature Extraction Methods ###

# 1. Bag of Words
vectorizer_bow = CountVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# 2. N-grams (unigrams and bigrams)
vectorizer_ngram = CountVectorizer(ngram_range=(1, 2))
X_train_ngram = vectorizer_ngram.fit_transform(X_train)
X_test_ngram = vectorizer_ngram.transform(X_test)

# 3. TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# 4. Word Embeddings (Word2Vec)
tokens = [review.split() for review in data["processed_review"]]
word2vec_model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, sg=1)  # Skip-Gram model
def get_word2vec_embedding(tokens, model, vector_size):
    embeddings = np.zeros(vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            embeddings += model.wv[word]
            count += 1
    return embeddings / count if count > 0 else embeddings

X_train_w2v = np.array([get_word2vec_embedding(review.split(), word2vec_model, 100) for review in X_train])
X_test_w2v = np.array([get_word2vec_embedding(review.split(), word2vec_model, 100) for review in X_test])

### Model Training and Evaluation Function ###

def train_and_evaluate(X_train, X_test, y_train, y_test, description):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{description} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

### Train and Evaluate Using Different Feature Extraction Methods ###

# Bag of Words
train_and_evaluate(X_train_bow, X_test_bow, y_train, y_test, "Bag of Words")

# N-grams
train_and_evaluate(X_train_ngram, X_test_ngram, y_train, y_test, "N-grams")

# TF-IDF
train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF")

# Word2Vec Embeddings
train_and_evaluate(X_train_w2v, X_test_w2v, y_train, y_test, "Word2Vec Embeddings")
