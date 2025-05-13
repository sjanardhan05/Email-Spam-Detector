import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Initialize PorterStemmer
ps = PorterStemmer()

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Select relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Map labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess text data
df['transformed_text'] = df['text'].apply(transform_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['transformed_text'], df['label'], test_size=0.2, random_state=42
)

# Initialize and fit TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the TF-IDF vectorizer and model
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("Model and vectorizer saved successfully!")
