from flask import Flask, render_template, request, url_for
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Initialize Flask app
app = Flask(__name__)

# Initialize PorterStemmer
ps = PorterStemmer()

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess and transform text
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

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['input_sms']
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        prediction = "Spam"
    else:
        prediction = "Not Spam"

    return render_template('index.html', prediction=prediction, message=input_sms)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True)
