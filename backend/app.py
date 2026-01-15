from flask import Flask, request, jsonify, send_from_directory
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__, static_folder='../frontend')

# Load model and vectorizer
try:
    model = joblib.load('classifier.pkl')
    tfidf = joblib.load('tfidf.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tfidf = None

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    custom_stopwords = set(stopwords.words('english')) - {'no', 'not', 'nor', "don't", "can't"}
    text = " ".join(word for word in text.split() if word not in custom_stopwords)
    return text

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tfidf:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    cleaned_text = clean_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    # Probabilities
    proba = model.predict_proba(vectorized_text)[0]
    
    result = "Positive" if prediction == 1 else "Negative"
    confidence = float(proba[prediction])
    
    return jsonify({
        'sentiment': result,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
