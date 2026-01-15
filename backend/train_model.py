import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import os

# Download stopwords if not present
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

def train():
    print("Loading dataset...")
    # Dataset path relative to this script: ../../../training.1600000.processed.noemoticon.csv
    # Adjust path if running from a different directory, but assuming running from 'backend' dir or root.
    # Let's verify file exists
    dataset_path = "../../../training.1600000.processed.noemoticon.csv"
    if not os.path.exists(dataset_path):
        # Fallback absolute path based on user context provided earlier if relative fails
        dataset_path = "c:/University/6th/AI/AI Lab/training.1600000.processed.noemoticon.csv"
        
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Load subset for speed (e.g., 50k rows)
    df = pd.read_csv(dataset_path, encoding='latin-1', header=None, nrows=50000)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Target: 0 = negative, 4 = positive -> 0 and 1
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    print("Cleaning text...", flush=True)
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Vectorizing...", flush=True)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['target']
    
    print("Training LightGBM model...", flush=True)
    # No splitting needed for final model artifact, but good practice to ensure it works.
    # We will train on all 100k for the app.
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)
    
    print("Saving artifacts...", flush=True)
    joblib.dump(model, 'classifier.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')
    print("Done!", flush=True)

if __name__ == "__main__":
    train()
