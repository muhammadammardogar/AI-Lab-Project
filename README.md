# 🎭 Large Scale Twitter Sentiment Analysis

A machine learning-powered web application that analyzes the sentiment of Twitter text, classifying it as **Positive** or **Negative** with a confidence score.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-orange.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Why Model Files Are Not Included](#-why-model-files-are-not-included)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)

## ✨ Features

- **Real-time Sentiment Analysis**: Instantly analyze the sentiment of any text
- **Confidence Score**: See how confident the model is about its prediction
- **Modern Web Interface**: Clean, responsive UI with gradient design
- **REST API**: Easy-to-use `/predict` endpoint for integration

## 📁 Project Structure

```
AI Lab Project/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # Script to train the ML model
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Web interface
├── Project Code.ipynb      # Jupyter notebook with full analysis
└── .gitignore
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/muhammadammardogar/AI-Lab-Project.git
cd AI-Lab-Project
```

### Step 2: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Download the Dataset

Download the **Sentiment140 dataset** (required for training):

1. Go to [Kaggle - Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Download `training.1600000.processed.noemoticon.csv`
3. Place it in the parent directory (one level above the project folder)

   **Expected path structure:**
   ```
   Parent Directory/
   ├── training.1600000.processed.noemoticon.csv  ← Place dataset here
   └── AI-Lab-Project/
       ├── backend/
       └── ...
   ```

### Step 4: Train the Model

```bash
cd backend
python train_model.py
```

This will generate two files in the `backend/` folder:
- `classifier.pkl` - The trained LightGBM model
- `tfidf.pkl` - The TF-IDF vectorizer

⏱️ **Training time**: Approximately 1-3 minutes (uses 50,000 samples for efficiency)

### Step 5: Run the Application

```bash
python app.py
```

The app will be available at: **http://localhost:5000**

## 💻 Usage

1. Open your browser and go to `http://localhost:5000`
2. Enter any text in the text area (e.g., "I love this product!")
3. Click **"Analyze Sentiment"**
4. View the result with confidence percentage

### API Endpoint

You can also use the REST API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.89
}
```

## ⚠️ Why Model Files Are Not Included

The trained model files (`classifier.pkl` and `tfidf.pkl`) are **excluded from this repository** for the following reasons:

### 1. **File Size Limitations**
   - GitHub has a file size limit of 100 MB per file
   - Large binary files bloat repository history and slow down cloning
   - Model files can be ~300 KB to several MB depending on training data

### 2. **Reproducibility**
   - Training locally ensures the model works with your Python environment
   - Different versions of scikit-learn/LightGBM may have compatibility issues with pre-trained models
   - You can verify the training process yourself

### 3. **Dataset Licensing**
   - The Sentiment140 dataset has its own terms of use
   - Redistributing derived model files may have legal implications

### 4. **Best Practice**
   - It's standard practice to include training scripts rather than binary artifacts
   - This approach is more transparent and educational

### How to Generate Model Files

Simply run the training script as described in the installation steps:

```bash
cd backend
python train_model.py
```

The script will:
1. Load the Sentiment140 dataset (50,000 samples)
2. Clean and preprocess the text
3. Create TF-IDF features (5,000 max features)
4. Train a LightGBM classifier
5. Save both `classifier.pkl` and `tfidf.pkl` to the backend folder

## 📊 Dataset

This project uses the **Sentiment140** dataset:

- **Source**: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Labels**: Negative (0) and Positive (4, converted to 1)
- **Training subset**: 50,000 samples (for faster training)

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| ML Model | LightGBM |
| Text Processing | TF-IDF, NLTK |
| Frontend | HTML, CSS, JavaScript |
| Data Handling | Pandas |

## 📝 License

This project is for educational purposes as part of an AI Lab course.

---

Made with ❤️ for AI Lab Term Project
