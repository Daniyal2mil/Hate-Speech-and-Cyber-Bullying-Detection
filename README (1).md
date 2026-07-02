# 🚫 Hate Speech & Cyberbullying Detection

An AI-powered system that detects **hate speech**, **cyberbullying**, and **neutral content** in social media text using a TF-IDF + Logistic Regression model, backed by a rule-based safety net and served through an interactive Streamlit app.

> Developed as a Natural Language Processing (CT-485) group project at NED University of Engineering & Technology.

---

## ✨ Features

- **Smart Hybrid Detection** — the trained ML model makes the primary decision; a curated keyword-based rule system only kicks in when the model's confidence is low, keeping predictions accurate while catching unambiguous edge cases.
- **3-Class Classification** — labels text as `Hate Speech`, `Cyberbullying`, or `Neutral`.
- **Interactive Web App** — a Streamlit interface for real-time text analysis, confidence visualization, and model performance stats.
- **Identical Preprocessing Pipeline** — the app reuses the exact cleaning/lemmatization logic from training, so predictions stay consistent between the notebook and the deployed app.
- **Adjustable Confidence Threshold** — tune how much the rule-based system is allowed to override the ML model directly from the sidebar.

## 🧠 How It Works

1. **Text Input** — user submits a piece of text (e.g. a tweet or comment).
2. **Cleaning** — URLs, mentions, and hashtags are stripped; text is lowercased and lemmatized with spaCy.
3. **Vectorization** — cleaned text is transformed with a TF-IDF vectorizer (unigrams to trigrams).
4. **ML Prediction** — a Logistic Regression classifier (trained with balanced class weights) predicts a class with associated probabilities.
5. **Confidence Check** — if the top probability is above the configured threshold, the ML prediction is returned as-is.
6. **Rule Assistance** — if the model is uncertain, a small set of high-severity keywords is checked to help disambiguate hate speech vs. cyberbullying.
7. **Result** — the final label, confidence scores, and (optionally) a note on whether rules assisted the decision are displayed.

## 📊 Model Performance

| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Hate Speech    | 93.8%     | 91.9%  | 92.8%    |
| Cyberbullying  | 81.5%     | 74.6%  | 77.9%    |
| Neutral        | 53.3%     | 65.8%  | 58.9%    |

**Overall Accuracy:** 82.0% &nbsp;|&nbsp; **Weighted F1-Score:** 82.5%

## 📁 Project Structure

```
Hate-Speech-And-Cyber-Bullying-Detection/
├── app.py                                    # Streamlit web application
├── Model.ipynb                                # Training notebook (data prep, TF-IDF, LogReg, evaluation)
├── labeled_data.csv                           # Hate speech / offensive language dataset
├── cyberbullying_tweets.csv                   # Cyberbullying tweets dataset
├── models/
│   ├── cyberbullying_tfidf_vectorizer.joblib  # Fitted TF-IDF vectorizer
│   ├── cyberbullying_logreg_model.joblib      # Trained Logistic Regression model
│   └── cyberbullying_rule_components.joblib   # Rule-based keyword components
└── README.md
```

## 🗂️ Datasets

- **`labeled_data.csv`** — tweets labeled as hate speech, offensive language, or neither.
- **`cyberbullying_tweets.csv`** — tweets labeled by cyberbullying type, remapped into the project's 3-class schema (`Hate Speech`, `Cyberbullying`, `Neutral`).

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, evaluation metrics
- **spaCy** (`en_core_web_sm`) — tokenization and lemmatization
- **NLTK** — stopwords
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualization (in the training notebook)
- **Streamlit** — web app interface
- **joblib** — model persistence

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Hate-Speech-And-Cyber-Bullying-Detection.git
cd Hate-Speech-And-Cyber-Bullying-Detection
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib nltk spacy matplotlib seaborn
python -m spacy download en_core_web_sm
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Pre-trained models in the `models/` folder are loaded automatically.

### 4. (Optional) Retrain the model

Open `Model.ipynb` to explore or rerun the full training pipeline — data loading, cleaning, TF-IDF vectorization, class imbalance handling, model training, and evaluation. Retrained artifacts can be saved back into `models/` for the app to pick up.

## ⚙️ App Overview

The Streamlit app is organized into three tabs:

- **🔍 Text Analysis** — enter text and get a real-time classification with confidence scores and a warning for borderline cases.
- **📊 Performance** — view model accuracy, F1-scores, and a breakdown of the smart hybrid architecture.
- **ℹ️ About** — technical details on the detection pipeline and project contributors.

## ⚠️ Limitations & Ethical Notes

- The model is trained on publicly available, English-language social media datasets and may not generalize well to other languages, dialects, or platforms.
- Automated hate speech/cyberbullying detection can reflect biases present in the training data; predictions should be reviewed by a human before any moderation action is taken.
- This project is intended for educational and research purposes, not as a production-grade content moderation system.

## 👥 Contributors

**NED University of Engineering & Technology**
Natural Language Processing (CT-485) — Group Project

- Syed Hammad Atif (CTAI-22017)
- Abdul Waqar (CTAI-22042)
- Syed Muhammad Daniyal Qadri (CTAI-22037)
- Nofal Shameem (CTAI-22039)

## 📄 License

This project is provided for academic and educational purposes. Add a license of your choice (e.g. MIT) if you plan to distribute or reuse it.
