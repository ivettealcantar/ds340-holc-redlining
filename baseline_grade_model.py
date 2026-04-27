from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "clean_holc_text.csv"
SUMMARY_OUTPUT_PATH = BASE_DIR / "baseline_model_summary.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "baseline_classification_report.csv"

df = pd.read_csv(INPUT_PATH)

X = df["combined_text"]
y = df["grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_tfidf, y_train)

y_train_pred = model.predict(X_train_tfidf)
y_pred = model.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Features": "TF-IDF",
        "Train Accuracy": train_accuracy,
        "Accuracy": accuracy,
        "Max Features": 5000,
        "Stop Words": "english",
        "Max Iter": 2000,
    }
])

report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df = report_df.rename(columns={"index": "Label"})

summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
report_df.to_csv(REPORT_OUTPUT_PATH, index=False)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print(f"\nSaved summary to: {SUMMARY_OUTPUT_PATH}")
print(f"Saved classification report to: {REPORT_OUTPUT_PATH}")
