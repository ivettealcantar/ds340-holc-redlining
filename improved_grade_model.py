from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "clean_holc_text.csv"

SUMMARY_OUTPUT_PATH = BASE_DIR / "improved_model_summary.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "improved_classification_report.csv"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.csv"

MAX_FEATURES = 5000
TOP_N_WORDS = 10


def print_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    print("\nConfusion Matrix:\n")
    print(matrix_df)
    matrix_df.to_csv(CONFUSION_MATRIX_PATH)
    print(f"\nSaved confusion matrix to: {CONFUSION_MATRIX_PATH}")


def print_top_words(model, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop Predictive Words By Grade:\n")
    for class_index, label in enumerate(model.classes_):
        coefs = model.coef_[class_index]
        top_indices = coefs.argsort()[-top_n:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"{label}: {', '.join(top_words)}")


df = pd.read_csv(DATA_PATH)

X = df["combined_text"]
y = df["grade"]
labels = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

y_train_pred = model.predict(X_train_tfidf)
y_pred = model.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Features": "TF-IDF + balanced weights",
        "Train Accuracy": train_accuracy,
        "Accuracy": accuracy,
        "Max Features": MAX_FEATURES,
        "Stop Words": "english",
        "Max Iter": 2000,
        "Class Weight": "balanced",
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

print_confusion_matrix(y_test, y_pred, labels)
print_top_words(model, vectorizer, top_n=TOP_N_WORDS)

print(f"\nSaved summary to: {SUMMARY_OUTPUT_PATH}")
print(f"Saved classification report to: {REPORT_OUTPUT_PATH}")
