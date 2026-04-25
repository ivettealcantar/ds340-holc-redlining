from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "clean_holc_text.csv"

SUMMARY_OUTPUT_PATH = BASE_DIR / "bert_model_summary.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "bert_classification_report.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

df = pd.read_csv(INPUT_PATH)
df["combined_text"] = df["combined_text"].fillna("").astype(str)
df = df[df["combined_text"].str.len() > 0].copy()

X_text = df["combined_text"]
y = df["grade"]

encoder = SentenceTransformer(MODEL_NAME)
embeddings = encoder.encode(
    X_text.tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=3000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Features": "BERT-style sentence embeddings",
        "Accuracy": accuracy,
        "Embedding Model": MODEL_NAME,
        "Max Iter": 3000,
        "Class Weight": "balanced",
    }
])

report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df = report_df.rename(columns={"index": "Label"})

summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
report_df.to_csv(REPORT_OUTPUT_PATH, index=False)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print(f"\nSaved summary to: {SUMMARY_OUTPUT_PATH}")
print(f"Saved classification report to: {REPORT_OUTPUT_PATH}")
