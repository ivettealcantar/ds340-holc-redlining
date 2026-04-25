from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "clean_holc_text.csv"

SUMMARY_OUTPUT_PATH = BASE_DIR / "bert_model_summary.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "bert_classification_report.csv"

WEIGHTS = "distilbert-base-uncased"


def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)


def get_tokens(text_series, tokenizer):
    return text_series.apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=256))


def pad_tokens(tokenized):
    max_len = max(len(i) for i in tokenized.values)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    return padded


def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)


def get_bert_sentence_vectors(model, padded_tokens):
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens).to(torch.int64), attention_mask=mask)
    return word_vecs[0][:, 0, :].numpy()


df = pd.read_csv(INPUT_PATH)
df["combined_text"] = df["combined_text"].fillna("").astype(str)
df = df[df["combined_text"].str.len() > 0].copy()

X_text = df["combined_text"]
y = df["grade"]

tokenizer = get_tokenizer()
tokens = get_tokens(X_text, tokenizer)
padded = pad_tokens(tokens)

model = get_model()
vecs = get_bert_sentence_vectors(model, padded)

train_features, test_features, train_labels, test_labels = train_test_split(
    vecs,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

clf = LogisticRegression(max_iter=3000, class_weight="balanced")
clf.fit(train_features, train_labels)

preds = clf.predict(test_features)

accuracy = accuracy_score(test_labels, preds)
report_dict = classification_report(test_labels, preds, output_dict=True)

summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Features": "DistilBERT embeddings",
        "Accuracy": accuracy,
        "Weights": WEIGHTS,
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
print(classification_report(test_labels, preds))
print(f"\nSaved summary to: {SUMMARY_OUTPUT_PATH}")
print(f"Saved classification report to: {REPORT_OUTPUT_PATH}")

