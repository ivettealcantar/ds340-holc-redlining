from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "baseline_classification_report.csv"
OUTPUT_PATH = BASE_DIR / "baseline_classification_report_pretty.csv"

df = pd.read_csv(INPUT_PATH)

# Keep only class rows + summary rows you care about
keep_labels = ["A", "B", "C", "D", "accuracy", "macro avg", "weighted avg"]
df = df[df["Label"].isin(keep_labels)].copy()

# Rename columns
df = df.rename(columns={
    "Label": "Class",
    "precision": "Precision",
    "recall": "Recall",
    "f1-score": "F1-score",
    "support": "Support",
})

# Make labels prettier
label_map = {
    "accuracy": "Accuracy",
    "macro avg": "Macro Avg",
    "weighted avg": "Weighted Avg",
}
df["Class"] = df["Class"].replace(label_map)

# Round numeric columns
for col in ["Precision", "Recall", "F1-score", "Support"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Precision"] = df["Precision"].round(3)
df["Recall"] = df["Recall"].round(3)
df["F1-score"] = df["F1-score"].round(3)
df["Support"] = df["Support"].round(0)

df.to_csv(OUTPUT_PATH, index=False)

print(df)
print(f"\nSaved pretty table to: {OUTPUT_PATH}")
