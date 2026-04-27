from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "overfitting_comparison.png"

summary_paths = [
    BASE_DIR / "baseline_model_summary.csv",
    BASE_DIR / "improved_model_summary.csv",
    BASE_DIR / "keyword_model_summary.csv",
    BASE_DIR / "random_forest_model_summary.csv",
    BASE_DIR / "bert_model_summary.csv",
]

frames = [pd.read_csv(path) for path in summary_paths]
df = pd.concat(frames, ignore_index=True)

plot_labels = {
    "TF-IDF": "TF-IDF LR",
    "TF-IDF + balanced weights": "Balanced TF-IDF LR",
    "Keyword features": "Keyword LR",
    "BERT-style sentence embeddings": "BERT LR",
}

df["Plot Label"] = df["Features"].replace(plot_labels)
df.loc[df["Model"] == "Random Forest", "Plot Label"] = "TF-IDF RF"

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))
bar_width = 0.38

ax.bar([i - bar_width / 2 for i in x], df["Train Accuracy"], width=bar_width, label="Train Accuracy")
ax.bar([i + bar_width / 2 for i in x], df["Accuracy"], width=bar_width, label="Test Accuracy")

ax.set_xticks(list(x))
ax.set_xticklabels(df["Plot Label"], rotation=20, ha="right")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Test Accuracy by Model")
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Saved figure to: {OUTPUT_PATH}")
