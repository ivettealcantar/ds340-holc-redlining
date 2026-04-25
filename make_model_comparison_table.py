from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

baseline_path = BASE_DIR / "baseline_model_summary.csv"
improved_path = BASE_DIR / "improved_model_summary.csv"
keyword_path = BASE_DIR / "keyword_model_summary.csv"
random_forest_path = BASE_DIR / "random_forest_model_summary.csv"
bert_path = BASE_DIR / "bert_model_summary.csv"

output_path = BASE_DIR / "model_comparison_table.csv"

baseline_df = pd.read_csv(baseline_path)
improved_df = pd.read_csv(improved_path)
keyword_df = pd.read_csv(keyword_path)
random_forest_df = pd.read_csv(random_forest_path)
bert_df = pd.read_csv(bert_path)

combined = pd.concat(
    [baseline_df, improved_df, keyword_df, random_forest_df, bert_df],
    ignore_index=True
)

combined = combined[["Model", "Features", "Accuracy"]].copy()
combined["Accuracy"] = combined["Accuracy"].round(3)

combined.to_csv(output_path, index=False)

print(combined)
print(f"\nSaved model comparison table to: {output_path}")
