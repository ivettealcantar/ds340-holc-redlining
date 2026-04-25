from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "top_predictive_words_by_grade.csv"
OUTPUT_PATH = BASE_DIR / "top_predictive_words_pretty.csv"

df = pd.read_csv(INPUT_PATH)

pretty_df = (
    df.groupby("Grade")["Word"]
    .apply(lambda words: ", ".join(words))
    .reset_index()
    .rename(columns={"Word": "Top Predictive Words"})
)

pretty_df.to_csv(OUTPUT_PATH, index=False)

print(pretty_df)
print(f"\nSaved pretty table to: {OUTPUT_PATH}")
