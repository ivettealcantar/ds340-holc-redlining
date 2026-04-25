from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "regression_results_table.csv"
OUTPUT_PATH = BASE_DIR / "regression_results_table_pretty.csv"

df = pd.read_csv(INPUT_PATH)

outcome_labels = {
    "depression_prev": "Depression prevalence",
    "diabetes_prev": "Diabetes prevalence",
    "obesity_prev": "Obesity prevalence",
    "access2_prev": "Lack of health insurance",
}

df["Outcome"] = df["Outcome"].replace(outcome_labels)

df = df.rename(columns={
    "Rows_Used": "Rows Used",
    "Coefficient": "Coefficient",
    "Intercept": "Intercept",
    "MSE": "MSE",
    "R2": "R²",
})

df["Coefficient"] = df["Coefficient"].round(3)
df["Intercept"] = df["Intercept"].round(3)
df["MSE"] = df["MSE"].round(3)
df["R²"] = df["R²"].round(3)

df.to_csv(OUTPUT_PATH, index=False)

print(df)
print(f"\nSaved pretty table to: {OUTPUT_PATH}")
