from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "regression_results_table_pretty.csv"
OUTPUT_PATH = BASE_DIR / "places_regression_coefficients.png"

df = pd.read_csv(INPUT_PATH)

plt.figure(figsize=(10, 6))
plt.bar(df["Outcome"], df["Coefficient"], color="steelblue")

plt.title("Regression Coefficients for PLACES Health Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Coefficient on Weighted HOLC Grade Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print(f"Saved figure to: {OUTPUT_PATH}")
