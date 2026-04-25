from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "tract_level_analysis_dataset.csv"
OUTPUT_PATH = BASE_DIR / "regression_results_table.csv"

df = pd.read_csv(INPUT_PATH)

outcomes = [
    "depression_prev",
    "diabetes_prev",
    "obesity_prev",
    "access2_prev",
]

results = []

for outcome in outcomes:
    model_df = df[["weighted_grade_score", outcome]].dropna().copy()

    X = model_df[["weighted_grade_score"]]
    y = model_df[outcome]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "Outcome": outcome,
        "Rows_Used": len(model_df),
        "Coefficient": model.coef_[0],
        "Intercept": model.intercept_,
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(results_df)
print(f"\nSaved results to: {OUTPUT_PATH}")
