from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "tract_keyword_exposure_dataset.csv"
OUTPUT_PATH = BASE_DIR / "keyword_exposure_regression_results.csv"

df = pd.read_csv(INPUT_PATH)

predictors = [
    "race_exposure",
    "class_exposure",
    "occupation_exposure",
    "environment_exposure",
]

outcomes = [
    "depression_prev",
    "diabetes_prev",
    "obesity_prev",
    "access2_prev",
]

results = []

for outcome in outcomes:
    model_df = df[predictors + [outcome]].dropna().copy()

    X = model_df[predictors]
    y = model_df[outcome]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    row = {
        "Outcome": outcome,
        "Rows_Used": len(model_df),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Intercept": model.intercept_,
    }

    for predictor, coef in zip(predictors, model.coef_):
        row[f"{predictor}_coef"] = coef

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(results_df)
print(f"\nSaved keyword exposure regression results to: {OUTPUT_PATH}")
