from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "holc_places_tract_merged.csv"
OUTPUT_PATH = BASE_DIR / "tract_level_analysis_dataset.csv"

df = pd.read_csv(INPUT_PATH, low_memory=False)

grade_map = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
}

df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
df["grade_score"] = df["grade"].map(grade_map)
df["pct_tract"] = pd.to_numeric(df["pct_tract"], errors="coerce")

# Drop rows missing key overlap or grade info for the tract summary
df_valid = df.dropna(subset=["pct_tract", "grade_score"]).copy()

# Weighted contribution of each HOLC overlap within a tract
df_valid["weighted_grade_component"] = df_valid["grade_score"] * df_valid["pct_tract"]

# Find dominant HOLC grade by largest overlap within each tract
dominant_idx = df_valid.groupby("GEOID")["pct_tract"].idxmax()
dominant_grades = (
    df_valid.loc[dominant_idx, ["GEOID", "grade"]]
    .rename(columns={"grade": "dominant_grade"})
)

tract_summary = df_valid.groupby("GEOID").agg(
    weighted_grade_sum=("weighted_grade_component", "sum"),
    total_overlap=("pct_tract", "sum"),
    num_holc_areas=("area_id", "nunique"),
    depression_prev=("DEPRESSION_CrudePrev", "first"),
    diabetes_prev=("DIABETES_CrudePrev", "first"),
    obesity_prev=("OBESITY_CrudePrev", "first"),
    access2_prev=("ACCESS2_CrudePrev", "first"),
    state=("state", "first"),
    city=("city", "first"),
).reset_index()

tract_summary["weighted_grade_score"] = (
    tract_summary["weighted_grade_sum"] / tract_summary["total_overlap"]
)

tract_summary = tract_summary.merge(dominant_grades, on="GEOID", how="left")

tract_summary = tract_summary[
    [
        "GEOID",
        "state",
        "city",
        "num_holc_areas",
        "total_overlap",
        "weighted_grade_score",
        "dominant_grade",
        "depression_prev",
        "diabetes_prev",
        "obesity_prev",
        "access2_prev",
    ]
]

print("Original input shape:", df.shape)
print("Valid overlap rows used:", df_valid.shape)
print("Tract-level shape:", tract_summary.shape)
print("Missing depression values:", tract_summary["depression_prev"].isna().sum())
print("Missing diabetes values:", tract_summary["diabetes_prev"].isna().sum())
print("Missing obesity values:", tract_summary["obesity_prev"].isna().sum())
print("Missing access2 values:", tract_summary["access2_prev"].isna().sum())

print("\nSample rows:\n")
print(tract_summary.head())

tract_summary.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved tract-level dataset to: {OUTPUT_PATH}")

