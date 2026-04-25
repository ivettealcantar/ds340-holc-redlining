import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "holc_places_tract_merged.csv"
OUTPUT_PATH = BASE_DIR / "tract_keyword_exposure_dataset.csv"

KEYWORD_GROUPS = {
    "race_exposure": [
        "negro", "negroes", "mexican", "mexicans", "oriental", "orientals",
        "hebrew", "hebrews", "italian", "italians", "jewish", "alien", "aliens", "white"
    ],
    "class_exposure": [
        "upper", "middle", "working", "lower", "class", "wealthy", "poor",
        "exclusive", "desirable", "restricted"
    ],
    "occupation_exposure": [
        "executive", "executives", "professional", "professionals", "business",
        "clerical", "clerks", "mechanics", "labor", "laborers", "laboring",
        "unskilled", "skilled", "domestic"
    ],
    "environment_exposure": [
        "factory", "factories", "industrial", "industry", "smoke", "odor", "odors",
        "railroad", "railway", "traffic", "noise", "dump", "river", "golf", "lawn"
    ],
}


def count_keyword(text, keyword):
    pattern = rf"\b{re.escape(keyword)}\b"
    return len(re.findall(pattern, text))


df = pd.read_csv(INPUT_PATH, low_memory=False)

df["combined_text"] = df["combined_text"].fillna("").astype(str).str.lower()
df["pct_tract"] = pd.to_numeric(df["pct_tract"], errors="coerce")

# Create keyword count columns for each overlap row
for group_name, keywords in KEYWORD_GROUPS.items():
    df[group_name] = 0
    for keyword in keywords:
        df[group_name] += df["combined_text"].apply(lambda text: count_keyword(text, keyword))

# Keep only valid overlap rows
df_valid = df.dropna(subset=["pct_tract"]).copy()

# Weighted keyword exposure for each overlap row
for group_name in KEYWORD_GROUPS.keys():
    df_valid[f"{group_name}_weighted"] = df_valid[group_name] * df_valid["pct_tract"]

# Aggregate to tract level
tract_summary = df_valid.groupby("GEOID").agg(
    total_overlap=("pct_tract", "sum"),
    num_holc_areas=("area_id", "nunique"),
    race_exposure_weighted_sum=("race_exposure_weighted", "sum"),
    class_exposure_weighted_sum=("class_exposure_weighted", "sum"),
    occupation_exposure_weighted_sum=("occupation_exposure_weighted", "sum"),
    environment_exposure_weighted_sum=("environment_exposure_weighted", "sum"),
    depression_prev=("DEPRESSION_CrudePrev", "first"),
    diabetes_prev=("DIABETES_CrudePrev", "first"),
    obesity_prev=("OBESITY_CrudePrev", "first"),
    access2_prev=("ACCESS2_CrudePrev", "first"),
    state=("state", "first"),
    city=("city", "first"),
).reset_index()

# Convert weighted sums into weighted average tract-level exposures
tract_summary["race_exposure"] = tract_summary["race_exposure_weighted_sum"] / tract_summary["total_overlap"]
tract_summary["class_exposure"] = tract_summary["class_exposure_weighted_sum"] / tract_summary["total_overlap"]
tract_summary["occupation_exposure"] = tract_summary["occupation_exposure_weighted_sum"] / tract_summary["total_overlap"]
tract_summary["environment_exposure"] = tract_summary["environment_exposure_weighted_sum"] / tract_summary["total_overlap"]

tract_summary = tract_summary[
    [
        "GEOID",
        "state",
        "city",
        "num_holc_areas",
        "total_overlap",
        "race_exposure",
        "class_exposure",
        "occupation_exposure",
        "environment_exposure",
        "depression_prev",
        "diabetes_prev",
        "obesity_prev",
        "access2_prev",
    ]
]

print("Input shape:", df.shape)
print("Valid overlap rows used:", df_valid.shape)
print("Tract-level shape:", tract_summary.shape)

print("Missing depression values:", tract_summary["depression_prev"].isna().sum())
print("Missing diabetes values:", tract_summary["diabetes_prev"].isna().sum())
print("Missing obesity values:", tract_summary["obesity_prev"].isna().sum())
print("Missing access2 values:", tract_summary["access2_prev"].isna().sum())

print("\nSample rows:\n")
print(tract_summary.head())

tract_summary.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved tract keyword exposure dataset to: {OUTPUT_PATH}")
