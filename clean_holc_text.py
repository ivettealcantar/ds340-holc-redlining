import json
import pandas as pd

json_path = "ad_data.json"
output_path = "clean_holc_text.csv"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

text_cols = [
    "description_of_terrain",
    "favorable_influences",
    "detrimental_influences",
    "foreign_born_nationality",
    "occupation_or_type",
    "infiltration_of",
]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

df["combined_text"] = (
    df[text_cols]
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

keep_cols = [
    "area_id",
    "city",
    "state",
    "grade",
    "combined_text",
] + text_cols

clean_df = df[keep_cols].copy()

clean_df = clean_df.dropna(subset=["grade"])
clean_df = clean_df[clean_df["combined_text"].str.len() > 0]
clean_df = clean_df.drop_duplicates(subset=["area_id"])

print("Final shape:", clean_df.shape)
print("Missing grades:", clean_df["grade"].isna().sum())
print("Duplicate area_id values:", clean_df["area_id"].duplicated().sum())
print("Blank combined_text rows:", (clean_df["combined_text"].str.len() == 0).sum())
print(clean_df["grade"].value_counts(dropna=False))

clean_df.to_csv(output_path, index=False)
print("Saved cleaned file to:", output_path)

