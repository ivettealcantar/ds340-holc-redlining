from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

HOLC_PATH = BASE_DIR / "holc_2020_tract_crosswalk_matched.geojson"
PLACES_PATH = Path("/Users/ivettealcantar/Downloads/PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260413 (1).csv")

OUTPUT_PATH = BASE_DIR / "holc_places_tract_merged.geojson"
OUTPUT_CSV_PATH = BASE_DIR / "holc_places_tract_merged.csv"

holc = gpd.read_file(HOLC_PATH)
places = pd.read_csv(PLACES_PATH)

# Make sure tract IDs are strings with leading zeros preserved
places["GEOID"] = places["TractFIPS"].astype(str).str.zfill(11)
holc["GEOID"] = holc["GEOID"].astype(str).str.zfill(11)

# Pick a smaller set of useful PLACES columns first
places_cols = [
    "GEOID",
    "StateAbbr",
    "StateDesc",
    "CountyName",
    "CountyFIPS",
    "TractFIPS",
    "TotalPopulation",
    "TotalPop18plus",
    "DEPRESSION_CrudePrev",
    "DIABETES_CrudePrev",
    "OBESITY_CrudePrev",
    "ACCESS2_CrudePrev",
    "CHECKUP_CrudePrev",
    "CSMOKING_CrudePrev",
]

places_subset = places[places_cols].copy()

merged = holc.merge(places_subset, on="GEOID", how="left")

print("HOLC matched file shape:", holc.shape)
print("PLACES shape:", places.shape)
print("Merged shape:", merged.shape)

print("Rows missing DEPRESSION_CrudePrev:", merged["DEPRESSION_CrudePrev"].isna().sum())
print("Rows missing DIABETES_CrudePrev:", merged["DIABETES_CrudePrev"].isna().sum())
print("Unique GEOIDs in HOLC file:", holc["GEOID"].nunique())
print("Unique GEOIDs in merged file:", merged["GEOID"].nunique())

print("\nSample rows:\n")
print(
    merged[
        [
            "GEOID",
            "area_id",
            "grade",
            "pct_tract",
            "DEPRESSION_CrudePrev",
            "DIABETES_CrudePrev",
            "OBESITY_CrudePrev",
            "ACCESS2_CrudePrev",
        ]
    ].head()
)

merged.to_file(OUTPUT_PATH, driver="GeoJSON")
merged.drop(columns="geometry").to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\nSaved GeoJSON to: {OUTPUT_PATH}")
print(f"Saved CSV to: {OUTPUT_CSV_PATH}")
# AI-use note:
# Portions of this script were developed with AI assistance for code refinement,
# debugging, and parameter adjustment.
