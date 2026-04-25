from pathlib import Path

import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent
HOLC_PATH = BASE_DIR / "holc_text_geo.geojson"
CROSSWALK_PATH = BASE_DIR / "MIv3Areas_2020TractCrosswalk.geojson"
FULL_OUTPUT_PATH = BASE_DIR / "holc_2020_tract_crosswalk_enriched.geojson"
MATCHED_OUTPUT_PATH = BASE_DIR / "holc_2020_tract_crosswalk_matched.geojson"

holc_gdf = gpd.read_file(HOLC_PATH)
crosswalk_gdf = gpd.read_file(CROSSWALK_PATH)

# Keep one row per HOLC area from the text+geometry file before joining to the tract crosswalk.
holc_cols = [
    "area_id",
    "city_geo",
    "state_geo",
    "grade_geo",
    "grade_text",
    "combined_text",
    "description_of_terrain",
    "favorable_influences",
    "detrimental_influences",
    "foreign_born_nationality",
    "occupation_or_type",
    "infiltration_of",
]
holc_for_merge = holc_gdf[holc_cols].drop_duplicates(subset=["area_id"]).copy()

full_merged = crosswalk_gdf.merge(holc_for_merge, on="area_id", how="left")
matched_merged = full_merged[full_merged["combined_text"].notna()].copy()

print("HOLC text+geo shape:", holc_gdf.shape)
print("2020 tract crosswalk shape:", crosswalk_gdf.shape)
print("Full merged shape:", full_merged.shape)
print("Matched-only shape:", matched_merged.shape)
print("Rows missing HOLC text in full merge:", full_merged["combined_text"].isna().sum())
print("Unique HOLC areas in matched file:", matched_merged["area_id"].nunique())
print("Unique 2020 GEOIDs in matched file:", matched_merged["GEOID"].nunique())
print("True grade mismatches among matched rows:", (matched_merged["grade"] != matched_merged["grade_text"]).sum())

print("\nSample matched rows:\n")
print(
    matched_merged[
        [
            "area_id",
            "GEOID",
            "pct_tract",
            "city",
            "state",
            "grade",
            "grade_text",
        ]
    ].head()
)

full_merged.to_file(FULL_OUTPUT_PATH, driver="GeoJSON")
matched_merged.to_file(MATCHED_OUTPUT_PATH, driver="GeoJSON")

print(f"\nSaved full enriched crosswalk to: {FULL_OUTPUT_PATH}")
print(f"Saved matched-only analysis file to: {MATCHED_OUTPUT_PATH}")
