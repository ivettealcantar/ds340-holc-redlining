import pandas as pd
import geopandas as gpd

text_df = pd.read_csv("clean_holc_text.csv")
geo_df = gpd.read_file("mappinginequality.json")

merged = geo_df.merge(text_df, on="area_id", how="inner", suffixes=("_geo", "_text"))

print("GeoJSON shape:", geo_df.shape)
print("Text shape:", text_df.shape)
print("Merged shape:", merged.shape)

print(merged[["area_id", "city_geo", "state_geo", "grade_geo", "grade_text"]].head())


merged.to_file("holc_text_geo.geojson", driver="GeoJSON")
print(merged.columns.tolist())
print("Grade mismatches:", (merged["grade_geo"] != merged["grade_text"]).sum())
