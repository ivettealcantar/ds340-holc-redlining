# DS340 HOLC Redlining Project

This project studies whether historical HOLC neighborhood descriptions predict HOLC redlining grades and whether worse historical HOLC exposure is associated with worse present-day tract-level outcomes. The workflow combines NLP classification, geospatial merging, and tract-level regression analysis.

## Research Question

- Can the language in HOLC area descriptions predict historical HOLC grades?
- Are worse historical HOLC grades associated with worse present-day tract-level disadvantage?

## Project Pipeline

1. Clean HOLC area description text from `ad_data.json`
2. Train text classification models to predict HOLC grade
3. Merge cleaned HOLC data with HOLC geometry
4. Link HOLC areas to 2020 census tracts using the tract crosswalk
5. Merge tract-linked HOLC data with CDC PLACES tract-level health outcomes
6. Aggregate overlap-level data to one row per tract
7. Run tract-level regressions using historical HOLC exposure measures

## Main Models

### Classification

- TF-IDF + Logistic Regression baseline
- TF-IDF + balanced Logistic Regression
- Keyword features + Logistic Regression
- TF-IDF + Random Forest
- BERT-style sentence embeddings + Logistic Regression

### Regression

- Linear regression predicting PLACES outcomes from `weighted_grade_score`
- Exploratory keyword-exposure regressions at the tract level

## Main Data Files

- `ad_data.json`: raw HOLC area descriptions
- `clean_holc_text.csv`: cleaned HOLC text dataset with one row per HOLC area
- `holc_text_geo.geojson`: cleaned HOLC text merged with HOLC geometry
- `holc_2020_tract_crosswalk_matched.geojson`: HOLC-to-tract overlap file restricted to areas with cleaned text
- `holc_places_tract_merged.csv`: overlap-level HOLC + PLACES merged dataset
- `tract_level_analysis_dataset.csv`: one-row-per-tract analysis dataset
- `tract_keyword_exposure_dataset.csv`: tract-level dataset with keyword exposure variables

## Key Scripts

### Text Cleaning and Classification

- `clean_holc_text.py`: cleans HOLC text and creates `clean_holc_text.csv`
- `baseline_grade_model.py`: baseline TF-IDF + Logistic Regression model
- `improved_grade_model.py`: balanced TF-IDF + Logistic Regression model
- `keyword_grade_model.py`: keyword-based Logistic Regression model
- `random_forest_grade_model.py`: Random Forest classifier on TF-IDF features
- `bert_grade_model.py`: BERT-style sentence embedding classifier

### Tables and NLP Outputs

- `export_top_words.py`: exports top predictive words by grade
- `make_model_comparison_table.py`: combines model summaries into one comparison table
- `pretty_bert_classification_table.py`: formats the BERT classification report
- `pretty_keyword_classification_table.py`: formats the keyword classification report

### Geospatial Merge and Regression

- `merge_holc_text_geo.py`: merges cleaned HOLC text with HOLC geometry
- `merge_holc_census_tracts.py`: links HOLC areas to 2020 census tracts
- `merge_places.py`: merges tract-linked HOLC data with CDC PLACES
- `make_tract_level_dataset.py`: aggregates overlap data to one row per tract
- `make_tract_keyword_exposure_dataset.py`: constructs tract-level keyword exposure variables
- `run_places_regression.py`: runs PLACES regressions using `weighted_grade_score`
- `run_keyword_exposure_regression.py`: runs exploratory keyword-exposure regressions

## Main Outputs

- `model_comparison_table.csv`
- `baseline_classification_report.csv`
- `improved_classification_report.csv`
- `keyword_classification_report.csv`
- `random_forest_classification_report.csv`
- `bert_classification_report.csv`
- `confusion_matrix.csv`
- `top_predictive_words_by_grade.csv`
- `regression_results_table.csv`
- `keyword_exposure_regression_results.csv`

## Setup

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Additional packages used by this project but not currently listed in `requirements.txt` include:

```bash
python3 -m pip install geopandas matplotlib sentence-transformers
```

## Notes

- Several generated GeoJSON files in this repository are large.
- The tract-level regressions are baseline association models.
- `merge_places.py` currently points to a local PLACES CSV path and may need that path updated before rerunning on another machine.
