# DS340 HOLC Redlining Project

This project studies how HOLC neighborhood descriptions and redlining grades relate to historical classification patterns and, later, modern outcomes such as life expectancy.

## Current Files
- `clean_holc_text.py`: cleans the HOLC JSON area descriptions into a modeling-ready CSV
- `clean_holc_text.csv`: cleaned text dataset with one row per HOLC area
- `baseline_grade_model.py`: TF-IDF + Logistic Regression baseline for predicting HOLC grade
- `improved_grade_model.py`: balanced version with confusion matrix and top predictive words

## Setup
Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Next Steps
- Add keyword-based features
- Add BERT text embeddings
- Merge HOLC geographies with modern life expectancy data
