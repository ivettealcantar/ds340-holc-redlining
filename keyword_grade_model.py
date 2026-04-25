import re
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "clean_holc_text.csv"

SUMMARY_OUTPUT_PATH = BASE_DIR / "keyword_model_summary.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "keyword_classification_report.csv"

KEYWORD_GROUPS = {
    "race_ethnicity": [
        "negro", "negroes", "mexican", "mexicans", "oriental", "orientals",
        "hebrew", "hebrews", "italian", "italians", "jewish", "alien", "aliens", "white"
    ],
    "class_status": [
        "upper", "middle", "working", "lower", "class", "wealthy", "poor",
        "exclusive", "desirable", "restricted"
    ],
    "occupation": [
        "executive", "executives", "professional", "professionals", "business",
        "clerical", "clerks", "mechanics", "labor", "laborers", "laboring",
        "unskilled", "skilled", "domestic"
    ],
    "environment": [
        "factory", "factories", "industrial", "industry", "smoke", "odor", "odors",
        "railroad", "railway", "traffic", "noise", "dump", "river", "golf", "lawn"
    ],
}


def count_keyword(text, keyword):
    pattern = rf"\b{re.escape(keyword)}\b"
    return len(re.findall(pattern, text))


def build_keyword_features(df):
    feature_df = pd.DataFrame(index=df.index)

    for group_name, keywords in KEYWORD_GROUPS.items():
        group_total = pd.Series(0, index=df.index, dtype="int64")

        for keyword in keywords:
            col_name = f"kw_{keyword.replace(' ', '_')}"
            feature_df[col_name] = df["combined_text"].apply(
                lambda text: count_keyword(text, keyword)
            )
            group_total += feature_df[col_name]

        feature_df[f"{group_name}_total"] = group_total
        feature_df[f"{group_name}_present"] = (group_total > 0).astype(int)

    feature_df["text_length"] = df["combined_text"].str.split().str.len()
    return feature_df


df = pd.read_csv(DATA_PATH)
df["combined_text"] = df["combined_text"].fillna("").astype(str).str.lower()

X = build_keyword_features(df)
y = df["grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=3000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Features": "Keyword features",
        "Accuracy": accuracy,
        "Max Iter": 3000,
        "Class Weight": "balanced",
        "Feature Count": X.shape[1],
    }
])

report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df = report_df.rename(columns={"index": "Label"})

summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
report_df.to_csv(REPORT_OUTPUT_PATH, index=False)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nFeature matrix shape:", X.shape)
print("\nTop keyword totals:")
print(X[[col for col in X.columns if col.endswith("_total")]].sum().sort_values(ascending=False))

print(f"\nSaved summary to: {SUMMARY_OUTPUT_PATH}")
print(f"Saved classification report to: {REPORT_OUTPUT_PATH}")
