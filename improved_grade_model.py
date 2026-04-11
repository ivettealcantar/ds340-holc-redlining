import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "clean_holc_text.csv"
MAX_FEATURES = 5000
TOP_N_WORDS = 10


def print_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    print("\nConfusion Matrix:\n")
    print(matrix_df)


def print_top_words(model, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop Predictive Words By Grade:\n")
    for class_index, label in enumerate(model.classes_):
        coefs = model.coef_[class_index]
        top_indices = coefs.argsort()[-top_n:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"{label}: {', '.join(top_words)}")


# Load cleaned text data
df = pd.read_csv(DATA_PATH)

X = df["combined_text"]
y = df["grade"]
labels = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print_confusion_matrix(y_test, y_pred, labels)
print_top_words(model, vectorizer, top_n=TOP_N_WORDS)
