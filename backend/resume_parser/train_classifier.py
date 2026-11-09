import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

print("Loading balanced dataset...")
df = pd.read_csv('resume_parser/data/resume_dataset.csv')

# Stratified split ensures exact class distribution in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        min_df=2
    )),
    ('clf', SVC(kernel='linear', probability=True))
])

param_grid = {
    'clf__C': [0.2, 0.5, 1.0, 2.0, 4.0]
}

print("Running 5-fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='f1_macro', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Optimal Regularization Parameter: {grid_search.best_params_}")

predictions = best_model.predict(X_test)
print("\n--- Optimized Model Evaluation ---")
print(classification_report(y_test, predictions, digits=3))

os.makedirs("resume_parser/models", exist_ok=True)
joblib.dump(best_model, "resume_parser/models/section_classifier.joblib")
print("Retrained classifier saved successfully.")