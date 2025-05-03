import argparse

from utils.load_data import load_texts_and_labels
from utils.preprocess import preprocess_texts
from utils.train_models import train_all_models, train_and_evaluate #, train_doc2vec
from vectorize import vectorize_texts
from sklearn.model_selection import train_test_split
from utils.visualize_data import visualize
from model_tuning import tune_models
from model_tuning import tune_models
import pandas as pd

def main():

    # Load data
    texts, labels = load_texts_and_labels()

    # Preprocess data
    texts_cleaned = preprocess_texts(texts)

    visualize(texts_cleaned, labels)
    
    # Vectorize
    X_tfidf, vectorizer = vectorize_texts(texts_cleaned)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts_cleaned)

    # Split
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X_tfidf, labels, test_size=0.2, random_state=42
    )

    # Train models
    baseline_results = train_all_models(X_train_tf, X_test_tf, y_train_tf, y_test_tf)

    # Tune models
    tuned_results = tune_models(X_train_tf, y_train_tf, X_test_tf, y_test_tf)

    # Combine and show
    all_results = baseline_results + tuned_results
    df = pd.DataFrame(all_results)
    print("\n📊 Comparison of Baseline vs Tuned Models:")
    print(df)

    # Optionally save or plot
    df.to_csv("model_comparison.csv", index=False)

if __name__ == "__main__":
    main()