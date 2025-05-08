import time
import pandas as pd
import os

from utils.load_data import load_texts_and_labels
from utils.train_models import train_all_models
from sklearn.model_selection import train_test_split
from utils.visualize_data import visualize
from model_tuning import tune_models
from utils.preprocess import parallel_preprocess


def main():

    #clear terminal for better readability
    os.system("cls||clear")
    #startt timer for total time
    total_start = time.time()

    # Load data
    texts, labels = load_texts_and_labels()

    #start timer for preprocessing
    start = time.time()
    # Preprocess data
    texts_cleaned, _ = parallel_preprocess(texts, enable_spell_check=False)
    end = time.time()
    #print preprocessing time
    print(f"Preprocessing time: {end - start:.2f} seconds")

    visualize(texts_cleaned, labels)
    
    # Vectorize data
    from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE
    )
    X = vectorizer.fit_transform(texts_cleaned)

    # Split data
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Train models
    baseline_results = train_all_models(X_train_tf, X_test_tf, y_train_tf, y_test_tf)

    # Tune models
    tuned_results = tune_models(X_train_tf, y_train_tf, X_test_tf, y_test_tf)

    # Combine and show
    all_results = baseline_results + tuned_results
    df = pd.DataFrame(all_results)
    baseline_df = pd.DataFrame(baseline_results)
    tuned_df = pd.DataFrame(tuned_results)

    merged_df = pd.merge(
        baseline_df, tuned_df, on="Model", suffixes=(" (Baseline)", " (Tuned)")
    )

    # Display the comparison
    print("\n📊 Comparison of Baseline vs Tuned Models:")
    print(merged_df.sort_values(by='Accuracy (Tuned)', ascending=False))

    # Save results to CSV
    print("\nResults saved to model_comparison.csv")
    df.to_csv("model_comparison.csv", index=False)

    total_end = time.time()
    #print total time
    print(f"Total time: {total_end - total_start:.2f} seconds")

if __name__ == "__main__":
    # Check if the script is being run directly
    main()