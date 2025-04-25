import argparse

from utils.load_data import load_texts_and_labels
from utils.preprocess import preprocess_texts
from utils.train_models import train_all_models, train_and_evaluate #, train_doc2vec
from vectorize import vectorize_texts
from sklearn.model_selection import train_test_split
from utils.visualize_data import visualize

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

    # X_doc2vec, y_doc2vec = train_doc2vec(texts_cleaned, labels)
    # X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v = train_test_split(X_doc2vec, y_doc2vec, test_size=0.2, random_state=42)
    # train_and_evaluate(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v, sorted(list(set(labels))))

    # Split
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X_tfidf, labels, test_size=0.2, random_state=42
    )

    # Train models
    train_all_models(X_train_tf, X_test_tf, y_train_tf, y_test_tf)

if __name__ == "__main__":
    main()