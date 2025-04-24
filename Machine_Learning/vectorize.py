from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    print(f"Shape of TF-IDF matrix: {X.shape}")
    return X, vectorizer