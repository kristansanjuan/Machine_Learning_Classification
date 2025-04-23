from utils.load_data import load_texts_and_labels
from utils.preprocess import preprocess_texts
from utils.train_models import train_all_models
from vectorize import vectorize_texts
from sklearn.model_selection import train_test_split

# Load data
texts, labels = load_texts_and_labels()

# Preprocess data
texts_cleaned = preprocess_texts(texts)

# Vectorize
X, vectorizer = vectorize_texts(texts_cleaned)
y = labels

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
train_all_models(X_train, X_test, y_train, y_test)