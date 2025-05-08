folder_paths = {
    "left": "C:\\Users\\A Gaming\\Downloads\\archive\\Data\\Left Data",
    "right": "C:\\Users\\A Gaming\\Downloads\\archive\\Data\\Right Data",
    "center": "C:\\Users\\A Gaming\\Downloads\\archive\\Data\\Center Data"
}

RANDOM_STATE = 42

# TF-IDF settings (updated)
TFIDF_MAX_FEATURES = 25000          # Maximum number of features to extract
TFIDF_NGRAM_RANGE = (1, 3)          # tri-grams
TFIDF_ANALYZER = 'word'             
TFIDF_MIN_DF = 3                    # Ignore rare terms
TFIDF_MAX_DF = 0.9                  # Ignore overly common terms

# SVM Hyperparameters
SVM_C_VALUES = [0.01, 0.1, 1, 5, 10]
SVM_MAX_ITER = 2000

# Logistic Regression
LOG_REG_C_VALUES = [0.01, 0.1, 1, 5, 10]
LOG_REG_SOLVER = "saga"

NB_ALPHA_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0]