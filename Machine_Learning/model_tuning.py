# model_tuning.py

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ['liblinear', 'lbfgs']
        }
    },
    "Multinomial Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.5, 1.0, 1.5]
        }
    },
    "Linear SVM": {
        "model": LinearSVC(),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    }
}

def tune_models(X_train, y_train, X_test, y_test):

    results = []

    for name, mp in models_params.items():
        print(f"\n🔍 Tuning {name}...")
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='f1_macro')
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        print(f"✅ Best params for {name}: {grid.best_params_}")
        print(f"📊 Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred))

    print("\n📋 Tuned Results:")
    for r in results:
        print(f"Model: {r['Model']}")
        print(f"Type: {r['Type']}")
        print(f"Accuracy: {r['Accuracy']:.4f}")
        print(f"F1: {r['F1']:.4f}")
        print("-" * 30)

    return results