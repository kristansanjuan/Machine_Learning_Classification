# model_tuning.py

import time
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
    "Linear SVM": {
        "model": LinearSVC(),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    },
    "Multinomial Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.5, 1.0, 1.5]
        }
    }
}

def tune_models(X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score

    results = []

    for name, mp in models_params.items():
        print(f"\n🔍 Tuning {name}...")
        start = time.time()
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='f1_macro')
        grid.fit(X_train, y_train)

        end = time.time()

        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"{name} Accuracy: {acc*100:.2f}%")
        print(f"{name} F1 Score: {f1*100:.2f}%")
        print(f"{name} Time: {end - start:.2f}s")

        print(f"✅ Best params for {name}: {grid.best_params_}")
        print(f"📊 Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Type": "Tuned",
            "Accuracy": acc,
            "F1": f1,
            "Time": f"{end - start:.2f}s"
        })

    print("\n📋 Tuned Results:")
    for r in results:
        print(f"Model: {r['Model']}")
        print(f"Type: {r['Type']}")
        print(f"Accuracy: {r['Accuracy']}")
        print(f"F1: {r['F1']}")
        print(f"Time: {r['Time']}")
        print()

    return results