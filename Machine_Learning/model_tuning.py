import time
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=3000),  # Added model instance
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ['liblinear', 'saga'],
            "penalty": ['l1', 'l2']
        }
    },
    "Linear SVM": {
        "model": LinearSVC(max_iter=5000),  # Added model instance
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "loss": ['hinge', 'squared_hinge']
        }
    },
    "Multinomial Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.1, 0.5, 1.0, 1.5, 2.0]
        }
    }
}

def tune_models(X_train, y_train, X_test, y_test):
    results = []

    for name, mp in models_params.items():
        print(f"\n🔍 Tuning {name}...")
        start = time.time()

        grid = GridSearchCV(
            mp["model"],
            mp["params"],
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=2
        )
        grid.fit(X_train, y_train)
        end = time.time()

        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"{name} Accuracy: {acc*100:.2f}%")
        print(f"{name} F1 Score: {f1*100:.2f}%")
        print(f"{name} Time: {end - start:.2f}s")
        print(f"✅ Best params for {name}: {grid.best_params_}")
        print(f"📊 Classification Report:\n{classification_report(y_test, y_pred)}")

        # Confusion matrix for Linear SVM
        if name == "Linear SVM":
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix for Linear SVM")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()

        results.append({
            "Model": name,
            "Type": "Tuned",
            "Accuracy": acc,
            "F1": f1,
            "Time": f"{end - start:.2f}s"
        })

    return results