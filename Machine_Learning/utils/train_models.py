from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Naive Bayes": MultinomialNB()
    }

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time()

        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%")
        print(f"Time: {end - start:.2f} seconds")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Left', 'Center', 'Right'], yticklabels=['Left', 'Center', 'Right'])
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()