from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# def train_doc2vec(texts, labels):
#     tagged_data = [TaggedDocument(words=t.split(), tags=[str(i)]) for i, t in enumerate(texts)]
#     model = Doc2Vec(vector_size=100, alpha=0.025, min_count=2, epochs=40)
#     model.build_vocab(tagged_data)
#     model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

#     vectors = [model.infer_vector(doc.words) for doc in tagged_data]
#     return np.array(vectors), labels

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB()
    }

def train_and_evaluate(X_train, X_test, y_train, y_test, label_names):
    models = get_models()
    results = []
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        start = time.time()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        end = time.time()
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"{name} Accuracy: {acc*100:.2f}%")
        print(f"{name} F1 Score: {f1*100:.2f}%")
        print(f"{name} Time: {end - start:.2f}s")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))

        cm = confusion_matrix(y_test, y_pred, labels=label_names)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        results.append({
            "Model": name,
            "Type": "Baseline",
            "Accuracy": acc,
            "F1": f1
        })

    print("\n📋 All Baseline Results:\n")
    for result in results:
        print(f"Model: {result['Model']}")
        print(f"Type: {result['Type']}")
        print(f"Accuracy: {result['Accuracy']:.4f}")
        print(f"F1: {result['F1']:.4f}")
        print()

    return results

def train_all_models(X_train, X_test, y_train, y_test):
    label_names = sorted(list(set(y_train) | set(y_test)))
    return train_and_evaluate(X_train, X_test, y_train, y_test, label_names)