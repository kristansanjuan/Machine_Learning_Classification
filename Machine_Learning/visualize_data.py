import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from utils.load_data import load_texts_and_labels
from utils.preprocess import preprocess_texts

def visualize():
    # Load and preprocess data
    texts, labels = load_texts_and_labels()
    texts_cleaned = preprocess_texts(texts)

    # ---------- HISTOGRAM OF LABELS ----------
    label_counts = Counter(labels)

    plt.figure(figsize=(8, 5))
    plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
    plt.title('Distribution of Class Labels')
    plt.xlabel('Labels')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------- WORD CLOUD ----------
    all_text = " ".join(texts_cleaned)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Dataset")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()