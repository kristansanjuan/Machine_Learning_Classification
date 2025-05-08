import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

def visualize(texts_cleaned, labels):
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

    # ---------- WORD CLOUD PER LABEL ----------
    os.makedirs("wordclouds", exist_ok=True)  # Create folder to save word clouds

    label_groups = {"left": [], "right": [], "center": []}

    for text, label in zip(texts_cleaned, labels):
        label_groups[label].append(text)

    for label, group_texts in label_groups.items():
        combined_text = " ".join(group_texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud - {label.capitalize()}")
        plt.tight_layout()

        # Save to file
        filename = f"wordclouds/{label}_wordcloud.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {filename}")