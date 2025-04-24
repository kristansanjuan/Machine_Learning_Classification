import re

def clean_text(text):
    original_length = len(text)

    # Keep only letters and spaces
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
    cleaned_length = len(cleaned_text)

    # Count how many characters were removed
    deleted_count = original_length - cleaned_length

    return cleaned_text, deleted_count

def preprocess_texts(texts):
    cleaned_texts = []
    deleted_counts = []

    for text in texts:
        cleaned, deleted = clean_text(text)
        cleaned_texts.append(cleaned)
        deleted_counts.append(deleted)

    total_deleted = sum(deleted_counts)
    print(f"Total number of deleted characters: {total_deleted}")

    return cleaned_texts
