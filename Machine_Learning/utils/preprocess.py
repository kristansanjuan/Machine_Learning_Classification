import re

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    #total_deleted = sum(deleted_counts)
    #print(f"Total number of deleted characters: {total_deleted}")

    return text

def preprocess_texts(texts):
    return [clean_text(t) for t in texts]