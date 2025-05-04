import re
import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from joblib import Parallel, delayed
from tqdm import tqdm

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
spell = SpellChecker()

def clean_text(text, enable_spell_check=False):
    if not text:
        return "", 0  # Return empty string and 0 deleted chars

    original_length = len(text)
    
    # Lowercase
    text = text.lower()

    # Remove punctuation and count deleted characters
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    punct_deleted = len(text) - len(text_no_punct)
    text = text_no_punct

    # Remove digits
    text_no_digits = re.sub(r'\d+', '', text)
    digits_deleted = len(text) - len(text_no_digits)
    text = text_no_digits

    # Optional spell check
    if enable_spell_check:
        corrected_words = []
        for word in text.split():
            corrected_word = spell.correction(word)
            corrected_words.append(corrected_word if corrected_word else word)
        text = " ".join(corrected_words)

    # Tokenize
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token]  # Filter empty tokens

    # POS tagging
    try:
        tagged = pos_tag(tokens)
    except Exception as e:
        print(f"POS tagging error on: {tokens}\n{e}")
        return "", original_length  # Consider all characters deleted if error occurs

    # Lemmatize and remove stopwords
    cleaned_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
        if word not in stop_words and word.isalpha()
    ]

    cleaned_text = " ".join(cleaned_tokens)
    total_deleted = original_length - len(cleaned_text)
    
    return cleaned_text, total_deleted

def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'  # adjective
    elif tag.startswith('V'): return 'v'  # verb
    elif tag.startswith('N'): return 'n'  # noun
    elif tag.startswith('R'): return 'r'  # adverb
    return 'n'  # default to noun

def parallel_preprocess(texts, enable_spell_check=False):
    results = Parallel(n_jobs=-1)(
        delayed(clean_text)(text, enable_spell_check)
        for text in tqdm(texts, desc="Processing texts")
    )
    
    # Unpack results into texts and deletion counts
    cleaned_texts, deleted_counts = zip(*results)
    total_deleted = sum(deleted_counts)
    
    print(f"\nTotal characters deleted: {total_deleted}")
    return list(cleaned_texts), total_deleted