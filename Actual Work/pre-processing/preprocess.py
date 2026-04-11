
import re
import nltk
import ftfy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Stopwords — keep negations since they affect sentiment
STOPWORDS = set(stopwords.words("english"))
KEEP_WORDS = {"not", "no", "never", "very", "but", "however"}
STOPWORDS = STOPWORDS - KEEP_WORDS

lemmatizer = WordNetLemmatizer()

def fix_encoding(text):
    """Fix mojibake encoding issues."""
    if not isinstance(text, str):
        return ""
    return ftfy.fix_text(text)

def lowercase(text):
    """Lowercase all text."""
    return text.lower()

def remove_html(text):
    """Remove HTML tags like <br/>."""
    return re.sub(r"<[^>]+>", " ", text)

def remove_punctuation(text):
    """Remove punctuation and special characters."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)

def tokenize(text):
    """Split text into tokens."""
    return word_tokenize(text)

def filter_ascii(tokens):
    """Remove non-ASCII tokens (garbled characters)."""
    return [t for t in tokens if t.isascii()]

def remove_stopwords(tokens):
    """Remove stopwords but keep negations."""
    return [t for t in tokens if t not in STOPWORDS]

def lemmatize(tokens):
    """Reduce words to base form e.g. running -> run."""
    return [lemmatizer.lemmatize(t) for t in tokens]

def remove_short_tokens(tokens, min_len=2):
    """Remove single character tokens."""
    return [t for t in tokens if len(t) >= min_len]

def clean_text(text):
    """
    Full preprocessing pipeline.

    Steps:
        1. Fix encoding (ftfy)
        2. Lowercase
        3. Remove HTML tags
        4. Remove punctuation
        5. Tokenize
        6. Filter non-ASCII tokens
        7. Remove stopwords (keeping negations)
        8. Lemmatize
        9. Remove short tokens

    Args:
        text (str): raw review text

    Returns:
        tuple: (cleaned_text as string, tokens as list)
    """
    text = fix_encoding(text)
    text = lowercase(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = filter_ascii(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = remove_short_tokens(tokens)
    cleaned_text = " ".join(tokens)
    return cleaned_text, tokens


def preprocess_dataframe(df, text_col="comments"):
    """
    Apply clean_text to an entire dataframe column.

    Adds two new columns:
        cleaned_text — preprocessed text as a string
        tokens       — preprocessed text as a list of tokens

    Args:
        df (pd.DataFrame): dataframe with a text column
        text_col (str): name of the column containing raw text

    Returns:
        pd.DataFrame: original dataframe with two new columns added
    """
    results = df[text_col].apply(clean_text)
    df["cleaned_text"] = results.apply(lambda x: x[0])
    df["tokens"]       = results.apply(lambda x: x[1])
    return df
