import re
import pandas as pd


ACCENTED_WORDS = {}
PUNCTUATION_REPLACEMENTS = str.maketrans(
    {
        "„": '"',
        "“": '"',
        "”": '"',
        "‘": '"',
        "’": '"',
        "–": "-",
        "—": "-",
        ";": ",",
    }
)
LETTER_REPLACEMENTS = str.maketrans(
    {
        "x": "ks",
        "w": "v",
        "q": "kv",
    }
)


def normalize_text(text):
    """
    Normalize text for TTS training.

    Args:
        text (str): Raw text to normalize

    Returns:
        str: Normalized text
    """
    # Remove punctuation except apostrophes (important for Lithuanian)
    # Keep basic punctuation that affects pronunciation
    text = text.translate(PUNCTUATION_REPLACEMENTS)
    text = re.sub(r"[^\w\s.,\-?!]", "", text)

    # Remove extra whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    # Add accents
    parts = re.split(r"\b", text)
    accented_parts = [ACCENTED_WORDS.get(part, part) for part in parts]
    text = "".join(accented_parts)

    # Convert to lowercase
    text = text.lower()

    # Apply letter replacements
    text = text.translate(LETTER_REPLACEMENTS)

    return text


def load_accented_words(accented_words_dict_path):
    global ACCENTED_WORDS

    accented_df = pd.read_csv(
        accented_words_dict_path,
        keep_default_na=False,
    )
    ACCENTED_WORDS = dict(accented_df.values)
