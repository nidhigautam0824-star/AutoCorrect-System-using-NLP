import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "words.csv"


def load_word_frequencies(csv_path=DATASET_PATH):
    """Load a CSV with columns: word, frequency."""
    df = pd.read_csv(csv_path)

    required_columns = {"word", "frequency"}
    if not required_columns.issubset(df.columns):
        raise ValueError("words.csv must contain 'word' and 'frequency' columns.")

    df = df.dropna(subset=["word", "frequency"]).copy()
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df = df.dropna(subset=["frequency"])
    df = df[df["word"].str.fullmatch(r"[a-z]+")]
    df = df[df["frequency"] >= 0]

    # Combine duplicate words, if any.
    frequencies = df.groupby("word")["frequency"].sum().to_dict()
    if not frequencies:
        raise ValueError("No valid words were found in words.csv.")
    return frequencies


def tokenize(text):
    """Split text into word tokens while preserving punctuation separately."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+", text)


def levenshtein_distance(word1, word2):
    """Minimum insertions, deletions, and substitutions to transform word1 into word2."""
    word1, word2 = word1.lower(), word2.lower()

    # Dynamic programming with a single row of memory.
    previous = list(range(len(word2) + 1))

    for i, char1 in enumerate(word1, start=1):
        current = [i]
        for j, char2 in enumerate(word2, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char1 != char2)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return previous[-1]


def suggest_correction(word, frequencies, max_distance=2):
    """Return the best dictionary match, preferring close and frequent words."""
    word_lower = word.lower()

    if word_lower in frequencies:
        return word_lower

    candidates = []
    for candidate, frequency in frequencies.items():
        # Avoid comparing words that differ greatly in length.
        if abs(len(candidate) - len(word_lower)) > max_distance:
            continue

        distance = levenshtein_distance(word_lower, candidate)
        if distance <= max_distance:
            candidates.append((distance, -frequency, candidate))

    if not candidates:
        return word

    # Lowest edit distance first; then higher frequency.
    return min(candidates)[2]


def autocorrect_text(text, frequencies, max_distance=2):
    """Correct words in a sentence and preserve punctuation/spacing."""
    tokens = tokenize(text)
    corrected_tokens = []

    for token in tokens:
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token):
            # Keep contractions as-is in this simple demo.
            if "'" in token:
                corrected_tokens.append(token)
                continue

            correction = suggest_correction(token, frequencies, max_distance)
            if token.isupper():
                correction = correction.upper()
            elif token.istitle():
                correction = correction.title()

            corrected_tokens.append(correction)
        else:
            corrected_tokens.append(token)

    return "".join(corrected_tokens)


def main():
    try:
        frequencies = load_word_frequencies()
    except FileNotFoundError:
        print(f"Dataset not found: {DATASET_PATH}")
        print("Keep words.csv in the same folder as autocorrect.py.")
        return
    except (ValueError, pd.errors.ParserError) as error:
        print(f"Could not load dataset: {error}")
        return

    print("=" * 52)
    print("       AUTOCORRECT SYSTEM USING NLP")
    print("=" * 52)
    print(f"Loaded {len(frequencies)} words from the dataset.")
    print("Type a sentence to correct it, or type 'exit' to quit.\n")

    while True:
        user_text = input("Enter text: ").strip()
        if user_text.lower() == "exit":
            print("Goodbye!")
            break
        if not user_text:
            print("Please enter some text.\n")
            continue

        corrected = autocorrect_text(user_text, frequencies)
        print("Corrected:", corrected, "\n")


if __name__ == "__main__":
    main()
