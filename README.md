# AutoCorrect System using NLP

## Features
- Tokenizes text using regular expressions.
- Finds spelling candidates using Levenshtein Distance.
- Uses word frequencies to rank candidates with the same edit distance.
- Uses Pandas to load and preprocess a CSV word-frequency dataset.
- Preserves punctuation and spacing in the input sentence.

## Setup
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open the VS Code terminal in this folder.
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Run:
   `python autocorrect.py`

Type a sentence, for example:
`Helo wrld, this is a beautful project.`
Type `exit` to stop.

## Dataset note
`words.csv` is a small demonstration dataset, not a large production dictionary. For better corrections, add more correctly spelled words and their frequency counts. Frequencies are illustrative.
