#!/bin/bash

# Download Python dependencies
python -c 'import nltk; nltk.download("wordnet"); nltk.download("averaged_perceptron_tagger"); nltk.download("punkt")'

python -m spacy download en_core_web_lg

# Run a test experiment
echo
echo 'Running test experiment'
python style_transfer/main.py --use_ppdb --use_wordnet --use_typos --spell_check
