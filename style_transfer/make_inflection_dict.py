# Copyright 2020 Secure Systems Group, Aalto University and University of Waterloo
# License: see LICENSE.txt

from nltk import pos_tag, sent_tokenize, word_tokenize
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
import argparse
import tqdm
import pickle
import os

lemmatizer = WordNetLemmatizer()

# Load pre-existing inflection dict
def load_inflections(infl_path = 'inflections.pkl'):
    if os.path.isfile(infl_path):
        with open(infl_path, 'rb') as f:
            infl = pickle.load(f)
        return infl
    else:
        print("No inflection file found")
        
# NLTK POS to WN POS
def wn_pos(postag):
    pos_wn = []
    if postag[0]=='V':
        pos_wn = ['v']
    elif postag[0]=='N':
        pos_wn = ['n']
    elif postag[:2] =='JJ':
        pos_wn = ['a', 's']
    elif postag[:2] =='RB':
        pos_wn = ['r']
    return pos_wn

# Map lemmas and tags to word form counts
def inflection_count(text):
    infl_count = defaultdict(lambda: defaultdict(lambda: Counter()))
    if type(text)==str:
        text = sent_tokenize(text)
    for sent in tqdm.tqdm(text):
        for word, tag in pos_tag(word_tokenize(sent)):
            wn_tags = wn_pos(tag)
            for t in wn_tags:
                lemma = lemmatizer.lemmatize(word, t)
                infl_count[lemma][tag][word.lower()] += 1
    return infl_count

# Dict from lemmas and tags to the most common word form in an inflection count dict
def max_count(count_dict):
    max_dict = {}
    for word in count_dict:
        if word not in max_dict:
            max_dict[word] = {}
        for tag in count_dict[word]:
            max_dict[word][tag] = count_dict[word][tag].most_common()[0][0]
    return max_dict

# Make inflection dict from text corpus and save it with pickle
def save_inflection_dict(corpus, fpath):
    count_dict = inflection_count(corpus)
    count_dict_max = max_count(count_dict)
    with open(fpath, 'wb') as f:
        pickle.dump(count_dict_max, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='Source corpus for inflections (txt-file)')
    parser.add_argument('--save', default='inflections.pkl', help='Path to save inflections file')
    args = parser.parse_args()
    src_corpus = open(args.src, 'r').readlines()
    save_inflection_dict(src_corpus, args.save)

if __name__=='__main__':
    main()
