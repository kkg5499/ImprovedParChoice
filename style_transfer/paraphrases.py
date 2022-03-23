# Copyright 2020 Secure Systems Group, Aalto University and University of Waterloo
# License: see LICENSE.txt

import collections
from collections import defaultdict
import spacy
from nltk import pos_tag, sent_tokenize, word_tokenize, ngrams
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from itertools import chain, combinations, product
from string import punctuation
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import random
import numpy as np
from symspellpy.symspellpy import SymSpell, Verbosity
import os
import pickle
from pywsd.lesk import original_lesk, simple_lesk, adapted_lesk, cosine_lesk


# Load required models and dicts

# Spacy parser ()
def load_parser(model='en_core_web_lg'):
    parser = spacy.load(model, disable=['ner'])
    return parser

# PPDB
def ppdb_to_dict(ppdb_list, num_chars=3):
    ppdb_dict = {'comp':defaultdict(lambda: defaultdict(lambda: defaultdict(set))), 'pos':defaultdict(lambda: defaultdict(lambda: defaultdict(set)))}
    for context, phr, par in ppdb_list:
        context = context[1:-1]
        if '/' in context:
            comp = context.split('/')[1]
            ppdb_dict['comp'][phr[:num_chars]][phr][comp].add(par)
        else:
            ppdb_dict['pos'][phr[:num_chars]][phr][context].add(par)
    return ppdb_dict

def load_ppdb(path='ppdb/ppdb_equivalent.txt', num_chars=3):
    with open(path, 'r') as f:
        ppdb = f.readlines()
    ppdb = [p.split('|') for p in ppdb]
    ppdb = [[x.strip() for x in l] for l in ppdb]
    ppdb = ppdb_to_dict(ppdb, num_chars=num_chars)
    return ppdb

# Inflection dict (for inflecting WordNet lemmas)
def load_inflections(infl_path = 'inflections/inflections.pkl'):
    with open(infl_path, 'rb') as f:
        infl = pickle.load(f)
    return infl

# Symspell: frequency_dictionary_en_82_765.txt from https://github.com/wolfgarbe/SymSpell/tree/master/SymSpell
def load_symspell(dict_path='symspell/frequency_dictionary_en_82_765.txt', max_edit_distance_dictionary=2, prefix_length=7, term_index=0, count_index=1):
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    sym_spell.load_dictionary(dict_path, term_index, count_index)
    return sym_spell

#parser = load_parser()
#ppdb = load_ppdb()
#infl = load_inflections()

###############################################################################

# Some helper functions

def subsets(ls):
    return chain.from_iterable(combinations(ls, n) for n in range(len(ls)+1))

def flatten(ls):
    return [item for l in ls for item in l]

# Post-processing for correct indefinite article
vowels = 'aeio'
def fix_articles(text):
    new_text = ''
    words = text.split()
    for i in range(len(words)-1):
        word = words[i]
        next_word = words[i+1]
        if word=='a' and next_word[0] in vowels:
            new_text += 'an '
        elif word=='an' and next_word[0] not in vowels:
            new_text += 'a '
        else:
            new_text += word + ' '
    new_text += words[-1]
    return new_text

def all_ngrams(word_list):
    all_ngr = [word_list]
    for i in range(1, len(word_list)):
        all_ngr += list(ngrams(word_list, i))
    return [' '.join(ngr) for ngr in all_ngr]

def strip_punctuation(text):
    while text and text[0] in punctuation:
        text = text[1:]
    while text and text[-1] in punctuation:
        text = text[:-1]
    return text

def len_of_product(lists):
    lens = [len(l) for l in lists]
    return np.prod(lens)

###############################################################################

# PPDB

# Syntax tree information from Spacy parse, used in checking PPDB's conditions
def all_subtrees(token, label, parser):
    subtree_list = []
    token_subtree = list(token.subtree)
    token_index = token_subtree.index(token)
    beg, end = token_subtree[:token_index+1], token_subtree[token_index+1:]
    end_str = ' '.join([t.orth_.lower() for t in end]).strip()
    for i in range(len(beg)):
        beg_str = ' '.join([t.orth_.lower() for t in beg]).strip()
        if len(beg)>1:
            subtree_list.append((label, beg_str))
        if end:
            subtree_list.append((label, beg_str + ' ' + end_str))
        beg = beg[1:]
    return subtree_list

def trees(sent, parser):
    if type(sent) == str:
        sent = parser(sent)
    sent_words = [t.orth_.lower() for t in sent]
    sent_orth = ' '.join(sent_words).strip()
    tree_list = []
    
    for token in sent:
        
        # Verb head
        if token.pos_ == 'VERB' and token.dep_ != 'aux':
            subtree_str = ' '.join([t.orth_ for t in token.subtree]).lower()            
            tree_list.append(('VB', token.orth_))
            
            # S/SBAR: Full sentences
            if token.dep_ == 'ROOT':
                tree_list.append(('S', subtree_str))
            elif token.dep_ == 'ccomp':
                tree_list.append(('SBAR', subtree_str))
            
            # VP: Sentence without subject
            subjects = [t for t in sent if t.head == token and t.dep_ == 'nsubj']
            vp_subtree = list(token.subtree)
            subject_parts = set()
            for subj in subjects:
                subject_parts = subject_parts | set(subj.subtree)
            vp_subtree = [t for t in vp_subtree if t not in subject_parts]
            tree_list.append(('VP', ' '.join([t.orth_.lower() for t in vp_subtree]).strip()))
            
            # VP: Verb + object, no modifiers except aux + negation
            aux_neg = [t for t in sent if t.head == token and t.dep_ in ['aux', 'neg']]
            objects = [t for t in sent if t.head == token and t.dep_ in ['dobj', 'dative']]
            object_parts = set()
            for obj in objects:
                object_parts = object_parts | set(obj.subtree)
            vp_subtree = [t for t in vp_subtree if t==token or t in object_parts | set(aux_neg)]
            tree_list.append(('VP', ' '.join([t.orth_.lower() for t in vp_subtree]).strip()))
            
            # VP: Verb + object, no modifiers
            vp_subtree = [t for t in vp_subtree if t==token or t in object_parts]
            tree_list.append(('VP', ' '.join([t.orth_.lower() for t in vp_subtree]).strip()))
            
            # VP: modifiers + verb, no subject or object
            vp_subtree = [t for t in token.subtree if t not in subject_parts | object_parts]
            token_index = vp_subtree.index(token)
            beg, end = vp_subtree[:token_index+1], vp_subtree[token_index:][::-1]
            for i in range(len(beg)):
                tree_list.append(('VP', ' '.join([t.orth_.lower() for t in beg]).strip()))
                beg = beg[1:]
            for i in range(len(end)):
                tree_list.append(('VP', ' '.join([t.orth_.lower() for t in end][::-1]).strip()))
                end = end[1:]
        
        else:        
            if token.pos_=='NOUN':
                tree_list.append(('NN', token.orth_))
                tree_list += all_subtrees(token, 'NP', parser=parser)
                
            elif token.pos_=='ADJ':
                tree_list.append(('JJ', token.orth_))
                tree_list += all_subtrees(token, 'ADJP', parser=parser)
            
            elif token.pos_=='ADJ':
                tree_list.append(('JJ', token.orth_))
                tree_list += all_subtrees(token, 'ADJP', parser=parser)
            
            elif token.pos_=='ADV':
                tree_list.append(('RB', token.orth_))
                tree_list += all_subtrees(token, 'ADVP', parser=parser)
            
            elif token.pos_=='ADP':
                subtree_str = ' '.join([t.orth_ for t in token.subtree]).lower()
                tree_list.append(('PP', subtree_str))
    
    tree_list = set([pair for pair in tree_list if pair[1] in sent_orth])
    tree_list = tree_list | set([(label, strip_punctuation(s)) for (label, s) in tree_list])
    return tree_list

# Map phrases in a sentence to paraphrases from PPDB
def ppdb_matches(ppdb_dict, sent, parser, sent_parsed=None, sent_words=None, single_words=False, max_cand=100, ppdb_dict_key_len=3):
    pars = defaultdict(set)
    sent = sent.lower()
    if not sent_parsed:
        sent_parsed = parser(sent.lower())
    if not sent_words:
        sent_words = [t.orth_ for t in sent_parsed]
    sent = ' '.join(sent_words)
    sent_trees = trees(sent_parsed, parser=parser)
    
    for ngr in all_ngrams(sent_words):
        start = sent.index(ngr)
        end = start + len(ngr)
        complement = sent[end:].strip()
        
        if ngr in ppdb_dict['comp'][ngr[:ppdb_dict_key_len]]:
            for required_label in ppdb_dict['comp'][ngr[:ppdb_dict_key_len]][ngr]:
                for label, phrase in sent_trees:
                    if phrase in complement and phrase[:10]==complement[:10] \
                    and (label == required_label or (label[0] in ['N', 'V'] and label[0]==required_label[0])):
                        pars[ngr] = pars[ngr] | ppdb_dict['comp'][ngr[:ppdb_dict_key_len]][ngr][required_label]
        
    if single_words:
        for label, phrase in sent_trees:
            char_key = phrase[:ppdb_dict_key_len]
            if phrase in ppdb_dict['pos'][char_key]:
                if label in ppdb_dict['pos'][char_key][phrase]:
                    pars[phrase] = pars[phrase] | ppdb_dict['pos'][char_key][phrase][label]
                    
    # Filter out ungrammatical paraphrases
    for phr in sorted(list(pars.keys())):
        pars_filter = list(pars[phr])
        phr_parsed = parser(phr)
        phr_words = []
        phr_tags = set()
        phr_prons = set()
        for t in phr_parsed:
            phr_words.append(t.orth_)
            phr_tags.add(t.tag_)
            if t.pos_=='PRON' or t.tag_=='PRP$':
                phr_prons.add(t.orth_)
        
        for par in sorted(list(pars[phr])):
            if par in pars_filter:
                par_parsed = parser(par)
                par_words = []
                par_tags = set()
                par_prons = set()
                for t in par_parsed:
                    par_words.append(t.orth_)
                    par_tags.add(t.tag_)
                    if t.pos_ == 'PRON' or t.tag_=='PRP$':
                        par_prons.add(t.orth_)

                # No pronoun changes
                if phr_prons != par_prons:
                    pars_filter.remove(par)                

                # First and last tags must be the same
                for i in [0, -1]:
                    if par in pars_filter and par_parsed[i].tag_ != phr_parsed[i].tag_:
                        pars_filter.remove(par)                

                # Tense/number inflection
                for tag in ['VB', 'VBP', 'VBD', 'VBG', 'VBN', 'VBZ', 'NN', 'NNP', 'NNPS', 'NNS']:
                    if par in pars_filter and tag in (phr_tags - par_tags) | (par_tags - phr_tags):
                        if not (tag in ['VB', 'VBP'] and {'VB', 'VBP'} & par_tags):
                            pars_filter.remove(par)                                

                # Copula inflection with 1SG personal pronoun ("I")
                for i, word in enumerate(par_words):
                    if word in ['am', 'are'] and word not in phr_words and (i==0 or par_words[i-1]!='i'):
                        prev = sent[:sent.index(phr)].split()
                        if par in pars_filter and ((word=='am' and 'i' not in prev) or (word=='are' and 'i' in prev)):
                            pars_filter.remove(par)

        pars[phr] = pars_filter    

    if pars:
        sorted_pars_keys = list(sorted(pars.keys()))
        while np.prod([len(pars[p])+1 for p in sorted_pars_keys]) > max_cand:
            rand_phr = random.choice(sorted_pars_keys)
            while not pars[rand_phr]:
                rand_phr = random.choice(sorted_pars_keys)
            rand_par = random.choice(pars[rand_phr])
            pars[rand_phr].remove(rand_par)
        pars = {p:pars[p] for p in sorted_pars_keys if pars[p]}
    return pars

# Product of possible PPDB-paraphrases (max_cand prevents combinatory explosion: random subset to truncate size)
def ppdb_paraphrases(ppdb_dict, sent, parser, single_words=True, ppdb_dict_key_len=3, max_cand=1000):
    if type(sent)==str:
        sent_parsed = parser(sent)
    else:
        sent_parsed = sent
    sent_words = [t.orth_.lower() for t in sent_parsed]
    sent = ' '.join(sent_words).strip()
    pars = ppdb_matches(ppdb_dict=ppdb_dict, sent=sent, parser=parser, sent_parsed=sent_parsed, sent_words=sent_words, single_words=single_words, max_cand=max_cand, ppdb_dict_key_len=ppdb_dict_key_len)

    # Non-overlapping paraphrases based on position
    par_positions = {p:range(sent.index(p),sent.index(p)+len(p)) for p in pars}
    non_overlapping = []
    for combination in subsets(pars):
        ranges = flatten([list(par_positions[p]) for p in combination])
        if 0 < len(ranges) == len(set(ranges)):
            non_overlapping.append(combination)
        
    # Make all paraphrases
    all_alternatives = {sent}
    for comb in non_overlapping:
        par_combs = [pars[p] for p in comb]
        par_combs = product(*par_combs)
        for par_comb in par_combs:
            new_sent = sent
            for i in range(len(comb)):
                orig = comb[i]
                new = par_comb[i]
                new_sent = new_sent.replace(orig, new)
            new_sent = fix_articles(new_sent)
            all_alternatives.add(new_sent)
    
    return list(all_alternatives)

###############################################################################

# WordNet

lemmatizer = WordNetLemmatizer()

# NLTK postag to WN postag
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

# Words to never paraphrase from WordNet
no_synonyms = ['do', 'does', 'did', 'done', 'doing',
               'have', 'has', 'had', "'d", 'having',
               'be', 'am', "'m", 'are', "'re", "re", 'is', "'s", 'was', 'were', 'been', 'being',
               'get', 'gets', 'got', 'gotten',
               'not', "n't"]

# Product of WN-synonyms for each word (max_cand prevents combinatory explosion: random subset to truncate size)
def wn_paraphrases(infl_dict, sent, wsd=simple_lesk, max_cand=1000):
    if type(sent)==str:
        words = word_tokenize(sent)
    else:
        words = sent
        sent = ' '.join(words)
    words_tags = pos_tag(words)
    sent_synonyms = []
    for word, tag in words_tags:
        word = word.lower()
        word_synonyms = {word}
        if len(word)>2 and word not in no_synonyms:
            for pos in wn_pos(tag):
                try:
                    synset = wsd(sent, word) if wsd==original_lesk else wsd(sent, word, pos)
                except:
                    synset = None
                if synset:
                    for lemma in synset.lemma_names():
                        if lemma in infl_dict and tag in infl_dict[lemma]:
                            word_synonyms.add(infl_dict[lemma][tag])
        sent_synonyms.append(list(word_synonyms))
    
    while np.prod([len(l) for l in sent_synonyms]) > max_cand:
        rand_syns = random.choice(sent_synonyms)
        while len(rand_syns)==1:
            rand_syns = random.choice(sent_synonyms)
        syn = random.choice(rand_syns)
        rand_syns.remove(syn)    
    return [' '.join(l) for l in product(*sent_synonyms)]

###############################################################################

# Simple paraphrasing rules

# Removing commas: all possible combinations
def comma_combinations(sent):
    parts = []
    comma_indices = [i for i,char in enumerate(sent) if char == ',']    
    prev = 0
    for i in comma_indices:
        parts.append(sent[prev:i])
        prev = i+1
    parts.append(sent[prev:])    
    comma_comb = [parts[0]]    
    if len(parts) > 1:
        for part in parts[1:]:
            comma_comb_new = []
            for s in comma_comb:
                comma_comb_new.append(s.strip() + ' ' + part.strip())
                comma_comb_new.append(s.strip() + ' , ' + part.strip())
            comma_comb = comma_comb_new    
    return comma_comb    

modal_aux = {'might', 'may', 'could', 'can', 'should', 'ought', 'must', 'will', 'wo', 'shall'}

modal_synonyms = {
        'aff': [['might', 'may', 'could', 'can'],
                ['should', 'ought', 'must'],
                ['will', 'shall']],
        'neg': [['can', 'may', 'ca'],
                ['should', 'ought', 'must'],
                ['will', 'shall', 'wo']]}

# Replacing modal auxiliaries and adverbs with semantically close alternatives
def modal_comb(sent, parser):
    if type(sent)!=str or set(sent.split()) & modal_aux:
        if type(sent)==str:
            sent = parser(sent.lower().replace("can't", "can not"))
        syn_list = []
        has_modal = False
        sent_len = len(sent)
        for i,t in enumerate(sent):
            syns = [t.orth_]
            next_neg = False
            if t.orth_ in modal_aux and t.tag_=='MD':
                if i < (sent_len - 1) and sent[i+1].dep_ == 'neg':
                    next_neg = True
                syn_type = 'neg' if next_neg else 'aff'
                for modals in modal_synonyms[syn_type]:
                    if t.orth_ in modals or 'can' in modals and t.orth_=='ca':
                        has_modal = True
                        syns=modals
            syn_list.append(syns)
        if not has_modal:
            return [' '.join([t.orth_ for t in sent]).strip()]
        else:
            pars = product(*syn_list)
            pars = [' '.join(s).strip() for s in pars]
            pars_corr = []
            for p in pars:
                for o in ["ought", "ought not", "oughtn't"]:
                    p = p.replace(o, o+' to')
                    p = p.replace('to to', 'to')
                    p = p.replace('ought to not to', 'ought not to')
                    p = p.replace('ought to not', 'ought not to')
                    p = p.replace("ought to n't to", 'ought not to')
                    p = p.replace("ought to n't", 'ought not to')
                for m in modal_aux-{'ought'}:
                    p = p.replace(m+' to', m)
                    p = p.replace(m+' not to', m+' not')
                    p = p.replace(m+" n't to", m+" n't")
                pars_corr.append(p)
            return pars_corr
    else:
        return [' '.join([w for w in word_tokenize(sent.lower())])]

# Contractions (e.g. "is not" <-> "isn't")
def contractions(sent):
    sent = sent.replace("' ", "'")
    words = sent.split()
    word_amt = len(words)

    if word_amt == 0:
        return [sent]

    alternatives = [{words[0]}]
    for i in range(1, word_amt):
        word = words[i]
        if word=="n't" or (word=='not' and alternatives[i-1] & {'is', 'are', 'was', 'were', 'have', 'has', 'had', 'wo', 'must', 'should', 'need', 'ought', 'could', 'can', 'ca', 'do', 'does', 'did'}):
            alternatives.append({'not', "n't"})
        elif i<word_amt-1 and words[i+1][0] not in punctuation:
            if word in ["am", "'m"]:
                alternatives.append({"am", "'m"})
            elif word in ["are", "'re"]:
                alternatives.append({"are", "'re"})
            elif word=="'ve":
                alternatives.append({"have", "'ve"})
            else:
                alternatives.append({word})
        else:
            alternatives.append({word})
    alternatives = product(*alternatives)
    return [' '.join(a) for a in alternatives]

# Filter out ungrammatical paraphrases produced by the simple rules (bad inflections)
def grammar_filter(sent_list):
    filter_sents = []
    for sent in sent_list:
        sent_nopunct = ''.join(c for c in sent if c=="'" or c not in punctuation)
        sent_words = sent_nopunct.split()
        bad = False
        if "n't 's" in sent_nopunct:
            bad=True
        elif "n't" in sent_words:
            prev_idx = sent_words.index("n't")-1
            if prev_idx<0 or sent_words[prev_idx] not in ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'wo', 'must', 'should', 'need', 'ought', 'could', 'can', 'ca', 'do', 'does', 'did']:
                bad = True
        if "'ve" in sent_words:
            prev_idx = sent_words.index("'ve")-1
            if prev_idx<0 or sent_words[prev_idx] not in ['i', 'you', 'we', 'they']:
                bad=True
        if not bad:
            filter_sents.append(sent)
        elif "can n't" in sent or "ca not" in sent:
            filter_sents.append(sent.replace("can n't", "ca n't").replace("ca not", "ca n't"))
            filter_sents.append(sent.replace("can n't", "ca n't").replace("ca not", "can not"))
            filter_sents.append(sent.replace("can n't", "can not").replace("ca not", "ca n't"))
            filter_sents.append(sent.replace("can n't", "can not").replace("ca not", "can not"))
    return set(filter_sents)

# Synonymous indefinite pronouns
one_prons = ['anyone', 'no one', 'someone', 'everyone']
body_prons = ['anybody', 'nobody', 'somebody', 'everybody']
indef_pron_repl = dict(list(zip(one_prons, body_prons)) + list(zip(body_prons, one_prons)))

def indefinite_pronouns(sents):
    if type(sents) == str:
        sents = [sents]
    sents_pron = []
    for sent in sents:
        alternatives = []
        for word in sent.split():
            if word in indef_pron_repl:
                alternatives.append([word, indef_pron_repl[word]])
            else:
                alternatives.append([word])
        alternatives = product(*alternatives)
        sents_pron += [' '.join(a) for a in alternatives]
    return sents_pron

def simple_paraphrases(sent, parser, incl_modal=True, incl_contractions=True, incl_comma=True, incl_indef_pron=True):
    pars = []
    if incl_modal:
        modals = modal_comb(sent, parser=parser)
    else:
        modals = [' '.join([w for w in word_tokenize(sent.lower())])]
    for s in modals:
        if incl_contractions:
            pars += contractions(s)
        else:
            pars.append(s)
    if incl_comma:
        pars_comma = []
        for s in pars:
            pars_comma += comma_combinations(s)
        pars = pars_comma
    if incl_indef_pron:
        pars = indefinite_pronouns(pars)
    return grammar_filter(pars)

###############################################################################

# Typos (from a specified corpus: tgt corpus by default in ParChoice pipeline)

def symspell_lookup_corr(sent, symspell, max_edit_distance=1):
    return symspell.lookup_compound(sent, max_edit_distance)[0].term

def symspell_segment_corr(sent, symspell, max_edit_distance=1):
    segm = symspell.word_segmentation(sent, max_edit_distance)
    return segm.segmented_string, segm.corrected_string

# Dict from words to their possible misspellings
def misspells(text, symspell=None, segment=False, min_length=5, max_sents=None, print_every=None):
    if not symspell:
        print("Loading SymSpell...", end=' ')
        symspell = load_symspell()
        print("Done")
    if type(text)==str:
        text = sent_tokenize(text)
    if max_sents and len(text)>max_sents:
        random.shuffle(text)
        text = text[:max_sents]
    text_len = len(text)
    miss_dict = collections.defaultdict(set)
    corr_text = []
    for i,sent in enumerate(text):
        if segment:
            segmented, corrected = symspell_segment_corr(' '.join(word_tokenize(sent.lower())), symspell)
            for word, corr in zip(segmented.split(), corrected.split()):
                if len(word)>=min_length and word.replace('.', '') != corr:
                    miss_dict[corr].add(word)
                    corr_text.append(corr)
                else:
                    corr_text.append(word)
            else:
                corr_text.append(word)
        else:
            for word in word_tokenize(sent.lower()):
                if len(word)>=min_length:
                    corr = symspell_lookup_corr(word, symspell)
                    if word.replace('.', '') != corr:
                        miss_dict[corr].add(word)
                        corr_text.append(corr)
                    else:
                        corr_text.append(word)
                else:
                    corr_text.append(word)
        if i>0 and print_every and i%print_every==0:
            print(i, '/', text_len, 'target sentences checked for typos')
#    print("Typo dict created")
    return miss_dict, ' '.join(corr_text).strip()

# Combinations of removing apostrophes
def apostrophes(sent):
    chars = [["'", " "] if c=="'" else [c] for c in sent]
    return [''.join(tupl).replace('  ', '') for tupl in product(*chars)]

# Typo combinations of words in sentence based on pre-built typo dict + apostrophes
def typo_candidates(sent, typo_dict, max_cand=100):
    sent_typos = [list({w} | typo_dict[w]) for w in word_tokenize(sent.lower())]
    while np.prod([len(l) for l in sent_typos]) > max_cand:
        rand_syns = random.choice(sent_typos)
        while len(rand_syns)==1:
            rand_syns = random.choice(sent_typos)
        syn = random.choice(rand_syns)
        rand_syns.remove(syn)  
    sent_typos = [' '.join(tupl) for tupl in product(*sent_typos)]
    sent_typos = [apostrophes(s) for s in sent_typos]
    sent_typos = [item for l in sent_typos for item in l]
    return sent_typos

###############################################################################

# Combining all paraphrase functions
# Order: simple (with modals) -> PPDB -> WN -> simple (without modals) -> typos
    
def paraphrase_candidates(sents, parser, ppdb_dict, infl_dict, use_ppdb=True, use_wn=True, max_cand=1000, print_time=False):
    time1 = time.time()
    if type(sents)==str:
        sents = [sents]
    sents = flatten([simple_paraphrases(s, parser=parser, incl_modal=True) for s in sents])
    
    if use_ppdb:
        sents_ppdb = [s for s in sents]
        idx_list = list(range(len(sents)))
        while len(sents_ppdb)<max_cand and idx_list:
            idx = random.choice(idx_list)
            rand_sent = sents[idx]
            sents_ppdb += ppdb_paraphrases(ppdb_dict, rand_sent, parser=parser, max_cand=max_cand)
            idx_list.remove(idx)
        sents = sents_ppdb
    
    if use_wn:
        sents_wn = [s for s in sents]
        idx_list = list(range(len(sents)))
        while len(sents_wn)<max_cand and idx_list:
            idx = random.choice(idx_list)
            rand_sent = sents[idx]
            sents_wn += wn_paraphrases(infl_dict, rand_sent, max_cand=max_cand)
            idx_list.remove(idx)
        sents = sents_wn
    
    sents = list(set(flatten([simple_paraphrases(s, parser=parser, incl_modal=False) for s in sents])))
    
    if len(sents)>max_cand:
        random.shuffle(sents)
        sents = sents[:max_cand]

    if print_time:
        print(time.time()-time1)
    return sents
