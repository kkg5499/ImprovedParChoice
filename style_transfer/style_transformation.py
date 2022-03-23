# Copyright 2020 Secure Systems Group, Aalto University and University of Waterloo
# License: see LICENSE.txt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize, word_tokenize
import numpy as np
import random
import collections
import gc
from tqdm import tqdm
import editdistance

from surrogate_classifier import LogisticRegressionSurrogate, MLPSurrogate, surrogate_kwargs
from paraphrases import load_parser, load_ppdb, load_inflections, load_symspell, misspells, typo_candidates, paraphrase_candidates
#parser = load_parser()
#ppdb = load_ppdb()
#infl = load_inflections()
#symspell = load_symspell()

# Connect punctuation
def connect_punct(sent):
    puncts = [c for c in sent if c in string.punctuation]
    for p in puncts:
        sent = sent.replace(' ' + p, p)
    return sent

def listify_corpus_into_sentence(corpus):
    if type(corpus) == str:
        return [s.strip() for s in sent_tokenize(corpus)]
    return corpus

#Garbage collection per gc_every iteration
def gc_loop(loop_count, gc_every=100):
    if loop_count > 0 and loop_count % gc_every == 0:
        gc.collect()

# Style transformation on src (corpus) to imitate style of surrogate corpus tgt class (imitating class 1 by default; class 0 if flip_target==True)
def transform(src,
              parser=None,
              ppdb_dict=None,
              infl_dict=None,
              symspell=None,
              use_ppdb=True,
              use_wn=True,
              use_typos=True,
              max_len=50,
              max_cand=1000,
              spell_check=True,
              max_loop = 1,
              max_edit_distance = 10,
              prob_threshold = 1,
              flip_target=False,
              show_progress=True,
              surrogate=None,
              surrogate_corpus=None,
              surrogate_corpus_labels=None,
              surrogate_vectorizer_type=CountVectorizer,
              surrogate_class=LogisticRegressionSurrogate,
              surrogate_feat='word',
              surrogate_ngr_range=(1,1),
              surrogate_max_feats=10000,
              save_surrogate=None):
    
    src = listify_corpus_into_sentence(src)
    
    if use_typos:
#        print("Finding misspellings...", end=' ')
        symspell = load_symspell() if not symspell else symspell
        typo_label = 0 if flip_target else 1
        if surrogate_corpus:
            typo_corpus = [s for s, i in zip(surrogate_corpus, surrogate_corpus_labels) if i == typo_label]
        else:
            typo_corpus = src
        typo_dict, _ = misspells(typo_corpus, symspell=symspell, max_sents=200000, print_every=None)
#        print('Done!')

    elif spell_check:
        typo_dict = collections.defaultdict(set)

    src_transf = []
    progress = tqdm(src, disable=not show_progress)

    for doc_count, doc in enumerate(progress):
        gc_loop(doc_count)

        try:
            best_doc = doc
            sents = sent_tokenize(doc)

            if spell_check:
                sents_corr = []
                for s in sents:
                    s_typo_dict, s_corr = misspells([s], symspell=symspell, print_every=1000)
                    sents_corr.append(s_corr)
                    for c in s_typo_dict:
                        typo_dict[c] = typo_dict[c] | s_typo_dict[c]
            else:
                sents_corr = [s.lower() for s in sents]

            sents_corr = [' '.join(s.split()[:max_len]) for s in sents_corr]
            sent_candidates = [[s] for s in sents_corr]
            
            max_prob = 0
            best_agreement_score = 0
            
            for loop_count in range(max_loop):
                for i, sent in enumerate(sents):

                    # Only transform if max_prob is below threshold
                    if max_prob <= prob_threshold:
                        candidates = sent_candidates[i]
                        best_sent = candidates[0]
                        left_context = ' '.join(sents_corr[:i])
                        right_context = ' '.join(sents_corr[i+1:])
                        
                        candidates = [s for s in candidates if editdistance.eval(sent.split(), s.split())<=max_edit_distance]

                        candidates = paraphrase_candidates(candidates,
                                                           parser=parser,
                                                           ppdb_dict=ppdb_dict,
                                                           infl_dict=infl_dict,
                                                           use_ppdb=use_ppdb,
                                                           use_wn=use_wn,
                                                           max_cand=max_cand)

                        max_typo_cand = int(max_cand / len(candidates)) if len(candidates)>0 else 0
                        if use_typos and typo_dict and max_typo_cand:
                            candidates_with_typos = []
                            for cand in candidates:
                                typo_cand = typo_candidates(cand, typo_dict, max_cand=max_typo_cand)
                                candidates_with_typos += typo_cand
                            candidates = set(candidates_with_typos)

                        candidates = [s for s in candidates if editdistance.eval(sent.split(), s.split())<=max_edit_distance]
                        candidates_with_context = [' '.join([left_context, cand, right_context]).strip() for cand in candidates]                        
                        
                        if len(candidates) > 1 and surrogate:
                            agrs = [0 for s in candidates_with_context]
                            surr_preds = surrogate.agreement(candidates_with_context)
                            if flip_target:
                                surr_preds = [1-p for p in surr_preds]
                            agrs = [p+s for (p,s) in zip(agrs, surr_preds)]
                            max_agr = max(agrs)

                            candidates = [s for (j,s) in enumerate(candidates) if agrs[j]==max_agr]
                            candidates_with_context = [s for (j,s) in enumerate(candidates_with_context) if agrs[j]==max_agr]

                            surr_probs = surrogate.score(candidates_with_context)
                            if flip_target:
                                surr_probs = np.array([1 - p for p in surr_probs])                            
                            probs = {p:j for (j,p) in enumerate(surr_probs)}
                            max_prob_new = max(probs)
                            if max_prob_new > max_prob:
                                max_prob = max_prob_new
                                best_ind = probs[max_prob]
                                best_sent = candidates[best_ind]
                                best_doc = candidates_with_context[best_ind]
                        
                        else:
                            best_ind = random.choice(list(range(len(candidates)))) if len(candidates)>0 else None
                            if best_ind:
                                best_sent = candidates[best_ind]
                                best_doc = candidates_with_context[best_ind]
                            
                        sent_candidates[i] = candidates
                        sents_corr[i] = best_sent
            
            if surrogate:
                progress.set_postfix(max_prob=max_prob)
            src_transf.append(best_doc.strip())
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error processing sentence", doc)
            src_transf.append(doc.strip())
    
    return src_transf
