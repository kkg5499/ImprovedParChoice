# Copyright 2020 Secure Systems Group, Aalto University and University of Waterloo
# License: see LICENSE.txt

import os
import spacy
import pickle
import random
import argparse
from surrogate_classifier import surrogate_kwargs
from style_transformation import load_inflections, load_parser, load_ppdb, load_symspell, transform, CountVectorizer, TfidfVectorizer, LogisticRegressionSurrogate, MLPSurrogate

# Don't print convergence warning when training clf
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--src', help='Source corpus for transforming', default='data/alice_test.txt')
    arg_parser.add_argument('--src_train', help='Training corpus for src class', default='data/alice_train.txt')
    arg_parser.add_argument('--tgt_train', help='Training corpus for tgt class', default='data/bob_train.txt')
    arg_parser.add_argument('--use_ppdb', action='store_true')
    arg_parser.add_argument('--use_wordnet', action='store_true')
    arg_parser.add_argument('--use_typos', action='store_true')
    arg_parser.add_argument('--spell_check', action='store_true')
    arg_parser.add_argument('--clf', help='Pre-trained classifier', default='clf/LR_clf.pkl')
    arg_parser.add_argument('--clf_vectorizer', help='count/tf-idf', default='count')
    arg_parser.add_argument('--clf_type', help='lr/mlp', default='lr')
    arg_parser.add_argument('--clf_feat', help='word/char', default='word')
    arg_parser.add_argument('--clf_ngram_range', help='(min, max)', default=(1,1))
    arg_parser.add_argument('--clf_max_feats', help='clf vocabulary size', default=10000)
    arg_parser.add_argument('--flip_tgt', help='Use 0 instead of 1 as imitated class', default=False)
    arg_parser.add_argument('--save', help='Save transformations to a txt-file', default='results/alice_test_transf.txt')
    arg_parser.add_argument('--save_clf', help='Save trained classifier to a pickle file', default='clf/LR_clf.pkl')
    args = arg_parser.parse_args()
    
    src = open(args.src, 'r').readlines()
    clf, surrogate_corpus, surrogate_corpus_labels = None, None, None
    
    if os.path.exists(args.clf):
        with open(args.clf, 'rb') as f:
            clf = pickle.load(f)
    
    elif args.src_train and args.tgt_train:
        src_train = open(args.src_train, 'r').readlines()
        tgt_train = open(args.tgt_train, 'r').readlines()

        surrogate_corpus = src_train + tgt_train
        surrogate_corpus_labels = [0 for s in src_train] + [1 for s in tgt_train]
        surrogate_corpus = list(zip(surrogate_corpus, surrogate_corpus_labels))
        random.shuffle(surrogate_corpus)
        surrogate_corpus_labels = [l for (s,l) in surrogate_corpus]
        surrogate_corpus = [s for (s,l) in surrogate_corpus]
        
        surrogate_class = MLPSurrogate if args.clf_type=='mlp' else LogisticRegressionSurrogate
        surrogate_vectorizer = TfidfVectorizer if args.clf_vectorizer=='tf-idf' else CountVectorizer
        print('\nTraining classifier...', end=' ')
        clf = surrogate_class(surrogate_vectorizer, surrogate_kwargs(surrogate_vectorizer, args.clf_feat, args.clf_ngram_range, args.clf_max_feats), 1).fit(surrogate_corpus, surrogate_corpus_labels)
        print('Done!')
    
    if clf:
        src_labels = [1 for s in src] if args.flip_tgt else [0 for s in src]
        clf_acc = clf.accuracy(src, src_labels)
        print("Classifier accuracy with source before transformation: ", clf_acc)
    
    print('\nLoading dependencies...', end=' ')
    parser = load_parser()
    ppdb, infl, symspell = None, None, None
    if args.use_ppdb:
        ppdb = load_ppdb()
    if args.use_wordnet:
        infl = load_inflections()
    if args.use_typos or args.spell_check:
        symspell = load_symspell()
    print('Done!')
    
    print('\nTransforming source:')
    
    src_transformed = transform(src, parser=parser, ppdb_dict=ppdb, infl_dict=infl, symspell=symspell,
                                use_ppdb=args.use_ppdb, use_wn=args.use_wordnet, use_typos=args.use_typos, spell_check=args.spell_check,
                                max_len=50, max_cand=1000, max_loop = 1, max_edit_distance = 10, prob_threshold = 1,
                                surrogate=clf, surrogate_corpus=surrogate_corpus, surrogate_corpus_labels=surrogate_corpus_labels,
                                flip_target=args.flip_tgt, show_progress=True)
    
    if clf:
        clf_acc = clf.accuracy(src_transformed, src_labels)
        print("Classifier accuracy with source after transformation: ", clf_acc)
        
        if args.save_clf:
            save_clf_dir = os.path.dirname(args.save_clf)
            if not os.path.exists(save_clf_dir):
                os.makedirs(save_clf_dir)
            
            print('\nSaving classifier to:', args.save_clf)
            with open(args.save_clf, 'wb') as f:
                pickle.dump(clf, f)
    
    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_file = args.save
        if len(save_file.split('.')) > 1:
            save_file = '.'.join(save_file.split('.')[:-1])
        if args.use_ppdb:
            save_file += '_ppdb'
        if args.use_wordnet:
            save_file += '_wn'
        if args.use_typos:
            save_file += '_typos'
        save_file += '.txt'
        
        print('Saving transformations to:', save_file)
        with open(save_file, 'w') as f:
            for s in src_transformed:
                f.write(s.strip() + '\n')

if __name__=='__main__':
    main()
