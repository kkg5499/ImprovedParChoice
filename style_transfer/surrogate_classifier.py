# Copyright 2020 Secure Systems Group, Aalto University and University of Waterloo
# License: see LICENSE.txt

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# Surrogate classifier for choosing paraphrase in transformation
class Surrogate(object):

    def __init__(self, vectorizer_type, vectorizer_kwargs, target_label):
        super().__init__()
        self.vectorizer = vectorizer_type(**vectorizer_kwargs)
        self.target_label = target_label

    def vectorize_corpus(self, corpus):
        return self.vectorizer.fit_transform(corpus)

    def score(self, text):
        raise NotImplementedError()

    def agreement(self, texts):
        raise NotImplementedError()

    def accuracy(self, test_texts, test_labels):
        raise NotImplementedError()

# Logistic regression surrogate
class LogisticRegressionSurrogate(Surrogate):
    
    def __init__(self, vectorizer_type, vectorizer_kwargs, target_label):
        super().__init__(vectorizer_type, vectorizer_kwargs, target_label)
        self.classifier = LogisticRegression(random_state=0)

    def fit(self, surrogate_corpus, corpus_labels):
        self.classifier.fit(self.vectorize_corpus(surrogate_corpus), corpus_labels)
        return self

    def score(self, texts):
        vecs = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(vecs)
        return np.array([p[self.target_label] for p in probs])

    def agreement(self, texts):
        vecs = self.vectorizer.transform(texts)
        return np.array([1 if p == self.target_label else 0 for p in self.classifier.predict(vecs)])

    def accuracy(self, test_texts, test_labels):
        return self.classifier.score(self.vectorizer.transform(test_texts), test_labels)

# Alternative surrogate: MLP
class MLPSurrogate(Surrogate):
    
    def __init__(self, vectorizer_type, vectorizer_kwargs, target_label):
        super().__init__(vectorizer_type, vectorizer_kwargs, target_label)
        self.classifier = MLPClassifier(random_state=0)

    def fit(self, surrogate_corpus, corpus_labels):
        self.classifier.fit(self.vectorize_corpus(surrogate_corpus), corpus_labels)
        return self

    def score(self, texts):
        vecs = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(vecs)
        return np.array([p[self.target_label] for p in probs])

    def agreement(self, texts):
        vecs = self.vectorizer.transform(texts)
        return np.array([1 if p == self.target_label else 0 for p in self.classifier.predict(vecs)])

    def accuracy(self, test_texts, test_labels):
        return self.classifier.score(self.vectorizer.transform(test_texts), test_labels)

# Keyword arguments for a given vectorizer type
def surrogate_kwargs(surrogate_vectorizer_type, surrogate_feat, surrogate_ngr_range, surrogate_max_feats):   
    if surrogate_vectorizer_type == CountVectorizer:
        return {"analyzer": surrogate_feat, "ngram_range": surrogate_ngr_range, "max_features": surrogate_max_feats}
    elif surrogate_vectorizer_type == TfidfVectorizer:
        return {"analyzer": surrogate_feat, "ngram_range": surrogate_ngr_range, "max_features": surrogate_max_feats, "use_idf": False}
