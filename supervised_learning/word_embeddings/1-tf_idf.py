#!/usr/bin/env python3
'''
    TF-IDF embedding
'''
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    '''
        tf-idf embeddings
    '''
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embeddings = x.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
