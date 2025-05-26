#!/usr/bin/env python3
'''
    Bag of words
'''
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    '''
        bag of words
    '''
    vectorizer = CountVectorizer(vocabulary=vocab)
    X_train_counts = vectorizer.fit_transform(sentences)
    embeddings = X_train_counts.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
