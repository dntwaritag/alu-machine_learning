#!/usr/bin/env python3
'''
    Script that defines a function
    bag_of_words
'''

def gensim_to_keras(model):
    '''
        Converts a gensim word2vec model to a Keras Embedding layer
    '''
    return model.wv.get_keras_embedding(train_embeddings=True)
