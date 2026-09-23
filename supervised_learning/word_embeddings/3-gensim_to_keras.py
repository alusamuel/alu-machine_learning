#!/usr/bin/env python3
'''
    Script that defines a function
    gensim_to_keras
'''


def gensim_to_keras(model):
    '''
        Converts a gensim word2vec model to a Keras Embedding layer
    '''
    return model.wv.get_keras_embedding(train_embeddings=True)
