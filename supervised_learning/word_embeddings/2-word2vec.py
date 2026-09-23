#!/usr/bin/env python3
"""
    word to vec
"""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    iterations=5,
    seed=0,
    workers=1,
):
    """
    word 2 vec
    """
    # gensim uses sg=0 for CBOW and sg=1 for Skip-gram
    cbow_flag = 0 if cbow else 1
    model = Word2Vec(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=cbow_flag,
        iter=iterations,
        seed=seed,
        workers=workers,
    )
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
