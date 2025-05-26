#!/usr/bin/env python3
"""
    FastText
"""

from gensim.models import FastText


def fasttext_model(
    sentences,
    size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    iterations=5,
    seed=0,
    workers=1,
):
    """
    fast text
    """
    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1
    model = FastText(
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
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model
