#!/usr/bin/env python3
'''
    calculates the unigram BLEU score
'''


import numpy as np


def uni_bleu(references, sentence):
    '''
        calculates the unigram BLEU score
        for a sentence
    '''
    sentence_length = len(sentence)

    # clip each word's count by the most times it appears in one reference
    total = 0
    for word in set(sentence):
        max_ref_count = max(ref.count(word) for ref in references)
        total += min(sentence.count(word), max_ref_count)

    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    best_match = len(references[index])

    if sentence_length > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(sentence_length))
    BLEU_score = BLEU * total / sentence_length

    return BLEU_score
