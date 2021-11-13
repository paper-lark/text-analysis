# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Dict

import numpy as np
from pymystem3 import Mystem

m = Mystem()
word_re = re.compile(r'[a-zA-ZА-Яа-яёЁ0-9]+')


def lemmatize_sentence(sentence: str) -> List[str]:
    return list(filter(lambda x: word_re.fullmatch(x), m.lemmatize(sentence)))


def get_sentence_lemma_tf(sentence_lemmas: List[List[str]], lemma_to_index: Dict[str, int]) -> np.ndarray:
    tf = np.zeros((len(sentence_lemmas), len(lemma_to_index)))
    for i in range(len(sentence_lemmas)):
        for lemma in sentence_lemmas[i]:
            if lemma in lemma_to_index:
                j = lemma_to_index[lemma]
                tf[i, j] += 1
    return tf


def calculate_idf(sentence_lemma_tf: np.ndarray) -> np.ndarray:
    df = np.sum(np.where(sentence_lemma_tf > 0, 1, 0), axis=0)
    assert len(df) == sentence_lemma_tf.shape[1]
    n = sentence_lemma_tf.shape[0]
    idf = np.log(n / df)
    return idf


def normalize_cosine(sentence_vectors: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(sentence_vectors**2, axis=1, keepdims=True))
    return sentence_vectors / np.where(norm > 0, norm, 1)
