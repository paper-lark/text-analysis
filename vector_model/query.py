# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from dictionary import Dictionary
from tf_idf import get_sentence_lemma_tf, lemmatize_sentence, normalize_cosine


def get_weights_for_query(query: str, vocab: Dictionary, log_tf: bool = False) -> np.ndarray:
    # calculate query vector
    query_lemmas = list(map(lemmatize_sentence, [query]))
    query_tf = get_sentence_lemma_tf(query_lemmas, vocab.lemma_to_index)
    query_vec = normalize_cosine(query_tf)

    # calculate dictionary vectors
    sentence_lemma_tf = vocab.sentence_lemma_tf
    if log_tf:
        sentence_lemma_tf = np.log10(sentence_lemma_tf + 1)
    sentences_vec = normalize_cosine(sentence_lemma_tf * vocab.lemma_idf[np.newaxis, :])

    weights = np.sum(query_vec * sentences_vec, axis=1)
    return weights


def get_relevant_sentences(query: str, vocab: Dictionary, n: int = 20, log_tf: bool = False) -> pd.DataFrame:
    weights = get_weights_for_query(query, vocab, log_tf=log_tf)
    idx = np.flip(np.argsort(weights))[:n]
    return pd.DataFrame(dict(
        weights=weights[idx],
        sentences=[vocab.sentences[i] for i in idx]
    ), index=idx)

