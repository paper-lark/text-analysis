# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from dictionary import Dictionary
from models.model import RelevanceModel
from tf_idf import get_sentence_lemma_tf, lemmatize_sentence


def get_relevant_sentences(query: str, vocab: Dictionary, model: RelevanceModel) -> pd.DataFrame:
    query_lemmas = list(map(lemmatize_sentence, [query]))
    query_tf = get_sentence_lemma_tf(query_lemmas, vocab.lemma_to_index)
    weights = model.get_serp_weights(query_tf)[0, :]
    idx = np.flip(np.argsort(weights))

    return pd.DataFrame(dict(
        relevance=vocab.scores[idx],
        weights=weights[idx],
        sentences=[vocab.sentences[i] for i in idx],
    ), index=idx)

