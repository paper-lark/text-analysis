# -*- coding: utf-8 -*-
import numpy as np

from models.model import RelevanceModel


class LanguageModel(RelevanceModel):
    def __init__(self, doc_lemma_tf: np.ndarray, la: float):
        assert 0 <= la <= 1

        collection_lemma = np.sum(doc_lemma_tf, axis=0, keepdims=True)
        collection_lemma_proba = collection_lemma / np.sum(collection_lemma)
        doc_lemma_proba = doc_lemma_tf / np.sum(doc_lemma_tf, axis=1, keepdims=True)

        self.__lambda = la
        self.__collection_lemma_proba = collection_lemma_proba
        self.__doc_lemma_proba = doc_lemma_proba

    def __str__(self) -> str:
        return f"LanguageModel(la={self.__lambda})"

    def get_serp_weights(self, queries_tf: np.ndarray) -> np.ndarray:
        """
        :param queries_tf: numpy array, dimensions: (query, lemma)
        :return: serp indexes
        """

        # NOTE: dimensions = (query, document, lemma)
        p_t = ((np.ones_like(queries_tf) * self.__collection_lemma_proba) ** queries_tf)[:, np.newaxis, :]
        p_t_m_d = (np.ones_like(queries_tf)[:, np.newaxis, :] * self.__doc_lemma_proba[np.newaxis, :, :]
                   ) ** queries_tf[:, np.newaxis, :]
        p_q_d = np.prod((1 - self.__lambda) * p_t + self.__lambda * p_t_m_d, axis=2)
        return p_q_d
