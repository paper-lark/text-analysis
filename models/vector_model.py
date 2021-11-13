# -*- coding: utf-8 -*-
import numpy as np

from models.model import RelevanceModel
from models.tf_idf import normalize_cosine


class VectorModel(RelevanceModel):
    def __init__(self, doc_lemma_tf: np.ndarray, lemma_idf: np.ndarray, log_tf: bool = False):
        if log_tf:
            doc_lemma_tf = np.log10(doc_lemma_tf + 1)

        self.__log_tf = log_tf
        self.__doc_lemma_tf_idf = doc_lemma_tf * lemma_idf[np.newaxis, :]

    def __str__(self) -> str:
        return f"VectorModel(log_tf={self.__log_tf})"

    def get_serp_weights(self, queries_tf: np.ndarray) -> np.ndarray:
        """
        :param queries_tf: numpy array, dimensions: (query, lemma)
        :return: serp indexes
        """

        # NOTE: dimensions = (query, document, lemma)
        if self.__log_tf:
            queries_tf = np.log10(queries_tf + 1)

        query_vec = normalize_cosine(queries_tf)[:, np.newaxis, :]
        sentences_vec = normalize_cosine(self.__doc_lemma_tf_idf)[np.newaxis, :, :]

        np.sum(query_vec[:, np.newaxis, :] * sentences_vec[np.newaxis, :, :], axis=2)
        weights = np.sum(query_vec * sentences_vec, axis=2)
        return weights
