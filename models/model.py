# -*- coding: utf-8 -*-
import abc
import numpy as np


class RelevanceModel(abc.ABC):
    @abc.abstractmethod
    def get_serp_weights(self, queries_tf: np.ndarray) -> np.ndarray:
        """
        :param queries_tf: numpy array, dimensions: (query, lemma)
        :return: serp weights, the higher the better
        """
        raise NotImplementedError("method is abstract")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        :return: relevance model description as a string
        """
        raise NotImplementedError("method is abstract")
