# -*- coding: utf-8 -*-
import numpy as np


def calculate_dcg(serp_scores: np.ndarray):
    return np.sum(serp_scores / np.log2(2 + np.indices(serp_scores.shape)))


def calculate_ndcg(serp_scores: np.ndarray) -> float:
    best_scores = np.flip(np.sort(serp_scores))
    return calculate_dcg(serp_scores) / calculate_dcg(best_scores)
