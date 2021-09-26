# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


def calculate_score(doc_w, query_w):
    # doc_w = doc_w * np.heaviside(query_w, 0)
    # TODO: normalize
    return 1 - cosine(doc_w, query_w)


def calculate_scores(w, docs):
    scores = sorted([
        (d, calculate_score(w[d], w['query']))
        for d in docs], key=lambda x: x[1], reverse=True)
    return scores


def calculate_weights(df, docs, f):
    return pd.DataFrame({
        d: f(df[d]) * (df['idf'] if d == 'query' else 1)
        for d in docs + ['query']
    })


def print_result(schema, scores):
    print(f'Результат по схеме {schema}: {scores}')


def main():
    df = pd.DataFrame({
        'lemma': ['car', 'auto', 'insurance', 'best'],
        'df': [18165, 6723, 19241, 25235],
        'idf': [1.65, 2.08, 1.62, 1.5],
        'd1': [27, 3, 0, 14],
        'd2': [4, 33, 33, 0],
        'd3': [24, 0, 29, 17]
    })
    query = 'car insurance'

    query_words = query.split(' ')
    df['query'] = df['lemma'].map(lambda x: query_words.count(x))
    docs = ['d1', 'd2', 'd3']
    # w_1 = pd.DataFrame({d: df[d] * (df['idf'] if d == 'query' else 1) for d in docs + ['query']})
    # w_2 = pd.DataFrame({d: np.log(df[d] + 1) * df['idf'] for d in docs + ['query']})
    w_1 = calculate_weights(df, docs, lambda x: x)
    w_2 = calculate_weights(df, docs, lambda x: np.log(x + 1))

    scores_1 = calculate_scores(w_1, docs)
    scores_2 = calculate_scores(w_2, docs)

    print_result('nnc.ntc', scores_1)
    print_result('lnc.ltc', scores_2)


if __name__ == '__main__':
    main()