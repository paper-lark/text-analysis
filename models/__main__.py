# -*- coding: utf-8 -*-
import argparse
import pathlib
from typing import List

import pandas as pd
import numpy as np

from extract import ArticleSource, extract_sentences_from_sources
from dictionary import Dictionary
from model import RelevanceModel
from models.language_model import LanguageModel
from ndcg import calculate_ndcg
from vector_model import VectorModel
from query import get_relevant_sentences
from utils import create_dir


def main():
    # parse args
    parser = argparse.ArgumentParser(description='Run tf-idf vector model')
    parser.add_argument('-b', '--build-vocabulary', dest='should_build_vocabulary',
                        action=argparse.BooleanOptionalAction,
                        help='should build vocabulary or use cache (default: use cache)')
    parser.add_argument('-q', '--query', dest='query', type=str, default=None,
                        help='query to process (default: all fact queries)')
    parser.add_argument('-s', '--scores', dest='scores_file', type=str, default=None,
                        help='scores file name in /cache folder, required if query is specified')

    args = parser.parse_args()
    should_build_vocabulary = args.should_build_vocabulary
    query = args.query
    scores_file = args.scores_file

    # build dictionary if necessary
    vocab_dir = pathlib.Path('cache')
    if should_build_vocabulary:
        build_dictionary(vocab_dir)
        return

    # create results folder
    results_dir = pathlib.Path('results')
    create_dir(results_dir)

    # process query if specified
    if query:
        print(f'Processing query = "{query}"')
        if not scores_file:
            raise ValueError(f'missing required scores parameter')

        vocab = Dictionary.read(vocab_dir, scores_file)
        model = LanguageModel(doc_lemma_tf=vocab.sentence_lemma_tf, la=0.5)
        ndcg = calculate_serp_ndcg(query, model, vocab, results_dir, "results")
        print(f'{str(model)}: NDCG = {ndcg:.5}')
        return

    # process all queries
    print(f"Processing fact queries")
    default_queries = [
        dict(query="По мнению Кеплера, Вифлеемская звезда могла быть Великим соединением.",
             scores='bethlehem_scores.json'),
        dict(query="Вор, которому смертную казнь заменили каторгой, стал одним из богатейших людей Австралии.",
             scores='hutchinson_scores.json'),
        dict(query="Творческому порыву слепого музыканта поспособствовали четыре дня в коме.",
             scores='stevie_wonder_scores.json')
    ]

    ndcg_results = np.zeros((3, 4), dtype=float)
    for i, desc in enumerate(default_queries):
        query, scores = desc['query'], desc['scores']
        vocab = Dictionary.read(vocab_dir, scores)
        print(f'Processing query = "{query}"')

        for j, model in enumerate([
            VectorModel(doc_lemma_tf=vocab.sentence_lemma_tf, lemma_idf=vocab.lemma_idf, log_tf=False),
            VectorModel(doc_lemma_tf=vocab.sentence_lemma_tf, lemma_idf=vocab.lemma_idf, log_tf=True),
            LanguageModel(doc_lemma_tf=vocab.sentence_lemma_tf, la=0.5),
            LanguageModel(doc_lemma_tf=vocab.sentence_lemma_tf, la=0.9)
        ]):
            ndcg = calculate_serp_ndcg(query, model, vocab, results_dir, f'result{i + 1}_{j + 1}')
            ndcg_results[i, j] = ndcg
            print(f'{str(model)}: NDCG = {ndcg:.5}')

    print("Mean results for models:", np.mean(ndcg_results, axis=0))


def build_dictionary(vocab_dir: pathlib.Path):
    print("Building vocabulary")
    links: List[ArticleSource] = [
        ArticleSource('Вифлеемская звезда', 'https://ru.wikipedia.org/wiki/Вифлеемская_звезда'),
        ArticleSource('Великое соединение', 'https://ru.wikipedia.org/wiki/Великое_соединение'),
        ArticleSource('Три царя', 'https://ru.wikipedia.org/wiki/%D0%A2%D1%80%D0%B8_%D1%86%D0%B0%D1%80%D1%8F'),
        ArticleSource('Уильям Хатчинсон', 'https://ru.wikipedia.org/wiki/Хатчинсон,_Уильям'),
        ArticleSource('Fulfillingness First Finale', 'https://ru.wikipedia.org/wiki/Fulfillingness’_First_Finale'),
        ArticleSource('Стиви Уандер', 'https://ru.wikipedia.org/wiki/Стиви_Уандер'),
        ArticleSource('Innervisions', 'https://ru.wikipedia.org/wiki/Innervisions')
    ]
    sentences = extract_sentences_from_sources(links)

    # prepare dictionary with idf
    vocab = Dictionary.from_sentences(sentences)
    create_dir(vocab_dir)
    vocab.store(vocab_dir)


def calculate_serp_ndcg(query: str, model: RelevanceModel, vocab: Dictionary, results_dir: pathlib.Path, results_filename: str) -> float:
    df = get_relevant_sentences(query, vocab, model)
    ndcg = calculate_ndcg(df['relevance'].to_numpy())
    save_results(df, results_dir, results_filename)
    return ndcg


def save_results(df: pd.DataFrame, results_dir: pathlib.Path, file_name: str):
    df.to_csv(str(results_dir / f'{file_name}.csv'), index=True, index_label='index')
    df.to_markdown(str(results_dir / f'{file_name}.md'), index=False)


if __name__ == '__main__':
    main()
