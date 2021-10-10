# -*- coding: utf-8 -*-
import argparse
import pathlib
from typing import List

import pandas as pd

from extract import ArticleSource, extract_sentences_from_sources
from dictionary import Dictionary
from query import get_relevant_sentences
from utils import create_dir


def main():
    # parse args
    parser = argparse.ArgumentParser(description='Run tf-idf vector model')
    parser.add_argument('-b', '--build-vocabulary', dest='should_build_vocabulary',
                        action=argparse.BooleanOptionalAction,
                        help='should build vocabulary or use cache (default: use cache)')
    parser.add_argument('-l', '--log-tf', dest='should_use_log_tf',
                        action=argparse.BooleanOptionalAction,
                        help='should use logarithmic formula for tf (default: false)')
    parser.add_argument('-q', '--query', dest='query', type=str, default='',
                        help='query to process (default: all fact queries)')
    parser.add_argument('-n', dest='top_n', type=str, default=10,
                        help='top results to return (default: 10)')

    args = parser.parse_args()
    should_build_vocabulary = args.should_build_vocabulary
    should_use_log_tf = args.should_use_log_tf
    query = args.query
    top_n = args.top_n

    # get dictionary
    vocab_dir = pathlib.Path('cache')
    if should_build_vocabulary:
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
    else:
        print("Using vocabulary from cache")
        vocab = Dictionary.read(vocab_dir)

    # create results folder
    results_dir = pathlib.Path('results')
    create_dir(results_dir)

    # process query
    if query:
        print(f"Processing query: {query}")
        df = get_relevant_sentences(query, vocab, top_n, log_tf=should_use_log_tf)
        save_results(df, results_dir, f'result')
    else:
        print(f"Processing fact queries")
        default_queries = [
            "По мнению Кеплера, Вифлеемская звезда могла быть Великим соединением.",
            "Вор, которому смертную казнь заменили каторгой, стал одним из богатейших людей Австралии",
            "Творческому порыву слепого музыканта поспособствовали четыре дня в коме."
        ]
        for i, q in enumerate(default_queries):
            for should_use_log_tf in (True, False):
                df = get_relevant_sentences(q, vocab, top_n, log_tf=should_use_log_tf)
                postfix = "log" if should_use_log_tf else "plain"
                save_results(df, results_dir, f'result{i}_{postfix}')


def save_results(df: pd.DataFrame, results_dir: pathlib.Path, file_name: str):
    df.to_csv(results_dir / f'{file_name}.csv', index=True, index_label='index')
    df.to_markdown(results_dir / f'{file_name}.md', index=False)


if __name__ == '__main__':
    main()
