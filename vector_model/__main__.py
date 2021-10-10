# -*- coding: utf-8 -*-
import os
import sys
from typing import List

from extract import ArticleSource, extract_sentences_from_sources
from dictionary import Dictionary
from query import get_relevant_sentences


def main():
    # parse args
    should_prepare_dict = len(sys.argv) == 2 and sys.argv[1] == "build-dict"

    # get dictionary
    if should_prepare_dict:
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
        vocab.store()
    else:
        vocab = Dictionary.read()

    # create results folder
    results_dir = 'results'
    try:
        os.mkdir(results_dir, mode=0o755)
    except FileExistsError:
        pass

    # process query
    queries = [
        "По мнению Кеплера, Вифлеемская звезда могла быть Великим соединением.",
        "Вор, которому смертную казнь заменили каторгой, стал одним из богатейших людей Австралии",
        "Творческому порыву слепого музыканта поспособствовали четыре дня в коме."
    ]
    for i, q in enumerate(queries):
        for should_use_log_tf in (True, False):
            df = get_relevant_sentences(q, vocab, log_tf=should_use_log_tf)
            postfix = "log" if should_use_log_tf else "plain"
            df.to_csv(f'{results_dir}/result{i}_{postfix}.csv', index=True, index_label='index')


if __name__ == '__main__':
    main()
