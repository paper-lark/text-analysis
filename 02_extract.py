import functools
import itertools
import re
from collections import Counter
from typing import List
from bs4 import BeautifulSoup
import requests
from pymystem3 import Mystem
import os
from dataclasses import dataclass
import pandas as pd


@dataclass
class ArticleSource:
    name: str
    link: str


def fetch_article(link: str) -> str:
    r = requests.get(link)
    r.raise_for_status()
    return r.text


def extract_text(page: str) -> str:
    soup = BeautifulSoup(page, 'html.parser')
    d = soup.find("div", {'id': 'mw-content-text'})
    d.sup.decompose()
    ext_links = d.find('div', {'data-name': 'External links'})
    if ext_links:
        ext_links.extract()
    [x.extract() for x in d.findAll('div', {'role': 'navigation'})]
    [x.extract() for x in d.select('div.thumb')]
    [x.extract() for x in d.select('div.printfooter')]
    return d.get_text()


def main():
    links: List[ArticleSource] = [
        ArticleSource('Вифлеемская звезда', 'https://ru.wikipedia.org/wiki/Вифлеемская_звезда'),
        ArticleSource('Великое соединение', 'https://ru.wikipedia.org/wiki/Великое_соединение'),
        ArticleSource('Три царя', 'https://ru.wikipedia.org/wiki/%D0%A2%D1%80%D0%B8_%D1%86%D0%B0%D1%80%D1%8F'),
        ArticleSource('Уильям Хатчинсон', 'https://ru.wikipedia.org/wiki/Хатчинсон,_Уильям'),
        ArticleSource('Fulfillingness First Finale', 'https://ru.wikipedia.org/wiki/Fulfillingness’_First_Finale'),
        ArticleSource('Стиви Уандер', 'https://ru.wikipedia.org/wiki/Стиви_Уандер'),
        ArticleSource('Innervisions', 'https://ru.wikipedia.org/wiki/Innervisions')
    ]

    # prepare dataset
    try:
        os.mkdir('02_dataset', mode=0o755)
    except FileExistsError:
        pass

    pages = []
    for link in links:
        filename = f'{link.name}.html'
        filepath = os.path.join("02_dataset", filename)
        if os.path.exists(filepath):
            print(f'File={filename} exists. Skipping')
            with open(filepath, 'r') as f:
                pages.append(f.read())
            continue

        t = fetch_article(link.link)
        with open(filepath, 'w') as f:
            f.write(t)
        pages.append(t)
        print(f'Saved file={filename}')

    # parse documents and count lemmas
    lemma_cnt = Counter()
    m = Mystem()
    word_re = re.compile(r'[a-zA-ZА-Яа-яёЁ]+')

    for p in pages:
        text = extract_text(p)
        for lemma in m.lemmatize(text):
            if word_re.fullmatch(lemma):
                lemma_cnt[lemma] += 1

    # create dataframe with frequencies
    lemmas = []
    frequencies = []
    for p in lemma_cnt.most_common():
        lemmas.append(p[0])
        frequencies.append(p[1])
    df = pd.DataFrame({'lemma': lemmas, 'frequency': frequencies})
    df.to_csv('02_frequencies.csv', index=False)


if __name__ == '__main__':
    main()
