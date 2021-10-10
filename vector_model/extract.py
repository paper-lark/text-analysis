# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import List, Optional, Iterable

from bs4 import BeautifulSoup
import requests
import os
from dataclasses import dataclass
from razdel import sentenize
import unicodedata

from utils import create_dir


@dataclass
class ArticleSource:
    name: str
    link: str


def fetch_page(link: str) -> str:
    r = requests.get(link)
    r.raise_for_status()
    return r.text


def extract_text_from_page(page: str) -> str:
    # get main page contents
    soup = BeautifulSoup(page, 'html.parser')
    d = soup.find("div", {'id': 'mw-content-text'})
    d.sup.decompose()

    # remove external links section
    ext_links = d.find('div', {'data-name': 'External links'})
    if ext_links:
        ext_links.extract()

    # remove insignificant elements by type
    for i in range(1, 7):
        [x.extract() for x in d.select(f'h{i}')]
    [x.extract() for x in d.select(f'table')]
    [x.extract() for x in d.select(f'cite')]

    # remove insignificant elements by class
    [x.extract() for x in d.findAll('div', {'role': 'navigation'})]
    [x.extract() for x in d.select('div.thumb')]
    [x.extract() for x in d.select('div.printfooter')]
    [x.extract() for x in d.select('div.reflist')]
    [x.extract() for x in d.select('div.ts-Конец_цитаты-source')]
    [x.extract() for x in d.select('div.mw-references-wrap')]
    [x.extract() for x in d.select('.references')]
    [x.extract() for x in d.select('span.citation')]

    # remove references
    text = d.get_text()
    text = re.sub(r"\[\d+\]", "", text)

    # normalize spaces
    text = re.sub(r'[   ⁠]', ' ', text)

    # normalize text
    text = strip_accents(text)
    return text


def strip_accents(s: str, accents=('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT')) -> str:
    accents = set(map(unicodedata.lookup, accents))
    chars = [c for c in unicodedata.normalize('NFD', s) if c not in accents]
    return unicodedata.normalize('NFC', ''.join(chars))


def get_pages(links: Iterable[ArticleSource], cache_dir: Optional[Path]) -> List[str]:
    pages: List[str] = []
    pages_dir = cache_dir or Path('cache', 'pages')
    create_dir(pages_dir)

    for link in links:
        filename = f'{link.name}.html'
        filepath = pages_dir / filename
        if os.path.exists(filepath):
            print(f'File={filename} exists. Skipping')
            with open(filepath, 'r') as f:
                pages.append(f.read())
            continue

        t = fetch_page(link.link)
        with open(filepath, 'w') as f:
            f.write(t)
        pages.append(t)
        print(f'Saved file={filename}')

    return pages


def get_sentences(texts: Iterable[str]) -> List[str]:
    result = []
    for text in texts:
        sentences = list(map(lambda x: re.sub(r"\n+", " ", x.text), sentenize(text)))
        result += sentences
    return result


def extract_sentences_from_sources(links: List[ArticleSource], page_cache_dir: Optional[Path] = None) -> List[str]:
    texts = map(extract_text_from_page, get_pages(links, page_cache_dir))
    return get_sentences(texts)
