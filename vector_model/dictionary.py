# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import numpy as np

from tf_idf import lemmatize_sentence, get_sentence_lemma_tf, calculate_idf


@dataclass
class Dictionary:
    lemma_to_index: Dict[str, int]
    lemma_idf: np.ndarray
    sentence_lemma_tf: np.ndarray
    sentences: List[str]

    def store(self, cache_dir: Path):
        np.save(cache_dir / "lemma_idf.npy", self.lemma_idf, allow_pickle=False)
        np.save(cache_dir / "sentence_tf.npy", self.sentence_lemma_tf, allow_pickle=False)
        with open(cache_dir / 'dict.json', 'w') as f:
            json.dump(dict(lemmas=self.lemma_to_index, sentences=self.sentences), f, ensure_ascii=False)

    @staticmethod
    def read(cache_dir: Path) -> Dictionary:
        idf = np.load(cache_dir / "lemma_idf.npy")
        tf = np.load(cache_dir / "sentence_tf.npy")
        with open(cache_dir / 'dict.json', 'r') as f:
            res = json.load(f)
            lemmas: Dict[str, int] = res['lemmas']
            sentences: List[str] = res['sentences']
        return Dictionary(lemmas, idf, tf, sentences)

    @staticmethod
    def from_sentences(sentences: List[str]) -> Dictionary:
        sentence_lemmas = list(map(lemmatize_sentence, sentences))
        lemma_to_index = {
            e: i for i, e in enumerate(
                {item for sublist in sentence_lemmas for item in sublist})}

        sentence_lemma_tf = get_sentence_lemma_tf(sentence_lemmas, lemma_to_index)
        lemma_idf = calculate_idf(sentence_lemma_tf)
        return Dictionary(lemma_to_index, lemma_idf, sentence_lemma_tf, sentences)
