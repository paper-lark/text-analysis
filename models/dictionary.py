# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np

from tf_idf import lemmatize_sentence, get_sentence_lemma_tf, calculate_idf


@dataclass
class Dictionary:
    lemma_to_index: Dict[str, int]
    lemma_idf: np.ndarray
    sentence_lemma_tf: np.ndarray
    sentences: List[str]
    scores: np.ndarray

    def store(self, cache_dir: Path):
        np.save(str(cache_dir / "lemma_idf.npy"), self.lemma_idf, allow_pickle=False)
        np.save(str(cache_dir / "sentence_tf.npy"), self.sentence_lemma_tf, allow_pickle=False)
        with open(cache_dir / 'dict.json', 'w') as f:
            json.dump(self.lemma_to_index, f, ensure_ascii=False)
        with open(cache_dir / 'documents.json', 'w') as f:
            json.dump([dict(doc=doc, score=score) for doc, score in zip(self.sentences, self.scores.tolist())], f,
                      ensure_ascii=False)

    @staticmethod
    def read(cache_dir: Path, scores_filename: Optional[str] = None) -> Dictionary:
        idf = np.load(str(cache_dir / "lemma_idf.npy"))
        tf = np.load(str(cache_dir / "sentence_tf.npy"))
        with open(cache_dir / 'dict.json') as f:
            lemmas: Dict[str, int] = json.load(f)
        with open(cache_dir / (scores_filename or 'documents.json')) as f:
            entries = json.load(f)
            sentences: List[str] = [e['doc'] for e in entries]
            scores: List[int] = [e['score'] for e in entries]
        return Dictionary(lemmas, idf, tf, sentences, np.array(scores))

    @staticmethod
    def from_sentences(sentences: List[str]) -> Dictionary:
        sentence_lemmas = list(map(lemmatize_sentence, sentences))
        lemma_to_index = {
            e: i for i, e in enumerate(
                {item for sublist in sentence_lemmas for item in sublist})}

        sentence_lemma_tf = get_sentence_lemma_tf(sentence_lemmas, lemma_to_index)
        lemma_idf = calculate_idf(sentence_lemma_tf)
        return Dictionary(lemma_to_index, lemma_idf, sentence_lemma_tf, sentences, np.zeros_like((len(sentences),)))
