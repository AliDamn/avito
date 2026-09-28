"""Финальное решение: строит answer.csv для benchmark_queries.parquet.

    python run.py

Шаги:
  1. Разбиение train (то же, что в validate.py, фиксированный seed).
  2. Обучение ранжировщика: индекс по stats-части, признаки для rank-запросов.
  3. Финальный индекс: статистика по всему train, кроме rank-групп; корпус — ровно benchmark_items.
  4. Кандидаты + ранжировщик для 2452 запросов бенчмарка -> top-50 -> answer.csv (+ проверка формата).

Все вычисления локальные; эмбеддинги берутся из cache/ (если их нет — считаются локальной моделью).
"""
import gc
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import ranker
from data import load_items, load_queries, load_train
from index import DEFAULT_PARAMS
from pipeline import build_index, candidates, make_splits, train_ranker

P = DEFAULT_PARAMS

train, corpus, bench = load_train(), load_items(), load_queries()
stats, rank_q, val_q, dev_corpus, final_stats = make_splits(train, corpus, bench)

# --- 2. ранжировщик
dev_index = build_index(stats, dev_corpus)
model = train_ranker(dev_index, rank_q, P)
del dev_index
gc.collect()                                   # на 8 ГБ не держим два индекса одновременно

# --- 3-4. финальный индекс и предсказания
index = build_index(final_stats, corpus)
pred = ranker.top_k(model, candidates(index, bench, P), index.item_ids, k=50)

answer = pd.DataFrame({"query_id": bench.query_id, "answer": [" ".join(p) for p in pred]})
answer.to_csv("answer.csv", index=False)

# --- проверка требований к файлу из условия
chk = pd.read_csv("answer.csv", dtype=str)
corpus_ids = set(corpus.item_id)
lists = chk.answer.fillna("").str.split()
assert list(chk.columns) == ["query_id", "answer"]
assert len(chk) == len(bench) and set(chk.query_id) == set(bench.query_id) and chk.query_id.is_unique
assert chk.query_id.str.len().eq(16).all()
assert lists.map(len).le(50).all() and lists.map(lambda l: len(l) == len(set(l))).all()
assert all(i in corpus_ids for l in lists for i in l)
print(f"answer.csv: {len(chk)} строк, кандидатов на запрос: min={lists.map(len).min()} "
      f"mean={lists.map(len).mean():.1f}; формат проверен")
