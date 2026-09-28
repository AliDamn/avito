"""Общий пайплайн: разбиение train, построение индекса, получение кандидатов.
Используется и в локальной валидации, и в финальном предсказании — валидируется ровно тот код,
который генерирует answer.csv."""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data import DATA_DIR, items_texts, norm_query, prepare_items, query_text
from embed import encode
from index import HierarchicalIndex

GROUP_KEY = ["search_query", "search_location_id", "search_infm_params_text", "search_category"]
BIG_LOCS = [637640, 653240, 107620, 621540, 107621]   # крупнейшие локации бенчмарка (см. eda.py)


def build_index(train, corpus):
    item_emb = encode(items_texts(corpus), "items")
    tq_texts = sorted(set(train.search_query.map(norm_query)))
    tq_emb = encode([f"query: {t}" for t in tq_texts], "train_queries")
    return HierarchicalIndex(corpus, item_emb, train, tq_texts, tq_emb)


def query_embeddings(queries):
    return encode([query_text(q, pr) for q, pr in zip(queries.search_query, queries.search_infm_params_text)],
                  "queries")


def candidates(index, queries, params):
    """Кандидаты и признаки по каждому запросу (см. HierarchicalIndex.candidates)."""
    return index.candidates(query_embeddings(queries), queries.search_query.tolist(),
                            queries.search_location_id.to_numpy(),
                            queries.search_infm_params_text.tolist(), params)


# --------------------------------------------------------------------------- разбиение train

def _stratum(loc, params):
    """Страта запроса: (пустой ли фильтр, одна из крупных локаций или «прочие»).
    По этим двум признакам бенчмарк сильнее всего отличается от train (см. eda.py)."""
    return (params == "") * 10 + (BIG_LOCS.index(loc) + 1 if loc in BIG_LOCS else 0)


def make_splits(train, corpus, bench_queries, n_val=2500, n_rank=6000, seed=42):
    """Делит train на три непересекающиеся части по группам запросов:

        stats — по ней строится индекс (гео-переходы, микрокатегории, collab, популярность);
        rank  — n_rank групп для обучения ранжировщика: признаки для них считаются по индексу из stats,
                т.е. так же «вслепую», как для настоящих тестовых запросов (без утечки меток в collab);
        val   — n_val групп для оценки Recall@50.

    Группа = уникальная комбинация (текст, локация, фильтры, категория) — ровно как запрос бенчмарка.
    EDA показал сдвиг: в бенчмарке каждый текст встречается один раз, фильтр чаще пустой (63% vs 35%),
    а запросы чаще из крупных городов/регионов. Поэтому группы для rank/val выбираются с весами
    P_bench(страта) / P_train(страта) и не более одной группы на текст запроса.

    Выбранные в rank/val объявления добавляются в корпус (бенчмарк-корпус + они), чтобы
    релевантные тонули среди тех же 189k «чужих» объявлений, что и в настоящем тесте.
    Возвращает: stats, rank_q, val_q, dev_corpus, final_stats
        (у rank_q/val_q колонка rel — список релевантных item_id;
         final_stats = весь train без rank-групп — по нему строится индекс для настоящего бенчмарка:
         val-группы возвращаются в статистику, а rank-группы исключены, т.к. на них обучен ранжировщик
         и их признаки должны оставаться «слепыми», как у тестовых запросов).
    """
    gid = train.groupby(GROUP_KEY, sort=False).ngroup().to_numpy()
    groups = train.groupby(gid)[["search_query", "search_location_id", "search_infm_params_text"]].first()
    groups = groups.sample(frac=1, random_state=seed).drop_duplicates("search_query")  # один текст — одна группа

    st = np.array([_stratum(l, p) for l, p in zip(groups.search_location_id, groups.search_infm_params_text)])
    bst = np.array([_stratum(l, p) for l, p in zip(bench_queries.search_location_id,
                                                    bench_queries.search_infm_params_text)])
    p_bench = pd.Series(bst).value_counts(normalize=True)
    p_train = pd.Series(st).value_counts(normalize=True)
    w = pd.Series(st).map(p_bench / p_train).fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(groups.index.to_numpy(), n_val + n_rank, replace=False, p=w / w.sum())
    val_g, rank_g = chosen[:n_val], chosen[n_val:]

    def to_queries(g):
        part = train[np.isin(gid, g)].assign(gid=gid[np.isin(gid, g)])
        return part.groupby("gid", sort=True).agg(**{c: (c, "first") for c in GROUP_KEY},
                                                   rel=("item_id", lambda s: list(set(s)))).reset_index(drop=True)

    stats = train[~np.isin(gid, chosen)].reset_index(drop=True)
    final_stats = train[~np.isin(gid, rank_g)].reset_index(drop=True)
    val_q, rank_q = to_queries(val_g), to_queries(rank_g)

    # тексты отложенных объявлений читаем точечно (pyarrow filter), не загружая все описания train
    need = sorted({i for r in pd.concat([val_q.rel, rank_q.rel]) for i in r} - set(corpus.item_id))
    extra = pq.read_table(os.path.join(DATA_DIR, "train.parquet"), columns=[c for c in corpus.columns],
                          filters=[("item_id", "in", need)]).to_pandas().drop_duplicates("item_id")
    dev_corpus = pd.concat([corpus, prepare_items(extra)], ignore_index=True)
    return stats, rank_q, val_q, dev_corpus, final_stats


# --------------------------------------------------------------------------- ранжировщик

def train_ranker(index, rank_q, params, feature_idx=None):
    """Считает кандидатов для rank-запросов по индексу (построенному БЕЗ этих запросов) и обучает ранжировщик.
    feature_idx — подмножество признаков (для абляций), None = все."""
    import ranker
    cands = candidates(index, rank_q, params)
    if feature_idx is not None:
        cands = ((r, X[:, feature_idx], s) for r, X, s in cands)
    X, y = ranker.build_training_set(cands, rank_q.rel, index.item_ids)
    return ranker.fit(X, y)
