"""Локальная валидация: Recall@50 на отложенных группах train и абляции по шагам решения.

    python validate.py

Каждая строка таблицы — отдельная ступень решения, чтобы было видно, что даёт каждый шаг.
"""
import sys
import time

sys.path.insert(0, "src")

import numpy as np

import ranker
from data import load_items, load_queries, load_train, recall_at_k
from index import DEFAULT_PARAMS, FEATURES
from pipeline import build_index, candidates, make_splits, query_embeddings, train_ranker

train, corpus, bench = load_train(), load_items(), load_queries()
stats, rank_q, val_q, dev_corpus, _ = make_splits(train, corpus, bench)
print(f"stats={len(stats)} rank_queries={len(rank_q)} val_queries={len(val_q)} dev_corpus={len(dev_corpus)}")
rel = val_q.rel.tolist()
P = DEFAULT_PARAMS

index = build_index(stats, dev_corpus)
model = train_ranker(index, rank_q, P)
ids = index.item_ids
q_emb = query_embeddings(val_q)
results = []

# 1. плоский поиск по косинусу во всём корпусе (то, от чего уходим)
flat = []
for i in range(0, len(q_emb), 200):     # чанками, чтобы матрица сходств не съела всю память
    s = q_emb[i:i + 200] @ index.item_emb.T
    flat += [list(ids[r]) for r in np.argpartition(-s, 50, axis=1)[:, :50]]
results.append(("плоский косинус по всему корпусу", recall_at_k(flat, rel)))

# 2. только своя локация + косинус
pred = []
for qe, loc in zip(q_emb, val_q.search_location_id):
    rows = index.loc2rows.get(loc, np.arange(len(ids)))
    pred.append(list(ids[rows[np.argsort(-(index.item_emb[rows] @ qe))[:50]]]))
results.append(("своя локация + косинус", recall_at_k(pred, rel)))

# 3-4. иерархический индекс с линейным скором: без сужения рубрик-ветки и с ним
for name, pp in [("иерархия + линейный скор, полная рубрик-ветка", {**P, "mc_leaf": 0}),
                 ("иерархия + линейный скор, суженная рубрик-ветка", P)]:
    cands = list(candidates(index, val_q, pp))       # rows отсортированы по линейному скору
    results.append((name, recall_at_k([list(ids[r[:50]]) for r, _, _ in cands], rel)))
results.append((f"  └ полнота префильтра (top-{P['n_pre']})",
                recall_at_k([list(ids[r]) for r, _, _ in cands], rel, k=P["n_pre"])))

# 5. ранжировщик без признаков популярности/качества — чтобы увидеть вклад шага 2
base_feats = [i for i, f in enumerate(FEATURES) if f not in
              {"item_pop", "mc_pop", "rating", "reviews", "price", "phone_hidden", "msg_forbidden"}]
m_base = train_ranker(index, rank_q, P, base_feats)
results.append(("+ ранжировщик (без популярности/качества)",
                recall_at_k(ranker.top_k(m_base, ((r, X[:, base_feats], s) for r, X, s in cands), ids), rel)))

# 6. полный ранжировщик — итоговое решение
t = time.time()
pred = ranker.top_k(model, candidates(index, val_q, P), ids)
ms = (time.time() - t) / len(val_q) * 1000
results.append(("+ популярность и качество объявления (итог)", recall_at_k(pred, rel)))

print("\nRecall@50 на валидации")
for name, r in results:
    print(f"  {name:52s} {r:.4f}")
print(f"\nскорость итогового решения: {ms:.1f} мс/запрос")

# --- разбор ошибок итогового решения: где теряется нужное объявление и на каких запросах
pool = [set(ids[r]) for r, _, _ in cands]            # пул после префильтра (top-n_pre)
known_locs = set(index.loc2rows)
lost_pre = lost_rank = n_rel = 0
for p_, pl, rel_ in zip(pred, pool, rel):
    for i in set(rel_):
        n_rel += 1
        lost_pre += i not in pl
        lost_rank += i in pl and i not in p_[:50]
print(f"\nРазбор ошибок: релевантных {n_rel}, потеряно {lost_pre + lost_rank} "
      f"({(lost_pre + lost_rank) / n_rel:.1%})")
print(f"  не попали в пул префильтра:        {lost_pre / n_rel:.1%}")
print(f"  в пуле, но ранжировщик ниже top-50: {lost_rank / n_rel:.1%}")

rec = np.array([len(set(p_[:50]) & set(r_)) / len(set(r_)) for p_, r_ in zip(pred, rel)])
seg = {
    "пустой фильтр":                     val_q.search_infm_params_text.eq("").to_numpy(),
    "фильтр задан":                      val_q.search_infm_params_text.ne("").to_numpy(),
    "локации поиска нет в корпусе":      ~val_q.search_location_id.isin(known_locs).to_numpy(),
    "текст запроса встречался в train":  val_q.search_query.isin(set(stats.search_query)).to_numpy(),
    "нужное объявление не из локации поиска": np.array(
        [all(index.item_loc[index.id2row[i]] != l for i in r_ if i in index.id2row)
         for r_, l in zip(rel, val_q.search_location_id)]),
}
print("\nRecall@50 по сегментам запросов")
for name, m in seg.items():
    print(f"  {name:42s} доля {m.mean():5.1%}   R@50 {rec[m].mean():.4f}")
