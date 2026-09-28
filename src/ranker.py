"""Обученный ранжировщик второй ступени.

Вместо ручной формулы веса сигналов учатся на данных: градиентный бустинг (sklearn
HistGradientBoostingClassifier) предсказывает вероятность того, что пользователь выберет объявление,
по признакам из index.FEATURES. Бустинг сам находит нелинейности и взаимодействия
(например, что для регионов гео-сигнал важен меньше, чем для городов) и нативно обрабатывает NaN
(нет рейтинга/отзывов/цены).

Обучающие примеры: запросы из rank-части train, кандидаты после префильтра.
Позитивы — выбранные пользователем объявления, негативы — остальные кандидаты того же запроса
(все «трудные» из верхушки префильтра + случайная подвыборка остальных).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

N_HARD, N_RANDOM = 100, 100   # негативов на запрос: верх префильтра + случайные из остальных


def build_training_set(cands, rel_lists, item_ids, seed=0):
    """cands — итератор (rows, X, s) по запросам rank-части; rel_lists — списки релевантных item_id."""
    rng = np.random.default_rng(seed)
    Xs, ys, n_q, n_pos_lost = [], [], 0, 0
    for (rows, X, _), rel in zip(cands, rel_lists):
        y = np.isin(item_ids[rows], list(rel)).astype(np.int8)
        n_pos_lost += len(set(rel)) - y.sum()
        if y.sum() == 0:
            continue                                  # нужное объявление не прошло префильтр — учиться не на чем
        neg = np.flatnonzero(y == 0)
        hard = neg[:N_HARD]                           # rows отсортированы по линейному скору
        rest = neg[N_HARD:]
        rnd = rng.choice(rest, min(N_RANDOM, len(rest)), replace=False) if len(rest) else rest
        idx = np.concatenate([np.flatnonzero(y == 1), hard, rnd])
        Xs.append(X[idx])
        ys.append(y[idx])
        n_q += 1
    print(f"ranker train: {n_q} запросов, {sum(map(len, ys))} примеров, "
          f"позитивов не прошло префильтр: {n_pos_lost}")
    return np.vstack(Xs), np.concatenate(ys)


def fit(X, y, seed=0):
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=50, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=30, random_state=seed)
    return model.fit(X, y)


def top_k(model, cands, item_ids, k=50):
    """Итоговые top-k item_id по каждому запросу."""
    out = []
    for rows, X, _ in cands:
        s = model.predict_proba(X)[:, 1]
        out.append(list(item_ids[rows[np.argsort(-s)[:k]]]))
    return out
