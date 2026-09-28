"""Иерархический индекс для кандидатогенерации.

Вместо плоского поиска по всем 189k объявлениям запрос спускается по дереву:

    Уровень 0  гео:        search_location_id -> распределение item_location_id (выучено по train)
    Уровень 1  рубрикатор: «Вид услуги» из фильтра запроса  (совпадает с объявлением в ~98% случаев)
    Уровень 2  микрокатегория: P(microcat | запрос) по ближайшим запросам из train
    Уровень 3  лист:       объявления выбранных узлов, ранжирование внутри них

Кандидаты собираются из трёх «веток» дерева:
    * гео-ветка      — все объявления в наиболее вероятных локациях;
    * рубрик-ветка   — объявления наиболее вероятных микрокатегорий из любых локаций
                       (для удалённых услуг: «обзвон по базе», «работа в фотошопе», ЭЦП…),
                       но только top-M ближайших по смыслу внутри каждой микрокатегории —
                       иначе ветка тащит в пул десятки тысяч объявлений со всей страны;
    * collab-ветка   — объявления, которые выбирали по похожим запросам в train.

Дальше двухступенчатое ранжирование:
    1) префильтр: простая линейная комбинация сигналов (score_linear) оставляет top-n_pre пула;
    2) обученный ранжировщик (ranker.py) по полному набору признаков выбирает итоговые 50.
"""
from collections import Counter, defaultdict

import numpy as np

from data import norm_query, service_kind

# Порядок признаков, подаваемых в ранжировщик (см. _candidates_one).
FEATURES = [
    # семантика
    "cos", "cos_gap", "cos_rank",
    # уровень 0: гео
    "geo", "same_loc", "loc_known", "loc_size",
    # уровни 1-2: рубрикатор
    "mc", "mc_top", "kind", "q_has_kind",
    # collab / популярность
    "col", "exact_q", "item_pop", "mc_pop",
    # качество объявления
    "rating", "reviews", "price", "phone_hidden", "msg_forbidden",
    # запрос
    "q_words",
]


class HierarchicalIndex:
    def __init__(self, items, item_emb, train, train_q_texts, train_q_emb):
        """items      — корпус (DataFrame, см. data.ITEM_COLS)
        item_emb   — эмбеддинги корпуса, (N, d), L2-нормированы
        train      — пары запрос-объявление, по которым учим априорные распределения
        train_q_texts / train_q_emb — уникальные нормализованные тексты запросов train и их эмбеддинги
        """
        self.item_ids = items.item_id.to_numpy()
        self.item_emb = item_emb
        self.item_loc = items.item_location_id.to_numpy()
        self.item_mc = items.item_microcat_id.to_numpy()
        self.item_kind = np.array([service_kind(p) for p in items.item_infm_params_text])
        self.id2row = {iid: i for i, iid in enumerate(self.item_ids)}

        # признаки качества объявления (NaN сохраняем — см. data.prepare_items)
        self.item_rating = items.item_rating.to_numpy(np.float32)
        self.item_reviews = np.log1p(items.item_rating_reviews_count.to_numpy(np.float32))
        self.item_price = np.log1p(items.item_price.to_numpy(np.float32))
        self.item_phone_hidden = items.item_is_phone_hidden.to_numpy(np.float32)
        self.item_msg_forbidden = items.item_is_message_forbidden.to_numpy(np.float32)

        # ---- листья дерева: списки строк корпуса по локации и по микрокатегории
        self.loc2rows = _group_rows(self.item_loc)
        self.mc2rows = _group_rows(self.item_mc)
        self.loc_size = {l: len(r) for l, r in self.loc2rows.items()}

        # ---- уровень 0: P(item_loc | search_loc) по train
        self.geo = defaultdict(Counter)
        for sl, il in zip(train.search_location_id.to_numpy(), train.item_location_id.to_numpy()):
            self.geo[sl][il] += 1

        # ---- популярность: сколько раз объявление / микрокатегорию выбирали в train (по любым запросам)
        pop = train.item_id.value_counts()
        self.item_pop = np.log1p(np.array([pop.get(i, 0) for i in self.item_ids], dtype=np.float32))
        mpop = train.item_microcat_id.value_counts(normalize=True)
        self.mc_pop = np.array([mpop.get(m, 0.0) for m in self.item_mc], dtype=np.float32)

        # ---- уровни 1-2 и collab: статистика по каждому уникальному тексту запроса train
        tq = train.search_query.map(norm_query).to_numpy()
        self.tq_index = {t: i for i, t in enumerate(train_q_texts)}
        self.tq_mc = [Counter() for _ in train_q_texts]
        self.tq_items = [Counter() for _ in train_q_texts]
        for t, mc, iid in zip(tq, train.item_microcat_id.to_numpy(), train.item_id.to_numpy()):
            j = self.tq_index[t]
            self.tq_mc[j][mc] += 1
            if iid in self.id2row:          # collab имеет смысл только для объявлений из корпуса
                self.tq_items[j][iid] += 1
        self.tq_emb = train_q_emb

    # ------------------------------------------------------------------ уровень 0: гео
    def geo_prior(self, loc, self_weight=3.0):
        """Распределение локаций объявлений для локации поиска.
        Своя локация получает псевдо-счётчик: для городов она и так доминирует,
        а для новых (не встречавшихся в train) локаций это единственный сигнал."""
        cnt = Counter(self.geo.get(loc, {}))
        tot = sum(cnt.values())
        cnt[loc] += self_weight * max(tot, 1) / 10 if tot else 1.0
        tot = sum(cnt.values())
        return {l: c / tot for l, c in cnt.items() if l in self.loc2rows}

    # ------------------------------------------------------------------ уровень 2: роутер по запросам train
    def neighbours(self, q_emb, k=30):
        """Ближайшие запросы train (косинус). 74k коротких векторов — «корневой» уровень дерева,
        в 2.5 раза меньше корпуса; считается матричными операциями по чанкам (экономим память)."""
        top_all, sim_all = [], []
        for i in range(0, len(q_emb), 512):
            sims = q_emb[i:i + 512] @ self.tq_emb.T
            top = np.argpartition(-sims, k, axis=1)[:, :k]
            top_all.append(top)
            sim_all.append(np.take_along_axis(sims, top, axis=1))
        return np.vstack(top_all), np.vstack(sim_all)

    # ------------------------------------------------------------------ спуск по дереву
    def candidates(self, q_embs, q_texts, q_locs, q_params, p):
        """Генератор по запросам: (строки корпуса после префильтра, матрица признаков FEATURES,
        линейный скор префильтра). Строки отсортированы по убыванию линейного скора."""
        nb_idx, nb_sim = self.neighbours(q_embs, p["knn"])
        for qi in range(len(q_embs)):
            yield self._candidates_one(q_embs[qi], q_texts[qi], q_locs[qi], q_params[qi],
                                       nb_idx[qi], nb_sim[qi], p)

    def _candidates_one(self, qe, q_raw, loc, params, nb_idx, nb_sim, p):
        qt, kind = norm_query(q_raw), service_kind(params)

        # --- распределения микрокатегорий и collab-объявлений по соседним запросам
        w = np.maximum(nb_sim, 0) ** p["knn_pow"]
        exact = self.tq_index.get(qt)
        mc_score, col_score = Counter(), Counter()
        for j, wj in zip(nb_idx, w):
            if j == exact:
                wj *= p["exact_boost"]          # точное совпадение текста запроса — самый сильный сигнал
            n = sum(self.tq_mc[j].values())
            for mc, c in self.tq_mc[j].items():
                mc_score[mc] += wj * c / n
            for iid, c in self.tq_items[j].items():
                col_score[iid] += wj * c / n
        z = sum(mc_score.values()) or 1.0
        mc_prob = {m: v / z for m, v in mc_score.items()}

        # --- гео-ветка: самые вероятные локации (накопленная вероятность до geo_mass)
        geo = self.geo_prior(loc)
        rows = []
        acc = 0.0
        for l, pr in sorted(geo.items(), key=lambda x: -x[1])[: p["geo_top"]]:
            rows.append(self.loc2rows[l])
            acc += pr
            if acc >= p["geo_mass"]:
                break
        # --- рубрик-ветка: топ микрокатегорий из любых локаций, внутри каждой — top-M по косинусу
        for m, _ in sorted(mc_prob.items(), key=lambda x: -x[1])[: p["mc_top"]]:
            if m in self.mc2rows:
                r = self.mc2rows[m]
                if p["mc_leaf"] and len(r) > p["mc_leaf"]:
                    r = r[np.argpartition(-(self.item_emb[r] @ qe), p["mc_leaf"])[: p["mc_leaf"]]]
                rows.append(r)
        # --- collab-ветка
        if col_score:
            rows.append(np.fromiter((self.id2row[i] for i in col_score), dtype=np.int64))
        pool = np.unique(np.concatenate(rows)) if rows else np.arange(len(self.item_ids))

        # --- базовые сигналы на всём пуле и префильтр
        cos = self.item_emb[pool] @ qe
        geo_v = np.array([geo.get(l, 0.0) for l in self.item_loc[pool]], dtype=np.float32)
        mc_v = np.array([mc_prob.get(c, 0.0) for c in self.item_mc[pool]], dtype=np.float32)
        kind_v = (self.item_kind[pool] == kind).astype(np.float32) if kind else np.zeros(len(pool), np.float32)
        col_v = (np.array([col_score.get(i, 0.0) for i in self.item_ids[pool]], dtype=np.float32)
                 if col_score else np.zeros(len(pool), np.float32))
        s = score_linear(cos, geo_v, mc_v, kind_v, col_v, p)
        keep = np.argsort(-s)[: p["n_pre"]]
        rows, s = pool[keep], s[keep]
        cos, geo_v, mc_v, kind_v, col_v = cos[keep], geo_v[keep], mc_v[keep], kind_v[keep], col_v[keep]

        # --- полный набор признаков для ранжировщика (порядок = FEATURES)
        n = len(rows)
        cos_rank = np.empty(n, np.float32)
        cos_rank[np.argsort(-cos)] = np.arange(n)
        const = lambda v: np.full(n, v, np.float32)
        X = np.column_stack([
            cos, cos - cos.max(), cos_rank,
            geo_v, (self.item_loc[rows] == loc).astype(np.float32), const(loc in self.geo),
            np.log1p(np.array([self.loc_size[l] for l in self.item_loc[rows]], np.float32)),
            mc_v, const(max(mc_prob.values()) if mc_prob else 0.0), kind_v, const(bool(kind)),
            col_v, const(exact is not None), self.item_pop[rows], self.mc_pop[rows],
            self.item_rating[rows], self.item_reviews[rows], self.item_price[rows],
            self.item_phone_hidden[rows], self.item_msg_forbidden[rows],
            const(len(qt.split())),
        ]).astype(np.float32)
        return rows, X, s


def score_linear(cos, geo, mc, kind, col, p):
    """Префильтр: семантическая близость + лог-априоры уровней иерархии.
    Его задача — не потерять нужное объявление в top-n_pre; точный порядок делает ранжировщик."""
    return (cos
            + p["w_geo"] * np.log(geo + p["eps_geo"])
            + p["w_mc"] * np.log(mc + p["eps_mc"])
            + p["w_kind"] * kind
            + p["w_col"] * np.log1p(col * 10))


DEFAULT_PARAMS = dict(knn=30, knn_pow=4.0, exact_boost=3.0,
                      geo_top=40, geo_mass=0.97, mc_top=4, mc_leaf=500, n_pre=1000,
                      w_geo=0.05, eps_geo=1e-3, w_mc=0.05, eps_mc=1e-3,
                      w_kind=0.05, w_col=0.3)


def _group_rows(keys):
    order = np.argsort(keys, kind="stable")
    uniq, start = np.unique(keys[order], return_index=True)
    return dict(zip(uniq, np.split(order, start[1:])))
