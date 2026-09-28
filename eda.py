"""Разведочный анализ данных (EDA). Выводы используются в решении — см. README, раздел «Анализ данных».

    python eda.py
"""
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from data import load_items, load_queries, load_train

pd.set_option("display.width", 200)
K = ["search_query", "search_location_id", "search_infm_params_text", "search_category"]

q = load_queries()
it = pd.read_parquet(__import__("os").path.join(__import__("data").DATA_DIR, "benchmark_items.parquet"))
tr = load_train()

# 1. Пропуски. NaN есть только в рейтинге/кол-ве отзывов (нет отзывов), в 35 описаниях корпуса
#    (это вакансии, категория 112) и в 1 паре координат. Пустой фильтр запроса — это "", а не NaN.
print("== NaN по колонкам (только ненулевые)")
for name, df in [("queries", q), ("items", it), ("train", tr)]:
    na = df.isna().sum()
    print(name, na[na > 0].to_dict())
print("пустой search_infm_params_text: bench", (q.search_infm_params_text == "").mean().round(3))

# 2. Дубликаты в train: одна и та же пара (группа запроса, объявление) встречается несколько раз.
print("\n== дубли train (группа + item):", tr.duplicated(K + ["item_id"]).sum())
g = tr.drop_duplicates(K + ["item_id"]).groupby(K).item_id.nunique()
print("групп:", len(g), "| релевантных на группу: 1 ->", (g == 1).mean().round(3), " 2 ->", (g == 2).mean().round(3))

# 3. Сдвиг распределений train -> benchmark. Бенчмарк — это уникальные тексты, они длиннее,
#    чаще без фильтра и чаще из крупных городов/регионов. Валидацию нужно делать похожей на бенчмарк.
gg = g.reset_index()
print("\n== сдвиг train-группы vs bench")
print("пустой фильтр:", (gg.search_infm_params_text == "").mean().round(3), "vs", (q.search_infm_params_text == "").mean().round(3))
print("search_category=0:", (gg.search_category == 0).mean().round(4), "vs", (q.search_category == 0).mean().round(3))
print("слов в запросе (медиана):", gg.search_query.str.split().str.len().median(), "vs", q.search_query.str.split().str.len().median())
for L in [637640, 653240, 107620, 621540, 107621]:
    print(f"loc {L}: {(gg.search_location_id == L).mean():.3f} vs {(q.search_location_id == L).mean():.3f}")

# 4. Локации: 17% запросов бенчмарка заданы в локации, где в корпусе нет ни одного объявления
#    (регионы: 107620 ~ Московская обл., 621540 ~ вся Россия). По train видно, куда они «ведут».
vc = it.item_location_id.value_counts()
print("\n== запросов бенчмарка с пустой локацией в корпусе:", (~q.search_location_id.isin(vc.index)).mean().round(3))
print("train: доля строк, где локация объявления = локации поиска:", (tr.search_location_id == tr.item_location_id).mean().round(3))
for L in [107620, 621540]:
    print(L, "->", tr[tr.search_location_id == L].item_location_id.value_counts(normalize=True).head(3).round(3).to_dict())

# 5. Рубрикатор: «Вид услуги» фильтра почти всегда совпадает с объявлением; микрокатегория запроса
#    предсказуема (у 86% запросов одна доминирующая).
print("\n== корпус вне категории 114:", (it.item_category_id != 114).mean().round(4),
      "| микрокатегорий корпуса, не встречавшихся в train:", len(set(it.item_microcat_id) - set(tr.item_microcat_id)))
print("пересечение текстов бенчмарка с train:", q.search_query.isin(set(tr.search_query)).mean().round(3))
print("объявлений корпуса, встречавшихся в train:", it.item_id.isin(set(tr.item_id)).mean().round(3))
