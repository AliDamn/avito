"""Шаг 1 дообучения e5: выгрузка пар «запрос -> выбранное объявление» для обучения.

    python export_pairs.py            # -> pairs.parquet (~36 МБ, 200k пар)

Пары берутся только из stats-части разбиения (make_splits): rank- и val-группы в обучение
не попадают, поэтому валидация дообученной модели честная.
Тексты строятся теми же функциями (query_text / item_text), что и при инференсе.
Дальше pairs.parquet обучается в finetune_e5.py (на GPU, например в Google Colab).
"""
import os
import sys

sys.path.insert(0, "src")

import pandas as pd
import pyarrow.parquet as pq

from data import DATA_DIR, ITEM_COLS, items_texts, load_items, load_queries, load_train, prepare_items, query_text
from pipeline import make_splits

N_PAIRS = 200_000

train, corpus, bench = load_train(), load_items(), load_queries()
stats, *_ = make_splits(train, corpus, bench)
pairs = stats.drop_duplicates(["search_query", "item_id"])
pairs = pairs.sample(n=min(N_PAIRS, len(pairs)), random_state=0)

# тексты объявлений читаем точечно из train (описания всего train в память не грузим)
items = prepare_items(pq.read_table(os.path.join(DATA_DIR, "train.parquet"), columns=ITEM_COLS,
                                    filters=[("item_id", "in", pairs.item_id.unique().tolist())])
                      .to_pandas().drop_duplicates("item_id"))
txt = dict(zip(items.item_id, items_texts(items)))
out = pd.DataFrame({
    "q": [query_text(q, p) for q, p in zip(pairs.search_query, pairs.search_infm_params_text)],
    "d": [txt[i] for i in pairs.item_id],
    "item_id": pairs.item_id.values,
})
out.to_parquet("pairs.parquet", compression="zstd")
print(len(out), "пар ->", round(os.path.getsize("pairs.parquet") / 2**20, 1), "МБ")
