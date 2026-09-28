"""Загрузка данных и подготовка текстов для эмбеддингов.

Все пути задаются через DATA_DIR (по умолчанию ~/Downloads/dataset),
можно переопределить переменной окружения AVITO_DATA_DIR.
"""
import os
import re

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("AVITO_DATA_DIR", os.path.expanduser("~/Downloads/dataset"))

QUERY_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
              "search_infm_params_text", "search_category"]
ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_location_id",
             "item_rating", "item_rating_reviews_count", "item_price",
             "item_is_phone_hidden", "item_is_message_forbidden"]


def prepare_items(df: pd.DataFrame) -> pd.DataFrame:
    """Приведение типов признаков объявления для ранжировщика.
    EDA: NaN в рейтинге/отзывах = «отзывов нет» — оставляем NaN, градиентный бустинг
    обрабатывает пропуски сам (учит отдельную ветку). Цена -1 / 0 — «договорная», тоже NaN."""
    df = df.copy()
    price = df.item_price.astype(float)
    df["item_price"] = price.where(price > 0)
    for c in ["item_is_phone_hidden", "item_is_message_forbidden"]:
        df[c] = df[c].astype(float)
    return df


def load_items():
    return prepare_items(pd.read_parquet(os.path.join(DATA_DIR, "benchmark_items.parquet"), columns=ITEM_COLS))


def load_queries():
    return pd.read_parquet(os.path.join(DATA_DIR, "benchmark_queries.parquet"))


def load_train():
    """train.parquet весит ~0.5 ГБ, в основном из-за описаний — их не грузим.
    Тексты нужных объявлений читаются точечно при сборке валидационного корпуса (pipeline.py)."""
    cols = QUERY_COLS + ["item_id", "item_microcat_id", "item_location_id", "item_category_id"]
    df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"), columns=cols)
    # EDA: 30.6k строк — точные повторы пары (группа запроса, объявление). Для статистик
    # (гео-переходы, микрокатегории, collab) они лишь раздувают веса частых пар — убираем.
    key = ["search_query", "search_location_id", "search_infm_params_text", "search_category", "item_id"]
    return df.drop_duplicates(key).reset_index(drop=True)


# --------------------------------------------------------------------------- тексты

def _s(x) -> str:
    """Пустые значения в parquet приходят как None или NaN (float) — приводим всё к строке."""
    return x if isinstance(x, str) else ""


def norm_query(s: str) -> str:
    """Нормализация текста запроса: нижний регистр, ё->е, схлопывание пробелов."""
    s = _s(s).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip()


_VID_RE = re.compile(r"Вид услуги (.+?)(?= Тип услуги| Место оказания| Кто оказывает| Онлайн-запись"
                     r"| Где вы| Тип стоимости|$)")


def service_kind(params: str):
    """Достаёт значение «Вид услуги …» из текстовых параметров (фильтра запроса или объявления).
    Это верхний уровень рубрикатора услуг: «Ремонт и отделка», «Красота, здоровье» и т.п."""
    m = _VID_RE.search(_s(params))
    return m.group(1).strip() if m else ""


# В параметрах объявления много служебного шума (цены, график, «Тип стоимости за услугу»),
# который только забивает 128 токенов модели. Вырезаем его, оставляя смысловые слова.
_NOISE_RE = re.compile(
    r"Начальная цена|Тип стоимости за [\w.]+|Стоимость \d+|График работы[^А-ЯA-Z]*\d+|"
    r"График работы, дни недели \w+\.?|Время работы, (?:с|до) [\d:]+|Продолжительность [\d\w. ]+?(?= [А-Я]|$)|"
    r"Время для связи[^А-ЯA-Z]*\d*\.?|Время для связи, дни недели \w*\.?|Рабочие дни \w+|Дни \w\w|Признак предзаполнения прайс листа \d+|Услуга Своя услуга|Минимальная сумма заказа \d+|"
    r"Выполняю заказы от \d+|Место оказания услуг [^А-Я]*(?:[А-Я][^А-Я]*?)?(?= [А-Я])"
)


def clean_params(params: str) -> str:
    s = _NOISE_RE.sub(" ", _s(params))
    return re.sub(r"\s+", " ", s).strip()


def item_text(title, params, desc, max_chars=400) -> str:
    """Текст объявления для модели e5: заголовок + очищенные параметры + начало описания."""
    p = clean_params(params)[:250]
    d = re.sub(r"\s+", " ", _s(desc))[:200]
    return f"passage: {_s(title)}. {p}. {d}"[:max_chars]


def query_text(q, params="") -> str:
    """Текст запроса. Вид услуги из фильтра добавляем как контекст — он уточняет смысл
    коротких запросов («баки» в «Красота, здоровье» — это бакенбарды, а не топливные баки)."""
    kind = service_kind(params)
    return f"query: {_s(q)}" + (f" ({kind})" if kind else "")


def items_texts(df: pd.DataFrame) -> list:
    return [item_text(t, p, d) for t, p, d in
            zip(df.item_title_raw, df.item_infm_params_text, df.item_description_raw)]


def recall_at_k(pred: list, rel: list, k=50) -> float:
    """Recall@k из условия: среднее по запросам |top-k ∩ rel| / |rel|."""
    return float(np.mean([len(set(p[:k]) & set(r)) / len(set(r)) for p, r in zip(pred, rel)]))
