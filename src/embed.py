"""Эмбеддинги текстов локальной моделью multilingual-e5 (sentence-transformers) с кешем на диск.

Модель скачивается с HuggingFace один раз и дальше работает офлайн; во время инференса
никаких внешних API не вызывается.

Кеш устроен «по текстам»: для каждой модели хранится хранилище (список текстов + матрица векторов).
При вызове encode() считаются только тексты, которых в хранилище ещё нет, — поэтому добавление
нескольких тысяч объявлений в корпус не требует пересчитывать остальные 190k.
"""
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
BASE_MODEL = "intfloat/multilingual-e5-small"
FINETUNED = "models/e5-small-avito"       # дообученная e5 (finetune_e5.py); путь относительно папки проекта
# По умолчанию — дообученная модель, если она лежит в models/; иначе исходная e5 с HuggingFace.
# Переопределяется переменной окружения: AVITO_MODEL=intfloat/multilingual-e5-small python validate.py
MODEL_NAME = os.environ.get("AVITO_MODEL", FINETUNED if os.path.isdir(FINETUNED) else BASE_MODEL)

_models = {}
_stores = {}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(name):
    if name not in _models:
        m = SentenceTransformer(name, device=device())
        m.max_seq_length = 128
        _models[name] = m
    return _models[name]


def _store_path(name):
    return os.path.join(CACHE_DIR, f"store_{name.strip('/').replace('/', '__')}")


def _load_store(name):
    if name not in _stores:
        p = _store_path(name)
        if os.path.exists(p + ".pkl"):
            texts = pickle.load(open(p + ".pkl", "rb"))
            emb = np.load(p + ".npy")
        else:
            texts, emb = [], np.zeros((0, 0), np.float32)
        _stores[name] = ({t: i for i, t in enumerate(texts)}, texts, emb)
    return _stores[name]


def _save_store(name):
    pos, texts, emb = _stores[name]
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = _store_path(name)
    pickle.dump(texts, open(p + ".pkl", "wb"))
    np.save(p + ".npy", emb)


def add_to_store(texts, emb, model_name=None):
    """Положить уже посчитанные векторы в хранилище (используется для миграции старого кеша)."""
    name = model_name or MODEL_NAME
    pos, all_texts, all_emb = _load_store(name)
    new = [i for i, t in enumerate(texts) if t not in pos]
    if not new:
        return
    for i in new:
        pos[texts[i]] = len(all_texts)
        all_texts.append(texts[i])
    all_emb = emb[new] if all_emb.size == 0 else np.vstack([all_emb, emb[new]])
    _stores[name] = (pos, all_texts, all_emb)
    _save_store(name)


def encode(texts, tag=None, batch_size=128, model_name=None):
    """L2-нормированные float32-векторы для texts (косинус = скалярное произведение).
    Считает только тексты, которых нет в кеше. tag оставлен для читаемости вызовов."""
    name = model_name or MODEL_NAME
    pos, _, _ = _load_store(name)
    missing = sorted({t for t in texts if t not in pos}, key=len)   # батчи одной длины кодируются быстрее
    if missing:
        emb = get_model(name).encode(missing, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=len(missing) > 5000,
                                     convert_to_numpy=True).astype(np.float32)
        add_to_store(missing, emb, name)
        pos, _, _ = _load_store(name)
    all_emb = _stores[name][2]
    return all_emb[[pos[t] for t in texts]]
