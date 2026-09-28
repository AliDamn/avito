"""Шаг 2 дообучения e5: обучение multilingual-e5-small на pairs.parquet (из export_pairs.py).

    python finetune_e5.py             # -> models/e5-small-avito

Именно этим кодом получена модель из решения. Обучалась в Google Colab на T4 (~10 мин):
    1) Runtime -> Change runtime type -> T4 GPU;
    2) загрузить pairs.parquet и этот файл в /content (панель файлов слева или Google Drive);
    3) !pip install -q sentence-transformers  &&  !python finetune_e5.py
    4) скачать models/e5-small-avito и положить в папку проекта models/.
На Mac M1 (MPS, 8 ГБ) обучение тоже запускается, но идёт на порядок медленнее (~10+ ч).

Лосс — MultipleNegativesRankingLoss (InfoNCE): для каждого запроса в батче его объявление — позитив,
объявления остальных 127 запросов батча — негативы. Модель учится ставить выбранное пользователем
объявление выше других похожих — ровно то, в чём слаб косинус «из коробки».
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="pairs.parquet")
ap.add_argument("--out", default="models/e5-small-avito")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--scale", type=float, default=20.0, help="обратная температура в InfoNCE")
args = ap.parse_args()

torch.manual_seed(0)
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
amp = dev == "cuda"                                   # fp16 только на CUDA

df = pd.read_parquet(args.pairs)
Q, D, I = df.q.tolist(), df.d.tolist(), df.item_id.to_numpy()
print(f"{len(Q)} пар, устройство {dev}")

model = SentenceTransformer("intfloat/multilingual-e5-small", device=dev)
model.max_seq_length = args.max_len
tokenize = getattr(model, "preprocess", None) or model.tokenize   # API отличается между версиями ST


def embed(texts):
    # preprocess() кроме тензоров возвращает служебные поля (modality: str) — на устройство переносим только тензоры
    f = tokenize(texts)
    f = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in f.items()}
    return F.normalize(model(f)["sentence_embedding"].float(), dim=-1)


def batches():
    """Без повторов объявления и текста запроса в батче: иначе «негатив» оказался бы позитивом."""
    b, seen_i, seen_q = [], set(), set()
    for j in rng.permutation(len(Q)):
        if I[j] in seen_i or Q[j] in seen_q:
            continue
        b.append(j)
        seen_i.add(I[j])
        seen_q.add(Q[j])
        if len(b) == args.batch:
            yield b
            b, seen_i, seen_q = [], set(), set()


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = args.epochs * (len(Q) // args.batch)
warm = max(1, total // 10)                            # 10% линейного разогрева, дальше косинусное затухание
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
scaler = torch.amp.GradScaler("cuda", enabled=amp)

model.train()
step, t0, avg = 0, time.time(), None
for ep in range(args.epochs):
    for b in batches():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            q, d = embed([Q[j] for j in b]), embed([D[j] for j in b])
            loss = F.cross_entropy(args.scale * q @ d.T, torch.arange(len(b), device=dev))  # позитивы на диагонали
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        avg = loss.item() if avg is None else 0.98 * avg + 0.02 * loss.item()
        if step % 100 == 0 or step == 10:
            el = time.time() - t0
            print(f"ep {ep} step {step}/{total} loss {avg:.3f} | {el / 60:.1f} мин, "
                  f"осталось ~{el / step * (total - step) / 60:.0f} мин", flush=True)

model.save(args.out)
print(f"модель сохранена в {args.out}")
