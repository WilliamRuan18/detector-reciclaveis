import os
import shutil
import random

ORIGEM = "capturas"
DESTINO = "dataset_auto"

classes = ["reciclavel", "nao_reciclavel"]

for classe in classes:
    os.makedirs(f"{DESTINO}/train/{classe}", exist_ok=True)
    os.makedirs(f"{DESTINO}/val/{classe}", exist_ok=True)

    imagens = os.listdir(f"{ORIGEM}/{classe}")
    random.shuffle(imagens)

    split = int(len(imagens) * 0.8)

    train = imagens[:split]
    val = imagens[split:]

    for img in train:
        shutil.copy(f"{ORIGEM}/{classe}/{img}", f"{DESTINO}/train/{classe}/{img}")

    for img in val:
        shutil.copy(f"{ORIGEM}/{classe}/{img}", f"{DESTINO}/val/{classe}/{img}")

print("✅ Dataset automático criado!")