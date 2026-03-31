import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import json

# ── CONFIG ─────────────────────────────
MODEL_PATH = "runs/detect/train14/weights/best.pt"
CONF_THRESHOLD = 0.4
MAX_IDS = 100

# ── CRIAR PASTAS ───────────────────────
os.makedirs("capturas/reciclavel", exist_ok=True)
os.makedirs("capturas/nao_reciclavel", exist_ok=True)

# ── MODELO ─────────────────────────────
model = YOLO(MODEL_PATH)

# ── FONTE ──────────────────────────────
try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
except:
    font = ImageFont.load_default()

# ── ESTATÍSTICAS ───────────────────────
estatisticas = {
    "Plástico": 0,
    "Metal": 0,
    "Vidro": 0,
    "Papel": 0,
    "Lixo Comum": 0
}

ids_contados = set()
ids_salvos = set()
historico = {}

def salvar_estatisticas():
    with open("estatisticas.json", "w", encoding="utf-8") as f:
        json.dump(estatisticas, f, indent=4, ensure_ascii=False)

# ── SALVAR IMAGEM ──────────────────────
def salvar_imagem(frame, x1, y1, x2, y2, tipo, track_id):
    objeto = frame[y1:y2, x1:x2]

    if objeto.size == 0:
        return

    pasta = "capturas/reciclavel" if tipo == "Reciclável" else "capturas/nao_reciclavel"
    nome = f"{pasta}/obj_{track_id}.jpg"

    cv2.imwrite(nome, objeto)

# ── TRADUÇÃO ───────────────────────────
def classificar(nome):
    mapa = {
        "Pet_Bottle": "Plástico",
        "Plastic_Bag": "Plástico",
        "can": "Metal",
        "Glass": "Vidro",
        "Paper_Bag": "Papel",
        "Garbage_Bag": "Lixo Comum"
    }
    return mapa.get(nome, "Desconhecido")

# ── TEXTO ──────────────────────────────
def draw_text(img, text, x, y, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    draw.rectangle([x, y, x+240, y+28], fill=color)
    draw.text((x+6, y+4), text, font=font, fill=(255,255,255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── CAMERA ─────────────────────────────
cap = cv2.VideoCapture(0)

print("🚀 SISTEMA COMPLETO COM IA TREINADA rodando")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))

    # YOLO OTIMIZADO
    try:
        results = model.track(frame, persist=True, conf=0.5, iou=0.5, verbose=False)
    except:
        continue

    if results and results[0].boxes is not None and results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, ids, classes, confs):

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)

            nome = model.names[cls]
            material = classificar(nome)

            # 🔥 SUAVIZAÇÃO (evita piscar)
            if track_id not in historico:
                historico[track_id] = []

            historico[track_id].append(material)

            if len(historico[track_id]) > 5:
                historico[track_id].pop(0)

            material = max(set(historico[track_id]), key=historico[track_id].count)

            tipo = "Reciclável" if material != "Lixo Comum" else "Não Reciclável"

            # 🧠 CONTAR UMA VEZ
            if track_id not in ids_contados:
                ids_contados.add(track_id)

                if material in estatisticas:
                    estatisticas[material] += 1
                    salvar_estatisticas()

            # 💾 SALVAR IMAGEM INTELIGENTE
            if track_id not in ids_salvos and conf < 0.7:
                ids_salvos.add(track_id)
                salvar_imagem(frame, x1, y1, x2, y2, tipo, track_id)

            # 🎨 VISUAL
            cor = (0,255,0) if tipo == "Reciclável" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), cor, 2)
            frame = draw_text(frame, f"{material} | ID:{track_id}", x1, y1-30, cor)

            largura = int(conf * (x2 - x1))
            cv2.rectangle(frame, (x1, y2+5), (x1+largura, y2+12), cor, -1)

    # 🔥 LIMPAR IDS (evita travar com o tempo)
    if len(ids_contados) > MAX_IDS:
        ids_contados.clear()
        ids_salvos.clear()
        historico.clear()

    # 📊 DASHBOARD
    total = sum(estatisticas.values())
    frame = draw_text(frame, f"TOTAL: {total}", 10, 10, (50,50,50))

    y = 50
    for mat, qtd in estatisticas.items():
        perc = (qtd / total * 100) if total > 0 else 0
        frame = draw_text(frame, f"{mat}: {qtd} ({perc:.1f}%)", 10, y, (80,80,80))
        y += 30

    cv2.imshow("♻️ Sistema Inteligente MAX", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()