import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import json
import os
from collections import Counter

# --- CONFIGURAÇÕES ---
MODEL_PATH = "melhor_modelo.h5"
CLASSES_PATH = "classes.json"
IMG_SIZE = (224, 224)

# --- CARREGAR MODELO ---
if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
    print("⏳ Carregando Inteligência Artificial...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)

    labels_map = {v: k for k, v in class_indices.items()}
else:
    print("❌ ERRO: Modelo ou classes não encontrados!")
    exit()

# --- FONTE ---
try:
    font = ImageFont.truetype("arial.ttf", 28)
except:
    font = ImageFont.load_default()

# --- ESTABILIZADOR ---
historico_predicoes = []

# --- FUNÇÃO TEXTO ---
def escrever_texto(img, texto, posicao, cor_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
    draw.text(posicao, texto, font=font, fill=cor_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- CLASSIFICAÇÃO ATUALIZADA ---
def classificar_reciclagem(classe_id):
    traducoes = {
        "metal": ("Metal", "Reciclável"),
        "papel": ("Papel", "Reciclável"),
        "plastico": ("Plástico", "Reciclável"),
        "organico": ("Orgânico", "Não Reciclável")
    }
    return traducoes.get(classe_id, ("Desconhecido", "---"))

# --- CONEXÃO CÂMERA ---
cap = cv2.VideoCapture("http://10.0.0.185:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("🚀 Sistema Pronto! Aponte para o objeto.")

while True:
    # Limpa buffer (reduz delay)
    for _ in range(3):
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        break

    # --- PRÉ-PROCESSAMENTO ---
    h, w, _ = frame.shape
    min_dim = min(h, w)

    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2

    crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    img_prep = cv2.resize(crop, IMG_SIZE)
    img_prep = np.expand_dims(img_prep, axis=0) / 255.0

    # --- PREDIÇÃO ---
    preds = model.predict(img_prep, verbose=0)
    idx_atual = np.argmax(preds)

    # --- ESTABILIZAÇÃO ---
    historico_predicoes.append(idx_atual)
    if len(historico_predicoes) > 10:
        historico_predicoes.pop(0)

    idx_estavel = Counter(historico_predicoes).most_common(1)[0][0]

    # --- RESULTADO FINAL ---
    objeto_nome = labels_map[idx_estavel]
    confianca = preds[0][idx_estavel] * 100

    material, reciclavel = classificar_reciclagem(objeto_nome)

    # --- INTERFACE ---
    frame_disp = cv2.resize(frame, (640, 480))

    cor_tema = (0, 255, 0) if reciclavel == "Reciclável" else (0, 0, 255)

    # Moldura central
    cv2.rectangle(frame_disp, (100, 80), (540, 400), cor_tema, 2)

    # Textos
    frame_disp = escrever_texto(
        frame_disp,
        f"Objeto: {objeto_nome.capitalize()} ({confianca:.1f}%)",
        (20, 20),
        (255, 255, 255)
    )

    frame_disp = escrever_texto(
        frame_disp,
        f"Material: {material}",
        (20, 60),
        cor_tema
    )

    frame_disp = escrever_texto(
        frame_disp,
        f"Status: {reciclavel}",
        (20, 100),
        cor_tema
    )

    cv2.imshow("♻️ Detector Inteligente de Reciclagem", frame_disp)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()