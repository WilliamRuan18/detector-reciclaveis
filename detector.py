import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont # Importações do Pillow

# Modelo pronto
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# --- CONFIGURAÇÃO DA FONTE ---
# No Windows, a Arial costuma estar neste caminho. No Linux/Mac, use o nome do arquivo .ttf
try:
    font_path = "arial.ttf" 
    font = ImageFont.truetype(font_path, 25)
except:
    font = ImageFont.load_default()

def escrever_texto_acentuado(img, texto, posicao, cor_bgr):
    # Converte de OpenCV (BGR) para Pillow (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Inverte a cor de BGR para RGB para o Pillow
    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
    
    # Desenha o texto com acento
    draw.text(posicao, texto, font=font, fill=cor_rgb)
    
    # Converte de volta para OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Função de classificação de material (AGORA COM ACENTOS!)
def classificar_material(objeto):
    objeto = objeto.lower()

    if objeto in ["banana", "orange", "lemon", "strawberry", "granny_smith", "custard_apple", "pomegranate", "pineapple", "fig", "croquet_ball"]:
        return "Organico", "Não Reciclável"
    
    elif objeto in ["water_bottle", "pill_bottle", "wine_bottle", "plastic_bag"]:
        return "Plástico", "Reciclável"

    elif objeto in ["can", "tin", "beer_can", "soda_can", "screwdriver"]:
        return "Metal", "Reciclável"
    
    elif objeto in ["paper_towel", "toilet_paper", "envelope", "toilet_tissue"]:
        return "Papel", "Reciclável"
    
    else:
        return "Desconhecido", "Verificar"

# Função que usa IA
def detectar_objeto(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    preds = model.predict(img, verbose=0)
    resultado = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
    return resultado[0][0][1] 

# Conexão com câmera
cap = cv2.VideoCapture("http://10.0.0.185:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
objeto = "aguardando..." 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 30 == 0: 
        objeto = detectar_objeto(frame)

    material, reciclavel = classificar_material(objeto) 

    frame_display = cv2.resize(frame, (640, 480))

    # USANDO PILLOW PARA OS TEXTOS (Substituindo cv2.putText)
    frame_display = escrever_texto_acentuado(frame_display, f"Objeto: {objeto}", (20, 20), (0, 255, 0))
    frame_display = escrever_texto_acentuado(frame_display, f"Material: {material}", (20, 60), (255, 0, 0))
    frame_display = escrever_texto_acentuado(frame_display, f"Reciclável: {reciclavel}", (20, 100), (0, 0, 255))

    cv2.imshow("Detector de Reciclavel", frame_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()