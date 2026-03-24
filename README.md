# ♻️ Detector de Recicláveis com Visão Computacional

Projeto em Python que utiliza **OpenCV** e **Inteligência Artificial (TensorFlow)** para identificar objetos em tempo real e classificá-los como recicláveis ou não.

---

## 🚀 Funcionalidades

* Captura de vídeo em tempo real (webcam ou celular via IP)
* Detecção de objetos com modelo pré-treinado (MobileNetV2)
* Classificação em:

  * Orgânico
  * Plástico
  * Metal
* Indicação se o objeto é reciclável

---

## 🛠 Tecnologias utilizadas

* Python
* OpenCV
* TensorFlow
* NumPy

---

## 📦 Instalação

Instale as dependências:

```bash
pip install opencv-python tensorflow numpy
```

---

## ▶️ Como executar

```bash
python main.py
```

---

## 📱 Uso com celular (IP Webcam)

Você pode usar a câmera do celular com o aplicativo **IP Webcam**:

1. Instale o app no celular
2. Inicie o servidor
3. Copie o endereço IP
4. Substitua no código:

```python
cv2.VideoCapture("http://SEU_IP:8080/video")
```

---

## ⚠️ Limitações

* O modelo utilizado não foi treinado especificamente para reciclagem
* A classificação de material é baseada em regras simples
* Pode haver erros na identificação dos objetos

---

## 🚀 Melhorias futuras

* Treinar modelo próprio para resíduos recicláveis
* Melhorar precisão da classificação
* Criar interface gráfica
