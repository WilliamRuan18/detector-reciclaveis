♻️ Detector de Recicláveis com IA

Projeto em Python que utiliza Visão Computacional e Inteligência Artificial para identificar objetos em tempo real e classificá-los como recicláveis ou não.

🚀 Funcionalidades
📷 Captura de vídeo em tempo real (webcam ou IP camera)
🤖 Detecção de objetos com MobileNetV2 (TensorFlow)
🧠 Classificação automática em:
Plástico
Metal
Papel
Orgânico
♻️ Indicação se o item é reciclável ou não
📝 Suporte a texto com acentos na tela usando Pillow
🛠️ Tecnologias Utilizadas
Python
OpenCV
TensorFlow / Keras
NumPy
Pillow (PIL)
📦 Instalação

Instale as dependências com:

pip install opencv-python tensorflow numpy pillow
▶️ Como Executar

Clone o repositório:

Clone o repositório:

git clone https://github.com/WilliamRuan18/detector-reciclaveis.git
cd detector-reciclaveis

Execute o script:

python main.py
📷 Configuração da Câmera

O código está configurado para usar uma câmera via IP:

cap = cv2.VideoCapture("http://10.0.0.185:8080/video")
🔧 Outras opções:

Webcam do PC:

cap = cv2.VideoCapture(0)

Outro IP de câmera:

cap = cv2.VideoCapture("http://SEU_IP:PORTA/video")
🧠 Como Funciona
📌 1. Captura de imagem

O vídeo é capturado frame a frame pela câmera.

📌 2. Detecção com IA

A cada 30 frames:

A imagem é redimensionada para 224x224
É processada pelo modelo MobileNetV2
O objeto principal é identificado
📌 3. Classificação de material

Com base no objeto detectado:

Objeto	Material	Reciclável
Garrafa	Plástico	Sim
Lata	Metal	Sim
Papel	Papel	Sim
Frutas	Orgânico	Não
📌 4. Exibição na tela

As seguintes informações aparecem no vídeo:

Objeto detectado
Tipo de material
Se é reciclável
✍️ Suporte a Acentos

O OpenCV não suporta bem caracteres acentuados.
Por isso, foi utilizada a biblioteca Pillow para renderizar textos corretamente:

escrever_texto_acentuado(...)
📁 Estrutura do Projeto
📁 projeto/
│-- main.py
│-- README.md
⚠️ Limitações
A precisão depende do modelo MobileNetV2
Nem todos os objetos são reconhecidos corretamente
A classificação de recicláveis é baseada em regras simples
💡 Melhorias Futuras
🔍 Treinar um modelo próprio para recicláveis
📊 Adicionar porcentagem de confiança
🎯 Melhorar a classificação de materiais
🧠 Utilizar modelos mais avançados como YOLO

