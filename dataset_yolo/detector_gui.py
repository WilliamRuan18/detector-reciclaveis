"""
DETECTOR DE RECICLÁVEIS — GUI COMPLETA
=======================================
Instale as dependências:
  pip install ultralytics opencv-python pyqt5 pillow numpy

Rode:
  python detector_gui.py
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from collections import Counter
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
    QProgressBar, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QPalette

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ── CONFIGURAÇÕES ──────────────────────────────────────────────────────────────
MODEL_PATH      = "runs/detect/train14/weights/best.pt"
CONF_THRESHOLD  = 0.4
MAX_IDS         = 100
CAMERA_INDEX    = 0

CORES_MATERIAL = {
    "Plástico":   "#E74C3C",
    "Metal":      "#95A5A6",
    "Vidro":      "#3498DB",
    "Papel":      "#F39C12",
    "Lixo Comum": "#7F8C8D",
    "Desconhecido": "#BDC3C7",
}

MAPA_CLASSES = {
    "Pet_Bottle":   "Plástico",
    "Plastic_Bag":  "Plástico",
    "can":          "Metal",
    "Glass":        "Vidro",
    "Paper_Bag":    "Papel",
    "Garbage_Bag":  "Lixo Comum",
}

os.makedirs("capturas/reciclavel",    exist_ok=True)
os.makedirs("capturas/nao_reciclavel", exist_ok=True)

# ── THREAD DE DETECÇÃO ────────────────────────────────────────────────────────
class DetectorThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    detection_new = pyqtSignal(str, float, int)   # material, confiança, track_id
    stats_updated = pyqtSignal(dict)
    fps_updated   = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running     = False
        self.model       = None
        self.estatisticas = {k: 0 for k in ["Plástico", "Metal", "Vidro", "Papel", "Lixo Comum"]}
        self.ids_contados = set()
        self.ids_salvos   = set()
        self.historico    = {}

    def load_model(self):
        try:
            self.model = YOLO(MODEL_PATH)
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False

    def classificar(self, nome):
        return MAPA_CLASSES.get(nome, "Desconhecido")

    def salvar_imagem(self, frame, x1, y1, x2, y2, tipo, track_id):
        obj = frame[y1:y2, x1:x2]
        if obj.size == 0:
            return
        pasta = "capturas/reciclavel" if tipo == "Reciclável" else "capturas/nao_reciclavel"
        cv2.imwrite(f"{pasta}/obj_{track_id}.jpg", obj)

    def salvar_estatisticas(self):
        with open("estatisticas.json", "w", encoding="utf-8") as f:
            json.dump(self.estatisticas, f, indent=4, ensure_ascii=False)

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        t_prev = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (960, 600))

            # FPS
            now  = time.time()
            fps  = 1.0 / (now - t_prev + 1e-9)
            t_prev = now
            self.fps_updated.emit(round(fps, 1))

            # FPS sobreposto no vídeo
            fps_label = f"FPS: {fps:.0f}"
            (fw, fh), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (8, 8), (fw + 18, fh + 18), (0, 0, 0), -1)
            cv2.putText(frame, fps_label, (13, fh + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, cv2.LINE_AA)

            if self.model is None:
                self.frame_ready.emit(frame)
                continue

            try:
                results = self.model.track(frame, persist=True, conf=0.5, iou=0.5, verbose=False)
            except:
                self.frame_ready.emit(frame)
                continue

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    nome     = self.model.names[cls]
                    material = self.classificar(nome)

                    # Estabilizador
                    if track_id not in self.historico:
                        self.historico[track_id] = []
                    self.historico[track_id].append(material)
                    if len(self.historico[track_id]) > 5:
                        self.historico[track_id].pop(0)
                    material = Counter(self.historico[track_id]).most_common(1)[0][0]

                    tipo = "Reciclável" if material != "Lixo Comum" else "Não Reciclável"
                    hex_cor = CORES_MATERIAL.get(material, "#BDC3C7")
                    r = int(hex_cor[1:3], 16)
                    g = int(hex_cor[3:5], 16)
                    b = int(hex_cor[5:7], 16)
                    cor_bgr = (b, g, r)

                    # Desenha caixa (espessura varia com confiança)
                    espessura = 3 if conf > 0.7 else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_bgr, espessura)

                    # Label com material + confiança % + tipo
                    icone = "R" if tipo == "Reciclavel" else "X"
                    label = f"{material}  {conf*100:.0f}%"
                    tipo_label = "Reciclavel" if tipo == "Reciclavel" else "Nao Reciclavel"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    (tw2, th2), _ = cv2.getTextSize(tipo_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    box_w = max(tw, tw2) + 14

                    # Fundo do label (dupla linha)
                    cv2.rectangle(frame, (x1, y1 - 44), (x1 + box_w, y1), (20, 20, 20), -1)
                    cv2.rectangle(frame, (x1, y1 - 44), (x1 + box_w, y1), cor_bgr, 1)

                    # Linha 1: material + confiança
                    cv2.putText(frame, label, (x1 + 6, y1 - 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    # Linha 2: tipo (reciclável / não reciclável) — cor diferente
                    cor_tipo = (100, 220, 100) if tipo == "Reciclavel" else (80, 80, 220)
                    cv2.putText(frame, tipo_label, (x1 + 6, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, cor_tipo, 1, cv2.LINE_AA)

                    # Barra de confiança com cor dinâmica
                    bw = int(conf * (x2 - x1))
                    cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 12), (40, 40, 40), -1)
                    cor_barra = (0, 200, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 0, 200)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bw, y2 + 12), cor_barra, -1)
                    # Percentual ao lado da barra
                    cv2.putText(frame, f"{conf*100:.0f}%", (x2 + 4, y2 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor_barra, 1, cv2.LINE_AA)

                    # Contar uma vez
                    if track_id not in self.ids_contados:
                        self.ids_contados.add(track_id)
                        if material in self.estatisticas:
                            self.estatisticas[material] += 1
                            self.salvar_estatisticas()
                            self.stats_updated.emit(dict(self.estatisticas))
                        self.detection_new.emit(material, float(conf)*100, int(track_id))

                    # Salvar imagem
                    if track_id not in self.ids_salvos and conf < 0.7:
                        self.ids_salvos.add(track_id)
                        self.salvar_imagem(frame, x1, y1, x2, y2, tipo, track_id)

                # Limpar memória
                if len(self.ids_contados) > MAX_IDS:
                    self.ids_contados.clear()
                    self.ids_salvos.clear()
                    self.historico.clear()

            self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ── WIDGET DE BARRA DE MATERIAL ───────────────────────────────────────────────
class MaterialBar(QWidget):
    def __init__(self, nome, cor, parent=None):
        super().__init__(parent)
        self.nome  = nome
        self.cor   = cor
        self.valor = 0
        self.total = 1
        self._setup()

    def _setup(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        self.lbl_nome = QLabel(self.nome)
        self.lbl_nome.setFixedWidth(80)
        self.lbl_nome.setStyleSheet("color: #ECEFF1; font-size: 12px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background: #37474F;
                border-radius: 5px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {self.cor};
                border-radius: 5px;
            }}
        """)

        self.lbl_qtd = QLabel("0")
        self.lbl_qtd.setFixedWidth(30)
        self.lbl_qtd.setAlignment(Qt.AlignRight)
        self.lbl_qtd.setStyleSheet("color: #ECEFF1; font-size: 12px; font-weight: bold;")

        self.lbl_pct = QLabel("0%")
        self.lbl_pct.setFixedWidth(38)
        self.lbl_pct.setAlignment(Qt.AlignRight)
        self.lbl_pct.setStyleSheet("color: #90A4AE; font-size: 11px;")

        lay.addWidget(self.lbl_nome)
        lay.addWidget(self.bar)
        lay.addWidget(self.lbl_qtd)
        lay.addWidget(self.lbl_pct)

    def update_values(self, valor, total):
        self.valor = valor
        self.total = max(total, 1)
        pct = int(valor / self.total * 100)
        self.bar.setValue(pct)
        self.lbl_qtd.setText(str(valor))
        self.lbl_pct.setText(f"{pct}%")


# ── PAINEL FLUTUANTE ──────────────────────────────────────────────────────────
class FloatingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._setup()

    def _setup(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # ── Card principal ──────────────────────────────────────────────────
        self.card = QFrame()
        self.card.setStyleSheet("""
            QFrame {
                background: rgba(18, 26, 32, 210);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.08);
            }
        """)
        card_lay = QVBoxLayout(self.card)
        card_lay.setContentsMargins(16, 14, 16, 14)
        card_lay.setSpacing(12)

        # Título
        title = QLabel("♻  Detector de Recicláveis")
        title.setStyleSheet("color: #4FC3F7; font-size: 14px; font-weight: bold; background: transparent; border: none;")
        card_lay.addWidget(title)

        # Separador
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.1); border: none; max-height: 1px;")
        card_lay.addWidget(sep)

        # Métricas do topo
        metrics_lay = QHBoxLayout()
        metrics_lay.setSpacing(8)

        self.lbl_total  = self._metric_card("TOTAL", "0", "#4FC3F7")
        self.lbl_rec    = self._metric_card("RECICLÁVEL", "0", "#66BB6A")
        self.lbl_nrec   = self._metric_card("NÃO RECIC.", "0", "#EF5350")
        self.lbl_fps    = self._metric_card("FPS", "0", "#FFA726")

        for w in [self.lbl_total, self.lbl_rec, self.lbl_nrec, self.lbl_fps]:
            metrics_lay.addWidget(w)
        card_lay.addLayout(metrics_lay)

        # Barras de materiais
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.1); border: none; max-height: 1px;")
        card_lay.addWidget(sep2)

        lbl_mat = QLabel("Materiais detectados")
        lbl_mat.setStyleSheet("color: #90A4AE; font-size: 11px; background: transparent; border: none;")
        card_lay.addWidget(lbl_mat)

        self.barras = {}
        for nome, cor in CORES_MATERIAL.items():
            if nome == "Desconhecido":
                continue
            b = MaterialBar(nome, cor)
            b.setStyleSheet("background: transparent;")
            self.barras[nome] = b
            card_lay.addWidget(b)

        # Separador
        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.1); border: none; max-height: 1px;")
        card_lay.addWidget(sep3)

        # Último detectado
        lbl_ult = QLabel("Último detectado")
        lbl_ult.setStyleSheet("color: #90A4AE; font-size: 11px; background: transparent; border: none;")
        card_lay.addWidget(lbl_ult)

        self.lbl_ultimo = QLabel("—")
        self.lbl_ultimo.setStyleSheet("color: #ECEFF1; font-size: 13px; font-weight: bold; background: transparent; border: none;")
        card_lay.addWidget(self.lbl_ultimo)

        self.lbl_conf = QLabel("")
        self.lbl_conf.setStyleSheet("color: #90A4AE; font-size: 11px; background: transparent; border: none;")
        card_lay.addWidget(self.lbl_conf)

        # Hora
        self.lbl_hora = QLabel("")
        self.lbl_hora.setStyleSheet("color: #546E7A; font-size: 10px; background: transparent; border: none;")
        card_lay.addWidget(self.lbl_hora)

        # Botões
        sep4 = QFrame(); sep4.setFrameShape(QFrame.HLine)
        sep4.setStyleSheet("color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.1); border: none; max-height: 1px;")
        card_lay.addWidget(sep4)

        btn_lay = QHBoxLayout()
        btn_lay.setSpacing(8)

        self.btn_reset = QPushButton("Resetar")
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: rgba(239,83,80,0.2);
                color: #EF5350;
                border: 1px solid rgba(239,83,80,0.4);
                border-radius: 8px;
                padding: 6px 0;
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(239,83,80,0.35); }
        """)

        self.btn_salvar = QPushButton("Salvar JSON")
        self.btn_salvar.setStyleSheet("""
            QPushButton {
                background: rgba(79,195,247,0.2);
                color: #4FC3F7;
                border: 1px solid rgba(79,195,247,0.4);
                border-radius: 8px;
                padding: 6px 0;
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(79,195,247,0.35); }
        """)

        btn_lay.addWidget(self.btn_reset)
        btn_lay.addWidget(self.btn_salvar)
        card_lay.addLayout(btn_lay)

        lay.addWidget(self.card)
        lay.addStretch()

    def _metric_card(self, label, valor, cor):
        w = QFrame()
        w.setStyleSheet(f"""
            QFrame {{
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.07);
            }}
        """)
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(2)

        lv = QLabel(valor)
        lv.setAlignment(Qt.AlignCenter)
        lv.setStyleSheet(f"color: {cor}; font-size: 18px; font-weight: bold; background: transparent; border: none;")
        lv.setObjectName("val")

        ll = QLabel(label)
        ll.setAlignment(Qt.AlignCenter)
        ll.setStyleSheet("color: #78909C; font-size: 9px; background: transparent; border: none;")

        vl.addWidget(lv)
        vl.addWidget(ll)
        return w

    def _get_val(self, card):
        return card.findChild(QLabel, "val")

    def update_stats(self, stats):
        total = sum(stats.values())
        rec   = sum(v for k, v in stats.items() if k != "Lixo Comum")
        nrec  = stats.get("Lixo Comum", 0)

        self._get_val(self.lbl_total).setText(str(total))
        self._get_val(self.lbl_rec).setText(str(rec))
        self._get_val(self.lbl_nrec).setText(str(nrec))

        for nome, barra in self.barras.items():
            barra.update_values(stats.get(nome, 0), total)

    def update_fps(self, fps):
        self._get_val(self.lbl_fps).setText(str(int(fps)))

    def update_last(self, material, confianca):
        cor = CORES_MATERIAL.get(material, "#BDC3C7")
        self.lbl_ultimo.setText(material)
        self.lbl_ultimo.setStyleSheet(f"color: {cor}; font-size: 13px; font-weight: bold; background: transparent; border: none;")
        self.lbl_conf.setText(f"Confiança: {confianca:.1f}%")
        self.lbl_hora.setText(datetime.now().strftime("Detectado às %H:%M:%S"))


# ── JANELA PRINCIPAL ──────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("♻ Detector Inteligente de Recicláveis")
        self.showFullScreen()
        self.setStyleSheet("background: #0D1B2A;")
        self._setup_ui()
        self._start_detector()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background: #0D1B2A;")

        # Fundo = label de vídeo
        self.video_label = QLabel(central)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #000;")
        self.video_label.setScaledContents(True)

        # Painel flutuante (sobreposto)
        self.panel = FloatingPanel(central)
        self.panel.btn_reset.clicked.connect(self._reset_stats)
        self.panel.btn_salvar.clicked.connect(self._salvar_manual)

        # Botão ESC / sair
        self.btn_sair = QPushButton("✕  Sair", central)
        self.btn_sair.setFixedSize(90, 34)
        self.btn_sair.setStyleSheet("""
            QPushButton {
                background: rgba(18,26,32,200);
                color: #EF5350;
                border: 1px solid rgba(239,83,80,0.4);
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(239,83,80,0.25); }
        """)
        self.btn_sair.clicked.connect(self.close)

        # Status bar
        self.status_lbl = QLabel("Carregando modelo...", central)
        self.status_lbl.setStyleSheet("color: #FFA726; font-size: 12px; background: transparent;")

        self._reposition()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'video_label'):
            self._reposition()

    def _reposition(self):
        w, h = self.width(), self.height()
        self.video_label.setGeometry(0, 0, w, h)
        self.panel.setGeometry(w - 296, 20, 280, h - 40)
        self.btn_sair.move(w - 296 - 100, 20)
        self.status_lbl.move(16, h - 30)

    def _start_detector(self):
        self.detector = DetectorThread()
        ok = self.detector.load_model()
        if ok:
            self.status_lbl.setText("Modelo carregado  |  Câmera ativa")
            self.status_lbl.setStyleSheet("color: #66BB6A; font-size: 12px; background: transparent;")
        else:
            self.status_lbl.setText("Modelo não encontrado — verifique o caminho")
            self.status_lbl.setStyleSheet("color: #EF5350; font-size: 12px; background: transparent;")

        self.detector.frame_ready.connect(self._update_frame)
        self.detector.stats_updated.connect(self.panel.update_stats)
        self.detector.fps_updated.connect(self.panel.update_fps)
        self.detector.detection_new.connect(
            lambda mat, conf, tid: self.panel.update_last(mat, conf)
        )
        self.detector.start()

    def _update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def _reset_stats(self):
        self.detector.estatisticas = {k: 0 for k in self.detector.estatisticas}
        self.detector.ids_contados.clear()
        self.detector.ids_salvos.clear()
        self.detector.historico.clear()
        self.panel.update_stats(self.detector.estatisticas)
        self.detector.salvar_estatisticas()

    def _salvar_manual(self):
        self.detector.salvar_estatisticas()
        self.status_lbl.setText(f"Salvo em estatisticas.json  —  {datetime.now().strftime('%H:%M:%S')}")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, e):
        self.detector.stop()
        super().closeEvent(e)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    sys.exit(app.exec_())