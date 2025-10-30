# ============================================
# DETECTOR DE FORMAS EN FOTOGRAMAS (STREAMLIT CLOUD)
# ============================================
# Autor: Yeltsin Solano Díaz
# Captura frames durante 5 segundos, detecta formas y genera imagen final
# Compatible con macOS M4, Windows y Streamlit Cloud
# ============================================

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
from PIL import Image
import io
from datetime import datetime

# --------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------
st.set_page_config(page_title="Detector de Formas (Frames)", layout="wide")
st.title("📸 Detección de Formas — Captura de Fotogramas (5 segundos)")
st.caption("Versión rápida y ligera para Streamlit Cloud • Compatible con Mac M4 y Windows.")

# --------------------------
# CONFIGURACIÓN STUN/TURN
# --------------------------
rtc_config = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

# --------------------------
# FUNCIÓN DE DETECCIÓN
# --------------------------
def detectar_formas(frame_bgr):
    conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    edges = cv2.Canny(gris, 60, 150)
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 800:
            continue
        peri = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        circularidad = 4 * np.pi * area / (peri * peri + 1e-5)
        forma, color = None, None

        if len(approx) == 3:
            forma, color = "Triangulo", (0, 255, 255)
        elif len(approx) == 4:
            if 0.9 <= aspect_ratio <= 1.1:
                forma, color = "Cuadrado", (255, 0, 0)
            else:
                forma, color = "Rectangulo", (0, 255, 0)
        elif circularidad > 0.82:
            forma, color = "Circulo", (0, 0, 255)

        if forma:
            conteo[forma] += 1
            cv2.drawContours(frame_bgr, [approx], -1, color, 2)
            cv2.putText(frame_bgr, forma, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame_bgr, conteo


# --------------------------
# PROCESADOR DE VIDEO STREAMLIT
# --------------------------
class FrameCollector(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        self.running = False
        self.start_time = None
        self.last_frame = None
        self.progress = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Mientras corre la captura (5s)
        if self.running:
            elapsed = time.time() - self.start_time
            if elapsed < 5:
                self.frames.append(img.copy())
                self.progress = (elapsed / 5) * 100
                cv2.putText(img, f"Capturando... {5 - int(elapsed)}s", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            else:
                self.running = False
        self.last_frame = img
        return img


# --------------------------
# INTERFAZ STREAMLIT
# --------------------------
st.info("Activa tu cámara (webcam o Iriun Camera) y luego presiona el botón para capturar fotogramas durante 5 segundos.")
ctx = webrtc_streamer(
    key="frame-detector",
    video_processor_factory=FrameCollector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration=rtc_config,
)

if ctx.video_processor:
    processor = ctx.video_processor

    if st.button("▶️ Capturar fotogramas (5 s)"):
        processor.frames = []
        processor.start_time = time.time()
        processor.running = True
        st.success("🎬 Capturando durante 5 segundos...")

    # Mostrar progreso en tiempo real
    if processor.running:
        st.progress(int(processor.progress))

    # Al terminar la captura
    elif not processor.running and len(processor.frames) > 0:
        st.success("✅ Captura completada. Analizando frames...")

        conteo_total = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
        ultimo_frame = None

        for frame in processor.frames:
            resultado, conteo = detectar_formas(frame)
            for k in conteo:
                conteo_total[k] += conteo[k]
            ultimo_frame = resultado

        # Mostrar conteo
        st.subheader("📊 Conteo total de formas detectadas")
        st.json(conteo_total)

        # Imagen final
        if ultimo_frame is not None:
            img_rgb = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="🖼️ Último frame procesado")

            # Guardar y ofrecer descarga
            image = Image.fromarray(img_rgb)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            filename = f"deteccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

            st.download_button(
                label="💾 Descargar imagen procesada",
                data=buffer,
                file_name=filename,
                mime="image/jpeg",
            )
