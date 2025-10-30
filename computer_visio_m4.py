# ============================================
# DETECTOR DE FORMAS EN VIDEO EN VIVO CON GUARDADO AUTOMÁTICO
# ============================================
# Autor: Yeltsin Solano Díaz
# Compatible con macOS M4, Windows y Streamlit Cloud
# ============================================

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from datetime import datetime
from PIL import Image
import io
import time

# --------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------
st.set_page_config(page_title="Detector de Formas Geométricas", layout="wide")
st.title("🎥 Detección de Formas Geométricas — En Tiempo Real")
st.caption("Versión con cámara en vivo, temporizador y opción para guardar imagen final.")

# --------------------------
# CONFIGURACIÓN STUN/TURN
# --------------------------
rtc_config = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:global.relay.metered.ca:80", "turn:global.relay.metered.ca:443"],
            "username": "openai",
            "credential": "streamlit2025",
        },
    ]
}

# --------------------------
# FUNCIÓN DE DETECCIÓN DE FORMAS
# --------------------------
def detectar_formas(frame_bgr):
    conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}

    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris_eq = clahe.apply(gris)
    blur = cv2.medianBlur(gris_eq, 5)

    edges = cv2.Canny(blur, 50, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame_bgr.shape[:2]
    area_min = 800.0
    area_max = 0.25 * h * w

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_min or area > area_max:
            continue

        peri = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.03 * peri, True)
        x, y, bw, bh = cv2.boundingRect(approx)

        aspect_ratio = bw / float(bh)
        circularidad = 4 * np.pi * area / (peri * peri + 1e-5)
        forma, color = None, None

        if len(approx) == 3 and circularidad > 0.3:
            forma, color = "Triangulo", (0, 255, 255)
        elif len(approx) == 4:
            if 0.9 <= aspect_ratio <= 1.1:
                forma, color = "Cuadrado", (255, 0, 0)
            elif 0.5 <= aspect_ratio <= 2.0:
                forma, color = "Rectangulo", (0, 255, 0)
        elif len(approx) > 6 and circularidad > 0.82:
            forma, color = "Circulo", (0, 0, 255)

        if forma:
            conteo[forma] += 1
            cv2.drawContours(frame_bgr, [approx], -1, color, 2)
            cv2.putText(frame_bgr, forma, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame_bgr, conteo

# --------------------------
# CLASE DE VIDEO STREAMLIT
# --------------------------
class ShapeDetector(VideoTransformerBase):
    def __init__(self):
        self.total_conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
        self.last_frame = None
        self.freeze = False
        self.remaining = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not self.freeze:
            resultado, conteo = detectar_formas(img)
            self.last_frame = resultado.copy()
            for k in conteo:
                self.total_conteo[k] += conteo[k]
        else:
            resultado = self.last_frame if self.last_frame is not None else img

        # Mostrar temporizador en video
        if self.remaining is not None:
            cv2.putText(resultado, f"⏱ {self.remaining:0.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return resultado

# --------------------------
# INTERFAZ STREAMLIT
# --------------------------
duracion = st.slider("⏱️ Duración del análisis (segundos):", 5, 30, 10, step=5)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Iniciar detección"):
        st.session_state.start_time = time.time()
        st.session_state.running = True
        st.session_state.resultado = None
with col2:
    if st.button("🔁 Reiniciar"):
        st.session_state.running = False
        st.session_state.resultado = None
        st.rerun()

st.info("Activa tu cámara para ver la detección en tiempo real. Usa la webcam o Iriun Camera.")
ctx = webrtc_streamer(
    key="form-detector-final",
    video_processor_factory=ShapeDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration=rtc_config,
)

# --------------------------
# TEMPORIZADOR Y GUARDADO
# --------------------------
if ctx.video_processor:
    processor = ctx.video_processor

    if "running" in st.session_state and st.session_state.running:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, duracion - elapsed)
        processor.remaining = remaining

        st.subheader(f"🕒 Tiempo restante: {remaining:0.1f} s")

        if remaining <= 0:
            processor.freeze = True
            st.session_state.running = False

            # Guardar imagen al finalizar
            if processor.last_frame is not None:
                img_bgr = processor.last_frame
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"deteccion_{timestamp}.jpg"

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                buffer.seek(0)

                st.success("✅ Análisis finalizado. Imagen guardada.")
                st.download_button(
                    label="💾 Descargar imagen con resultados",
                    data=buffer,
                    file_name=filename,
                    mime="image/jpeg",
                )

                st.subheader("📊 Conteo de formas detectadas")
                st.json(processor.total_conteo)
    else:
        st.warning("Presiona ▶️ para iniciar el análisis.")
