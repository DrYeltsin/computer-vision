# ============================================
# DETECTOR DE FORMAS EN VIDEO (v3)
# Con contador de 10 segundos y botón de reinicio
# Compatible con macOS M1–M4 y Windows
# ============================================

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import matplotlib.pyplot as plt
import time

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
# CLASE STREAMING
# --------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.total_conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
        self.freeze = False

    def transform(self, frame):
        if self.freeze:
            return frame.to_ndarray(format="bgr24")
        frame_bgr = frame.to_ndarray(format="bgr24")
        resultado, conteo = detectar_formas(frame_bgr)
        for k in conteo:
            self.total_conteo[k] += conteo[k]
        return resultado

# --------------------------
# INTERFAZ STREAMLIT
# --------------------------
st.set_page_config(page_title="Detector de Formas (Live + Timer)", layout="wide")
st.title("🎥 Detector de Formas Geométricas — En Tiempo Real (10s)")
st.caption("Optimizado para macOS M4, Windows y cámaras virtuales (Iriun, OBS, etc.)")

# Estado persistente
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False
if "conteo_final" not in st.session_state:
    st.session_state.conteo_final = None

# Botones
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Iniciar detección (10s)"):
        st.session_state.start_time = time.time()
        st.session_state.running = True
        st.session_state.conteo_final = None
with col2:
    if st.button("🔁 Reiniciar"):
        st.session_state.start_time = None
        st.session_state.running = False
        st.session_state.conteo_final = None
        st.experimental_rerun()

# Stream de video (CORREGIDO)
ctx = webrtc_streamer(
    key="form-detector-timer",
    mode=WebRtcMode.TRANSFORM,  # ✅ CORREGIDO
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# --------------------------
# TEMPORIZADOR
# --------------------------
if ctx.video_processor:
    processor = ctx.video_processor

    if st.session_state.running:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, 10 - elapsed)
        st.subheader(f"🕒 Tiempo restante: {remaining:0.1f} segundos")

        if remaining <= 0:
            processor.freeze = True
            st.session_state.running = False
            st.session_state.conteo_final = processor.total_conteo
            st.success("✅ ¡Tiempo completado! Detección finalizada.")
    else:
        if st.session_state.conteo_final:
            st.subheader("📊 Conteo Final de Formas Detectadas (10s)")
            conteo = st.session_state.conteo_final
            fig, ax = plt.subplots(figsize=(4, 2))
            formas = list(conteo.keys())
            valores = list(conteo.values())
            ax.bar(formas, valores, color=['gold', 'blue', 'green', 'red'])
            ax.set_ylabel("Cantidad")
            st.pyplot(fig)
            st.json(conteo)
