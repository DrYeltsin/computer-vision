# ============================================
# DETECTOR DE FORMAS EN VIDEO (FINAL STREAMLIT CLOUD)
# En vivo + contador configurable + reinicio + overlay
# Compatible con macOS M4 / Windows / Streamlit Cloud
# ============================================

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt
import time

# --------------------------
# FUNCIÓN DE DETECCIÓN
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
# PROCESADOR DE VIDEO
# --------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.total_conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
        self.freeze = False
        self.remaining = None

    def transform(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")

        if not self.freeze:
            resultado, conteo = detectar_formas(frame_bgr)
            for k in conteo:
                self.total_conteo[k] += conteo[k]
        else:
            resultado = frame_bgr

        # Mostrar contador overlay si existe
        if self.remaining is not None:
            cv2.putText(
                resultado,
                f"⏱ {self.remaining:0.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return resultado


# --------------------------
# INTERFAZ STREAMLIT
# --------------------------
st.set_page_config(page_title="Detector de Formas (Final Live)", layout="wide")
st.title("🎥 Detector de Formas Geométricas — En Tiempo Real (Versión Final)")
st.caption("Optimizado para Streamlit Cloud • Compatible con macOS M4 y Windows")

# Estado persistente
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "running" not in st.session_state:
    st.session_state.running = False
if "conteo_final" not in st.session_state:
    st.session_state.conteo_final = None

# Ajuste de duración
duracion = st.slider("⏱️ Duración de detección (segundos):", 5, 30, 10, step=5)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Iniciar detección"):
        st.session_state.start_time = time.time()
        st.session_state.running = True
        st.session_state.conteo_final = None
with col2:
    if st.button("🔁 Reiniciar"):
        st.session_state.start_time = None
        st.session_state.running = False
        st.session_state.conteo_final = None
        st.rerun()  # ✅ versión moderna

# Cámara en vivo (modo automático, sin 'mode')
ctx = webrtc_streamer(
    key="form-detector-final",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# Temporizador y lógica
if ctx.video_processor:
    processor = ctx.video_processor

    if st.session_state.running:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, duracion - elapsed)
        processor.remaining = remaining
        st.subheader(f"🕒 Tiempo restante: {remaining:0.1f} segundos")

        if remaining <= 0:
            processor.freeze = True
            st.session_state.running = False
            st.session_state.conteo_final = processor.total_conteo
            st.success("✅ ¡Tiempo completado! Detección finalizada.")
    else:
        processor.remaining = None
        if st.session_state.conteo_final:
            st.subheader("📊 Conteo Final de Formas Detectadas")
            conteo = st.session_state.conteo_final
            fig, ax = plt.subplots(figsize=(4, 2))
            formas = list(conteo.keys())
            valores = list(conteo.values())
            ax.bar(formas, valores, color=['gold', 'blue', 'green', 'red'])
            ax.set_ylabel("Cantidad")
            st.pyplot(fig)
            st.json(conteo)
