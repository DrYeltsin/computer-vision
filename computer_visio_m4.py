# ============================================
# DETECTOR DE FORMAS - CAPTURA DE VIDEO 5s
# ============================================
# Autor: Yeltsin Solano Díaz
# Compatible con macOS M4, Windows y Streamlit Cloud
# ============================================

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import time
import os
from moviepy.editor import ImageSequenceClip
from datetime import datetime
from PIL import Image
import io

# --------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------
st.set_page_config(page_title="Detector de Formas en Video", layout="wide")
st.title("🎥 Detección de Formas en Video (5 segundos)")
st.caption("Captura un video de 5 segundos, analiza las formas y genera un video procesado con los resultados.")

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
# CLASE STREAMLIT VIDEO
# --------------------------
class VideoRecorder(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        self.recording = False
        self.start_time = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.recording:
            if time.time() - self.start_time < 5:
                self.frames.append(img.copy())
                cv2.putText(img, "🎥 Grabando...", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
            else:
                self.recording = False

        return img


# --------------------------
# INTERFAZ STREAMLIT
# --------------------------
st.info("Presiona 'Iniciar Grabación' para capturar un video de 5 segundos desde tu cámara.")
ctx = webrtc_streamer(
    key="form-detector-video",
    video_processor_factory=VideoRecorder,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration=rtc_config,
)

if ctx.video_processor:
    processor = ctx.video_processor

    if st.button("▶️ Iniciar Grabación (5s)"):
        processor.frames = []
        processor.recording = True
        processor.start_time = time.time()
        st.success("🎬 Grabando por 5 segundos...")

    if not processor.recording and len(processor.frames) > 0:
        st.success("✅ Grabación completada. Procesando video...")

        # Crear carpeta temporal
        temp_dir = tempfile.mkdtemp()
        processed_frames = []
        conteo_total = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}

        for frame in processor.frames:
            resultado, conteo = detectar_formas(frame)
            processed_frames.append(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
            for k in conteo:
                conteo_total[k] += conteo[k]

        # Crear video con MoviePy
        output_path = os.path.join(temp_dir, "deteccion_final.mp4")
        clip = ImageSequenceClip(processed_frames, fps=15)
        clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)

        # Imagen final
        last_frame = processed_frames[-1]
        image = Image.fromarray(last_frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Mostrar resultados
        st.subheader("📊 Conteo de formas detectadas")
        st.json(conteo_total)

        st.subheader("🎞️ Video procesado con detecciones")
        st.video(output_path)

        # Botones de descarga
        with open(output_path, "rb") as f:
            st.download_button("💾 Descargar video procesado", f, "deteccion_final.mp4", "video/mp4")

        st.download_button("🖼️ Descargar imagen final", buffer, "frame_final.jpg", "image/jpeg")
