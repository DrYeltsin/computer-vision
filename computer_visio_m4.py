# ============================================
# APP WEB: DETECTOR DE FORMAS GEOMÉTRICAS
# Compatible con macOS (Intel, M1, M2, M3, M4) y Windows
# Funciona con cámaras físicas y virtuales (Iriun, OBS, etc.)
# ============================================

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import io

# --------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------
st.set_page_config(page_title="Detector de Formas", layout="wide")
st.title("🎥 Detector de Formas Geométricas")
st.caption("Compatible con macOS (M1–M4), Windows y cámaras virtuales (Iriun, OBS, etc.)")

# --------------------------
# FUNCIÓN: DETECTOR DE FORMAS
# --------------------------
def detectar_formas(frame_bgr):
    conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Aumentar contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gris_eq = clahe.apply(gris)
    blur = cv2.GaussianBlur(gris_eq, (3, 3), 0)

    # Detección de bordes
    v = np.median(blur)
    sigma = 0.25
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blur, lower, upper)

    # Morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame_bgr.shape[:2]
    area_min = max(500.0, 0.0002 * (h * w))

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_min:
            continue

        peri = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.04 * peri, True)
        x, y, bw, bh = cv2.boundingRect(approx)

        circularidad = 0
        if peri > 0:
            circularidad = 4 * np.pi * (area / (peri * peri))

        forma = None
        color = None

        if len(approx) == 3:
            forma = "Triangulo"
            color = (0, 255, 255)
        elif len(approx) == 4:
            aspect_ratio = bw / float(bh)
            if 0.95 <= aspect_ratio <= 1.05:
                forma = "Cuadrado"
                color = (255, 0, 0)
            else:
                forma = "Rectangulo"
                color = (0, 255, 0)
        else:
            if len(approx) >= 8 and circularidad > 0.85:
                forma = "Circulo"
                color = (0, 0, 255)

        if forma:
            conteo[forma] += 1
            cv2.drawContours(frame_bgr, [approx], -1, color, 2)
            cv2.putText(frame_bgr, forma, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame_bgr, conteo

# --------------------------
# FUNCIÓN: HISTOGRAMA
# --------------------------
def mostrar_histograma(conteo):
    fig, ax = plt.subplots(figsize=(4, 2))
    formas = list(conteo.keys())
    valores = list(conteo.values())
    ax.bar(formas, valores, color=['gold', 'blue', 'green', 'red'])
    ax.set_title("Conteo de Formas")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)

# --------------------------
# INTERFAZ DE USUARIO
# --------------------------
modo = st.radio("Selecciona el modo de entrada:", ["📸 Cámara", "📁 Subir imagen"])

col1, col2 = st.columns(2)

if modo == "📸 Cámara":
    st.info("Usa tu cámara integrada o Iriun Webcam. Luego captura una imagen.")
    img_cam = st.camera_input("Captura una imagen:")
    if img_cam:
        img_pil = Image.open(img_cam)
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        resultado, conteo = detectar_formas(frame_bgr)

        with col1:
            st.subheader("Resultado de detección")
            st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Conteo de formas")
            mostrar_histograma(conteo)
            st.json(conteo)

elif modo == "📁 Subir imagen":
    img_file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if img_file:
        img_pil = Image.open(img_file)
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        resultado, conteo = detectar_formas(frame_bgr)

        with col1:
            st.subheader("Resultado de detección")
            st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Conteo de formas")
            mostrar_histograma(conteo)
            st.json(conteo)
