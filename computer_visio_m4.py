# ============================================
# APP WEB: DETECTOR DE FORMAS GEOMÉTRICAS v2
# Precisión mejorada en círculos (contornos + Hough)
# Compatible con macOS (M1–M4) y Windows
# ============================================

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------
st.set_page_config(page_title="Detector de Formas Geométricas", layout="wide")
st.title("🎯 Detector de Formas Geométricas (v2)")
st.caption("Detección precisa de triángulos, cuadrados, rectángulos y círculos — optimizado para cámaras reales/virtuales.")

# --------------------------
# FUNCIÓN: HISTOGRAMA
# --------------------------
def mostrar_histograma(conteo):
    fig, ax = plt.subplots(figsize=(4, 2))
    formas = list(conteo.keys())
    valores = list(conteo.values())
    colores = ['gold', 'blue', 'green', 'red']
    ax.bar(formas, valores, color=colores)
    ax.set_title("Conteo de Formas Detectadas")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)

# --------------------------
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# --------------------------
def detectar_formas_mejorado(frame_bgr):
    """
    Estrategia híbrida:
      1) Contornos + circularidad para polígonos y círculos muy definidos.
      2) HoughCircles solo si la escena tiene pocos bordes (evita saturación).
    """
    conteo = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Aumentar contraste + reducir ruido (medianBlur funciona muy bien en círculos)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris_eq = clahe.apply(gris)
    blur = cv2.medianBlur(gris_eq, 5)

    # 2) BORDES
    edges = cv2.Canny(blur, 50, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    # 3) CONTORNOS POLIGONALES
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame_bgr.shape[:2]
    area_min = max(500.0, 0.0002 * h * w)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_min:
            continue

        peri = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.04 * peri, True)
        x, y, bw, bh = cv2.boundingRect(approx)

        circularidad = 0.0
        if peri > 0:
            circularidad = 4 * np.pi * (area / (peri * peri))

        forma, color = None, None

        if len(approx) == 3:
            forma, color = "Triangulo", (0, 255, 255)
        elif len(approx) == 4:
            aspect_ratio = bw / float(bh)
            if 0.9 <= aspect_ratio <= 1.1:
                forma, color = "Cuadrado", (255, 0, 0)
            else:
                forma, color = "Rectangulo", (0, 255, 0)
        else:
            # círculo por contorno (bien definido)
            if circularidad > 0.85 and 0.8 <= (bw / bh) <= 1.2:
                forma, color = "Circulo", (0, 0, 255)

        if forma:
            conteo[forma] += 1
            cv2.drawContours(frame_bgr, [approx], -1, color, 2)
            cv2.putText(frame_bgr, forma, (x, max(15, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4) HOUGHCIRCLES: solo si la escena no está saturada de bordes
    edges_count = cv2.countNonZero(edges)
    if edges_count < (h * w * 0.02):
        circulos = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.4,
            minDist=60,
            param1=120,
            param2=45,   # subir reduce falsos positivos
            minRadius=20,
            maxRadius=250
        )
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for c in circulos[0, :]:
                cv2.circle(frame_bgr, (c[0], c[1]), c[2], (0, 0, 255), 2)
                cv2.circle(frame_bgr, (c[0], c[1]), 2, (255, 255, 255), 3)
                conteo["Circulo"] += 1

    return frame_bgr, conteo

# --------------------------
# INTERFAZ STREAMLIT
# --------------------------
modo = st.radio("Selecciona el modo de entrada:", ["📸 Cámara", "📁 Subir imagen"])
col1, col2 = st.columns(2)

if modo == "📸 Cámara":
    st.info("Captura una imagen con tu cámara integrada o Iriun Webcam.")
    img_cam = st.camera_input("Captura una imagen:")
    if img_cam:
        img_pil = Image.open(img_cam)
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        resultado, conteo = detectar_formas_mejorado(frame_bgr)

        with col1:
            st.subheader("Resultado de detección")
            st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Conteo de formas")
            mostrar_histograma(conteo)
            st.json(conteo)

elif modo == "📁 Subir imagen":
    img_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])
    if img_file:
        img_pil = Image.open(img_file)
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        resultado, conteo = detectar_formas_mejorado(frame_bgr)

        with col1:
            st.subheader("Resultado de detección")
            st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Conteo de formas")
            mostrar_histograma(conteo)
            st.json(conteo)
