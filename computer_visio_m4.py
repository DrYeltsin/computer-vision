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
        circularidad = 4 * np.pi * area / (
