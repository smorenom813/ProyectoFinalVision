# Proyecto de Control de Juego 2048 con Reconocimiento de Formas

Este proyecto utiliza la Raspberry Pi, OpenCV y MediaPipe para detectar formas geométricas con la cámara, las cuales sirven como una "contraseña" para activar el control de un juego de 2048 mediante el movimiento de la mano.

## Requisitos

- Raspberry Pi 4 o modelo superior con cámara compatible.
- Python 3.7 o superior.
- Librerías necesarias:
  - `opencv-python`
  - `numpy`
  - `pyautogui`
  - `webbrowser`
  - `picamera2`
  - `mediapipe`

### Instalación de dependencias

Para instalar las librerías necesarias, puedes ejecutar el siguiente comando:

```bash
pip install opencv-python numpy pyautogui mediapipe picamera2
