import cv2
import numpy as np
import pyautogui
import webbrowser
from picamera2 import Picamera2
import time
import threading
import mediapipe as mp

# Parámetros de calibración
intrinsics = np.array([[1.48870074e+03, 0.00000000e+00, 6.74119880e+02],
                       [0.00000000e+00, 1.48905929e+03, 3.79822213e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs = np.array([[-0.05313665, 0.85315342, 0.00570786, 0.00509274, -1.55220213]])

# Configuración de la contraseña esperada
expected_sequence = ["square", "circle", "square", "circle"]
detected_sequence = []
stable_shape = None
stable_count = 0
required_stability = 10  # Cuadros necesarios para confirmar una forma
game_mode = False  # Modo 2048 activado tras detectar la contraseña

# Mapeo de formas a direcciones
shape_to_direction = {
    "square": "up",
    "circle": "down",
    "triangle": "left",
    "pentagon": "right"
}

# Inicialización de MediaPipe para detectar manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Función para identificar formas
def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    if len(approx) == 4:
        return "square"
    elif len(approx) > 4:
        return "circle"
    elif len(approx) == 3:
        return "triangle"
    elif len(approx) == 5:
        return "pentagon"
    else:
        return "unknown"

# Función para abrir el juego 2048 en Google
def open_2048_game():
    print("Abriendo 2048...")
    webbrowser.open("https://play2048.org/es/")
    time.sleep(5)  # Esperar a que el juego cargue

# Función para enviar comandos al juego 2048
def send_2048_control(direction):
    if game_mode:
        pyautogui.press(direction)
        time.sleep(0.2)  # Añadir un pequeño retraso entre acciones para evitar comandos repetidos

# Controlar el movimiento de la mano en relación al centro
def control_flechas(center_x, center_y):
    screen_center_x = 320  # Centro horizontal (640 px de ancho)
    screen_center_y = 240  # Centro vertical (480 px de alto)
    
    # Detectar si la mano está más cerca de un borde
    if center_x < screen_center_x - 50:  # Mano a la izquierda
        send_2048_control("left")
    elif center_x > screen_center_x + 50:  # Mano a la derecha
        send_2048_control("right")
    elif center_y < screen_center_y - 50:  # Mano arriba
        send_2048_control("up")
    elif center_y > screen_center_y + 50:  # Mano abajo
        send_2048_control("down")

# Configuración de la cámara
def stream_video():
    global stable_shape, stable_count, detected_sequence, game_mode, tracker, tracking

    picam = Picamera2()
    picam.preview_configuration.main.size = (640, 480)  # Resolución más baja para optimizar rendimiento
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    print("Muestra las formas en el orden: cuadrado, círculo, cuadrado, círculo.")
    
    # Crear el tracker
    tracker = cv2.TrackerCSRT_create()
    tracking = False  # Bandera que indica si se está siguiendo una forma

    while True:
        frame = picam.capture_array()

        # Corregir la distorsión en la imagen
        undistorted_frame = cv2.undistort(frame, intrinsics, dist_coeffs)

        # Convertir a escala de grises y detectar bordes
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shape = None
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filtrar contornos pequeños
                shape = detect_shape(contour)
                if game_mode or shape in expected_sequence[len(detected_sequence):len(detected_sequence)+1]:
                    detected_shape = shape
                    cv2.drawContours(undistorted_frame, [contour], -1, (0, 255, 0), 2)
                    
                    # Si es el momento adecuado, iniciar el tracker
                    if not tracking:
                        # Se puede añadir un buffer de estabilidad para asegurar que la forma se detecta correctamente
                        x, y, w, h = cv2.boundingRect(contour)
                        tracker.init(undistorted_frame, (x, y, w, h))
                        tracking = True
                    break

        # Si estamos en modo de juego 2048, hacer el seguimiento
        if game_mode and tracking:
            # Actualizar el tracker
            success, bbox = tracker.update(undistorted_frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Establecer la dirección de movimiento basada en la forma detectada
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                control_flechas(center_x, center_y)

        # Detectar la mano con MediaPipe
        results = hands.process(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = y_min = 640
                x_max = y_max = 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * 640), int(landmark.y * 480)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                # Si no se está siguiendo nada, iniciar el tracker
                if not tracking:
                    tracker.init(undistorted_frame, bbox)
                    tracking = True
                else:
                    success, bbox = tracker.update(undistorted_frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Controlar el juego 2048 basado en la posición de la mano
                        center_x = int(x + w / 2)
                        center_y = int(y + h / 2)
                        control_flechas(center_x, center_y)

        # Estabilización de la forma detectada
        if detected_shape == stable_shape:
            stable_count += 1
        else:
            stable_shape = detected_shape
            stable_count = 1

        # Confirmar forma estable
        if stable_count >= required_stability and stable_shape:
            if not game_mode:
                detected_sequence.append(stable_shape)
                print(f"Secuencia detectada: {detected_sequence}")
                
                # Validar si la contraseña es correcta
                if detected_sequence == expected_sequence:
                    print("¡Contraseña correcta! Activando el modo 2048.")
                    # Ejecutar el juego en un hilo separado
                    threading.Thread(target=open_2048_game).start()
                    game_mode = True

            stable_shape = None
            stable_count = 0

        # Mostrar en pantalla con OpenCV
        mode_text = "Modo: 2048" if game_mode else "Modo: Contraseña"
        sequence_text = "Secuencia: " + " -> ".join(detected_sequence) if not game_mode else "Controlando el 2048..."
        cv2.putText(undistorted_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(undistorted_frame, sequence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Mostrar la imagen con OpenCV
        cv2.imshow("Video Stream", undistorted_frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
