import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np

MODELO_RUTA = '../modelos/modelos-entrenados/DEF_RELU.h5'
TAMANIO_IMAGEN = 48
DIC_CLASES = {0: 'enojado', 1: 'neutral', 2: 'disgusto', 3: 'miedo', 4: 'feliz', 5: 'triste'}


def cargar_modelo(ruta_modelo):
    """
    :param ruta_modelo: La ruta del modelo.
    :return: load_model: El modelo cargado.
    """
    return load_model(ruta_modelo)


def detectar_emocion(rostro, modelo):
    """
    :param rostro: La imagen del rostro.
    :param modelo: El modelo de detección de emociones.
    :return: La emoción detectada en el rostro.
    """
    rostro_normalizado = rostro / 255.0
    rostro_normalizado = np.expand_dims(rostro_normalizado, axis=0)
    rostro_normalizado = np.expand_dims(rostro_normalizado, axis=-1)

    prediccion = modelo.predict(rostro_normalizado)

    return DIC_CLASES[np.argmax(prediccion)]


def procesar_rostro(rostro, carpeta_emocion, modelo):
    """
    :param rostro: La imagen del rostro.
    :param carpeta_emocion: La carpeta de salida para la emoción detectada.
    :param modelo: El modelo de detección de emociones.
    :return: La emoción detectada en el rostro.
    """
    emocion = detectar_emocion(rostro, modelo)

    if not os.path.exists(carpeta_emocion):
        os.makedirs(carpeta_emocion)

    ruta_salida = f"{carpeta_emocion}/rostro_{len(os.listdir(carpeta_emocion))}.jpg"
    cv2.imwrite(ruta_salida, rostro)

    return emocion


def procesar_cuadro(frame, cascade, modelo, carpeta_salida):
    """
    :param frame: El cuadro actual del video.
    :param cascade: El clasificador de detección de rostros.
    :param modelo: El modelo de detección de emociones.
    :param carpeta_salida: La carpeta de salida para las imágenes de emociones detectadas.
    :return: El cuadro procesado con la detección de rostros y emociones.
    """

    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = cascade.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(TAMANIO_IMAGEN, TAMANIO_IMAGEN))
    for (x, y, w, h) in rostros:
        rostro = frame_gris[y:y + h, x:x + w]
        rostro_redimensionado = cv2.resize(rostro, (TAMANIO_IMAGEN, TAMANIO_IMAGEN))
        emocion = procesar_rostro(rostro_redimensionado, os.path.join(carpeta_salida, DIC_CLASES[0]), modelo)
        carpeta_emocion = os.path.join(carpeta_salida, emocion)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1, cv2.LINE_AA)
        procesar_rostro(rostro_redimensionado, carpeta_emocion, modelo)

    return frame


def detectar_rostros_video(input_video, carpeta_salida, espera_entre_cuadros_ms):
    """

    :param input_video: La ruta del archivo de video de entrada.
    :param carpeta_salida: La carpeta de salida para las imágenes de emociones detectadas.
    :param espera_entre_cuadros_ms: Tiempo de espera entre cada procesamiento de cuadro en milisegundos.
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    modelo = cargar_modelo(MODELO_RUTA)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error al abrir el archivo de video")
        return

    contador_cuadros = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        contador_cuadros += 1
        if contador_cuadros % espera_entre_cuadros_ms == 0:
            frame_procesado = procesar_cuadro(frame.copy(), cascade_clasificador, modelo, carpeta_salida)
            cv2.imshow('Video con detección de rostros', frame_procesado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


ruta_video_entrada = ''
carpeta_salida = 'rostros-por-emocion'
espera_entre_cuadros_ms = 24
detectar_rostros_video(ruta_video_entrada, carpeta_salida, espera_entre_cuadros_ms)
