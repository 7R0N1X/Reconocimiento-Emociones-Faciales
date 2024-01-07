import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np

MODELO_RUTA = '../modelos/modelos-entrenados/DEF.h5'
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


def procesar_imagen(ruta_imagen, carpeta_salida, modelo, cascade):
    """
    :param ruta_imagen: La ruta de la imagen a procesar.
    :param carpeta_salida: La carpeta donde se guardarán los rostros detectados por emoción.
    :param modelo: El modelo entrenado para la detección de emociones.
    :param cascade: El clasificador Haar Cascade para detección facial.
    """

    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        print(f"No se pudo leer la imagen: {ruta_imagen}")
        return

    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = cascade.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro = imagen_gris[y:y + h, x:x + w]
        rostro_redimensionado = cv2.resize(rostro, (TAMANIO_IMAGEN, TAMANIO_IMAGEN))
        emocion = detectar_emocion(rostro_redimensionado, modelo)

        carpeta_emocion = os.path.join(carpeta_salida, emocion)
        if not os.path.exists(carpeta_emocion):
            os.makedirs(carpeta_emocion)

        ruta_salida = f"{carpeta_emocion}/rostro_{len(os.listdir(carpeta_emocion))}.jpg"
        cv2.imwrite(ruta_salida, rostro_redimensionado)

    cv2.destroyAllWindows()


def procesar_imagenes_en_carpeta(carpeta_entrada, carpeta_salida, modelo, cascade):
    """
    :param carpeta_entrada: La ruta de la carpeta que contiene las imágenes de entrada a procesar.
    :param carpeta_salida: La carpeta donde se guardarán los rostros detectados por emoción en las imágenes.
    :param modelo: El modelo entrenado para la detección de emociones.
    :param cascade: El clasificador Haar Cascade para detección facial.
    """

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    for nombre_imagen in os.listdir(carpeta_entrada):
        ruta_imagen = os.path.join(carpeta_entrada, nombre_imagen)
        procesar_imagen(ruta_imagen, carpeta_salida, modelo, cascade)


ruta_carpeta_imagenes = 'E:\\img'
carpeta_salida = 'rostros-por-emocion'

cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
modelo = cargar_modelo(MODELO_RUTA)

procesar_imagenes_en_carpeta(ruta_carpeta_imagenes, carpeta_salida, modelo, cascade_clasificador)
