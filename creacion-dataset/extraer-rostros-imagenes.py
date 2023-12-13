import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo entrenado para la detección de emociones
nombre_modelo = '../modelos/modelos-entrenados/DEF.h5'
modelo = load_model(nombre_modelo)

# Clasificador Haar Cascade para detección facial
cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tamaño de la imagen que se espera en el modelo
tamanio_de_imagen = 48

# Mapeo de etiquetas numéricas a nombres de clase
diccionario_clases = {0: 'enojado', 1: 'neutral', 2: 'disgusto', 3: 'miedo', 4: 'feliz'}


def procesar_imagen(ruta_imagen, carpeta_salida):
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"No se pudo leer la imagen: {ruta_imagen}")
        return

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    rostros = cascade_clasificador.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5)

    # Iterar sobre los rostros detectados
    for (x, y, w, h) in rostros:
        # Extraer el rostro de la región de interés
        rostro = imagen_gris[y:y+h, x:x+w]
        rostro_redimensionado = cv2.resize(rostro, (tamanio_de_imagen, tamanio_de_imagen))

        # Normalizar el rostro para la entrada del modelo
        rostro_normalizado = rostro_redimensionado / 255.0
        rostro_normalizado = np.expand_dims(rostro_normalizado, axis=0)
        rostro_normalizado = np.expand_dims(rostro_normalizado, axis=-1)

        # Predecir la emoción utilizando el modelo entrenado
        prediccion = modelo.predict(rostro_normalizado)
        emocion = diccionario_clases[np.argmax(prediccion)]

        # Crear la carpeta para la emoción detectada si no existe
        carpeta_emocion = os.path.join(carpeta_salida, emocion)
        if not os.path.exists(carpeta_emocion):
            os.makedirs(carpeta_emocion)

        # Guardar el rostro en la carpeta correspondiente a la emoción detectada
        ruta_salida = f"{carpeta_emocion}/rostro_{len(os.listdir(carpeta_emocion))}.jpg"
        cv2.imwrite(ruta_salida, rostro_redimensionado)

    cv2.destroyAllWindows()


def procesar_imagenes_en_carpeta(carpeta_entrada, carpeta_salida):
    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Iterar sobre cada imagen en la carpeta de entrada
    for nombre_imagen in os.listdir(carpeta_entrada):
        ruta_imagen = os.path.join(carpeta_entrada, nombre_imagen)
        procesar_imagen(ruta_imagen, carpeta_salida)


# Ruta de la carpeta que contiene las imágenes de entrada
ruta_carpeta_imagenes = 'E:\\img'

# Carpeta de salida para los rostros detectados por emoción en las imágenes
carpeta_salida = 'rostros-por-emocion'

# Llamar a la función para procesar las imágenes en la carpeta de entrada
procesar_imagenes_en_carpeta(ruta_carpeta_imagenes, carpeta_salida)
