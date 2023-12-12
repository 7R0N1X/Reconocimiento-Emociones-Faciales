import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo entrenado para la detección de emociones
nombre_modelo = '../modelos-entrenados/DEF.h5'
modelo = load_model(nombre_modelo)

# Clasificador Haar Cascade para detección facial
cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tamaño de la imagen que se espera en el modelo
tamanio_de_imagen = 48

# Mapeo de etiquetas numéricas a nombres de clase
diccionario_clases = {0: 'enojado', 1: 'neutral', 2: 'disgusto', 3: 'miedo', 4: 'feliz'}


def procesar_cuadro(frame, carpeta_salida):
    # Convertir el frame a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    rostros = cascade_clasificador.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    # Iterar sobre los rostros detectados
    for (x, y, w, h) in rostros:
        # Extraer el rostro de la región de interés
        rostro = frame_gris[y:y+h, x:x+w]
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

        # Dibujar rectángulo y mostrar la emoción detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1, cv2.LINE_AA)

    return frame


def detectar_rostros_video(input_video, carpeta_salida, espera_entre_cuadros_ms):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Inicializar el objeto de captura de video
    cap = cv2.VideoCapture(input_video)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el archivo de video")
        return

    contador_cuadros = 0

    while True:
        # Capturar un frame del video
        ret, frame = cap.read()

        # Verificar si se ha llegado al final del video
        if not ret:
            break

        contador_cuadros += 1

        # Procesar el frame cada cierto número de milisegundos
        if contador_cuadros % espera_entre_cuadros_ms == 0:
            frame_procesado = procesar_cuadro(frame.copy(), carpeta_salida)

            # Mostrar el frame con la detección de rostros y emociones
            cv2.imshow('Video con detección de rostros', frame_procesado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar el objeto de captura y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()


# Ruta del archivo de video de entrada
ruta_video_entrada = 'E:\\video.mp4'

# Carpeta de salida para los rostros detectados por emoción
carpeta_salida = 'rostros-por-emocion'

# Espera entre cada procesamiento de cuadro en milisegundos (reducción de carga CPU)
espera_entre_cuadros_ms = 13

# Llamar a la función para detectar rostros en el video y guardarlos según la emoción detectada
detectar_rostros_video(ruta_video_entrada, carpeta_salida, espera_entre_cuadros_ms)
