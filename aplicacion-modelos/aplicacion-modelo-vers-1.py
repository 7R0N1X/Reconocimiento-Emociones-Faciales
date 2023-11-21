import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo
nombre_modelo = '../modelos-entrenados/modelo_13.h5'
modelo = load_model(nombre_modelo)

# Inicializar el clasificador Haar Cascade para detección facial
cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tamaño de la imagen que se espera en el modelo
tamaño_de_imagen = 48

# Mapeo de etiquetas numéricas a nombres de clase
diccionario_clases = {0: 'anger', 1: 'neutral', 2: 'disgust', 3: 'fear', 4: 'happy'}


# Función para preprocesar el frame de la cámara
def preprocesar_frame(frame):
    frame_redimensionado = cv2.resize(frame, (tamaño_de_imagen, tamaño_de_imagen))
    frame_normalizado = frame_redimensionado / 255.0
    frame_reshape = np.reshape(frame_normalizado, (1, tamaño_de_imagen, tamaño_de_imagen, 1))
    return frame_reshape


# Iniciar la captura de la cámara
captura = cv2.VideoCapture(0)

while True:
    # Leer el frame de la cámara
    ret, frame = captura.read()

    # Convertir el frame a escala de grises para la detección facial
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame
    caras = cascade_clasificador.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterar sobre las caras detectadas
    for (x, y, w, h) in caras:
        # Recortar la región de la cara
        cara_recortada = frame_gris[y:y + h, x:x + w]  # Convertir la cara a escala de grises

        # Preprocesar la cara
        cara_preprocesada = preprocesar_frame(cara_recortada)

        # Realizar la predicción
        predicción = modelo.predict(cara_preprocesada)

        # Convertir las probabilidades a una etiqueta y porcentaje
        etiqueta_predicha = np.argmax(predicción)
        porcentajes = predicción[0] * 100

        # Obtener el nombre de la clase con el porcentaje más alto
        nombre_clase = diccionario_clases.get(etiqueta_predicha, 'Desconocida')
        porcentaje_maximo = max(porcentajes)

        # Mostrar los porcentajes en la nueva ventana
        resultados_texto = f"{diccionario_clases[0]}: {porcentajes[0]:.2f}%, {diccionario_clases[1]}: {porcentajes[1]:.2f}%, {diccionario_clases[2]}: {porcentajes[2]:.2f}%, {diccionario_clases[3]}: {porcentajes[3]:.2f}%, {diccionario_clases[4]}: {porcentajes[4]:.2f}%"
        cv2.putText(frame, resultados_texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar el nombre de la clase predicha y el porcentaje de confianza sobre el cuadro
        texto_clase_maxima = f"Clase: {nombre_clase}, Confianza: {porcentaje_maximo:.2f}%"
        cv2.putText(frame, texto_clase_maxima, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow('Captura', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
captura.release()
cv2.destroyAllWindows()
