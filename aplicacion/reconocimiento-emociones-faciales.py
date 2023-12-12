import cv2
import numpy as np
from keras.models import load_model
import openpyxl
from openpyxl.chart import BarChart, Reference

# Cargar el modelo
nombre_modelo = '../modelos-entrenados/DEF.h5'
modelo = load_model(nombre_modelo)

# Inicializar el clasificador Haar Cascade para detección facial
cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tamaño de la imagen que se espera en el modelo
tamanio_de_imagen = 48

# Mapeo de etiquetas numéricas a nombres de clase
diccionario_clases = {0: 'enojado', 1: 'neutral', 2: 'disgusto', 3: 'miedo', 4: 'feliz'}


# Función para preprocesar el frame de la cámara
def preprocesar_frame(frame):
    # Redimensionar el frame de imagen a un tamaño específico
    frame_redimensionado = cv2.resize(frame, (tamanio_de_imagen, tamanio_de_imagen))

    # Normalizar los valores
    frame_normalizado = frame_redimensionado / 255.0

    # Reorganizar la forma de la matriz para adaptarse al modelo de red neuronal convolucional (CNN)
    # Añade una dimensión adicional para el número de muestras (1), el tamaño de la imagen y un canal de color (1)
    frame_reshape = np.reshape(frame_normalizado, (1, tamanio_de_imagen, tamanio_de_imagen, 1))

    # Devuelve el frame preprocesado
    return frame_reshape


# Iniciar la captura de la cámara
captura = cv2.VideoCapture(0)

# Diccionario para almacenar recuentos de emociones detectadas
recuento_emociones = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

# Variable para almacenar la emoción previa
emocion_previa = None

while True:
    # Leer el frame de la cámara
    ret, frame = captura.read()

    # Convertir el frame a escala de grises para la detección facial
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame
    caras = cascade_clasificador.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in caras:
        # Recortar la región de la cara
        cara_recortada = frame_gris[y:y + h, x:x + w]

        # Preprocesar la cara
        cara_preprocesada = preprocesar_frame(cara_recortada)

        # Realizar la predicción
        prediccion = modelo.predict(cara_preprocesada)

        # Convertir las probabilidades a una etiqueta y porcentaje
        etiqueta_predicha = np.argmax(prediccion)
        porcentajes = prediccion[0] * 100

        # Obtener el nombre de la clase con el porcentaje más alto
        nombre_clase = diccionario_clases.get(etiqueta_predicha, 'Desconocida')
        porcentaje_maximo = max(porcentajes)

        # Verificar si la emoción detectada es diferente a la emoción previa
        if nombre_clase != emocion_previa:
            # Incrementar el recuento de emociones solo si la emoción cambia
            recuento_emociones[etiqueta_predicha] += 1
            emocion_previa = nombre_clase  # Actualizar la emoción previa

        # Mostrar los porcentajes en la ventana
        resultados_texto = (f"{diccionario_clases[0]}: {porcentajes[0]:.2f}%, "
                            f"{diccionario_clases[1]}: {porcentajes[1]:.2f}%, "
                            f"{diccionario_clases[2]}: {porcentajes[2]:.2f}%, "
                            f"{diccionario_clases[3]}: {porcentajes[3]:.2f}%, "
                            f"{diccionario_clases[4]}: {porcentajes[4]:.2f}%")
        cv2.putText(frame, resultados_texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, resultados_texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar el nombre de la clase predicha y el porcentaje de confianza sobre el cuadro
        texto_clase_maxima = f"Clase: {nombre_clase}, Confianza: {porcentaje_maximo:.2f}%"
        cv2.putText(frame, texto_clase_maxima, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, texto_clase_maxima, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Mostrar el frame
    cv2.imshow('Captura', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nombre del archivo Excel que se va a crear o modificar
archivo_excel = 'estadisticas_emociones.xlsx'

# Crear un nuevo libro de trabajo de Excel
workbook = openpyxl.Workbook()

# Obtener la hoja de trabajo activa del libro de Excel
sheet = workbook.active

# Establecer los encabezados de las columnas en la hoja de cálculo
sheet['A1'] = 'Emoción'
sheet['B1'] = 'Cantidad'
sheet['C1'] = 'Porcentaje'

# Calcular el total de emociones sumando los valores del diccionario 'recuento_emociones'
total_emociones = sum(recuento_emociones.values())

# Iterar sobre las emociones y cantidades en el diccionario 'recuento_emociones'
for row, (emocion, cantidad) in enumerate(recuento_emociones.items(), start=2):
    # Obtener el nombre de la emoción del diccionario 'diccionario_clases' o establecerlo como 'Desconocida' si no
    # está definido
    nombre_emocion = diccionario_clases.get(emocion, 'Desconocida')

    # Calcular el porcentaje de la emoción en relación con el total de emociones
    porcentaje_emocion = (cantidad / total_emociones) * 100 if total_emociones > 0 else 0
    porcentaje_emocion = round(porcentaje_emocion, 2)

    # Agregar los datos de la emoción, cantidad y porcentaje en la hoja de cálculo
    sheet[f'A{row}'] = nombre_emocion
    sheet[f'B{row}'] = cantidad
    sheet[f'C{row}'] = porcentaje_emocion

# Agregar un gráfico de barras para visualizar las estadísticas de emociones
values = Reference(sheet, min_col=2, min_row=1, max_col=2, max_row=len(recuento_emociones) + 1)
categories = Reference(sheet, min_col=1, min_row=2, max_row=len(recuento_emociones) + 1)

# Crear un gráfico de barras
chart = BarChart()

# Agregar los datos y configurar el gráfico
chart.add_data(values, titles_from_data=True)
chart.set_categories(categories)
chart.title = 'Estadísticas de emociones detectadas'
chart.legend = None

# Agregar el gráfico a la hoja de cálculo en la posición E2
sheet.add_chart(chart, 'E2')

# Guardar los cambios en el libro de trabajo de Excel con el nombre de archivo especificado
workbook.save(filename=archivo_excel)

# Liberar la captura y cerrar las ventanas
captura.release()
cv2.destroyAllWindows()
