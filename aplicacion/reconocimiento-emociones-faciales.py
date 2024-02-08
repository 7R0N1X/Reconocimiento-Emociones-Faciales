import cv2
import numpy as np
from keras.models import load_model
import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment
import time
from datetime import timedelta

# Constantes
RUTA_MODELO = '../modelos/modelos-entrenados/DEF_RELU.h5'
RUTA_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
TAMANO_IMAGEN = 48
ETIQUETAS_EMOCIONES = {0: 'enojado', 1: 'neutral', 2: 'disgusto', 3: 'miedo', 4: 'feliz', 5: 'triste'}
ARCHIVO_EXCEL = 'estadisticas_emociones.xlsx'


class AnalizadorEmociones:

    def __init__(self, ruta_modelo, ruta_cascade, tamano_imagen, etiquetas_emociones):
        """
        :param ruta_modelo: Ruta al archivo que contiene el modelo.
        :param ruta_cascade: Ruta al archivo de clasificador Haar Cascade para detección facial.
        :param tamano_imagen: Tamaño de imagen esperado por el modelo.
        :param etiquetas_emociones: Mapeo de etiquetas numéricas a nombres de clases emocionales.
        """
        self.modelo = load_model(ruta_modelo)
        self.cascade_clasificador = cv2.CascadeClassifier(ruta_cascade)
        self.tamano_imagen = tamano_imagen
        self.etiquetas_emociones = etiquetas_emociones
        # Diccionario para almacenar recuentos de emociones detectadas
        self.contadores_emociones = {etiqueta: 0 for etiqueta in etiquetas_emociones}
        # Emoción detectada en el frame anterior
        self.emocion_previa = None
        self.tiempo_inicio = time.time()

    def preprocesar_frame(self, frame):
        """
        Realiza el preprocesamiento de un frame de imagen para adaptarlo al modelo de red neuronal.
        :param frame: Frame de imagen a preprocesar.
        :return: Frame preprocesado.
        """
        frame_redimensionado = cv2.resize(frame, (self.tamano_imagen, self.tamano_imagen))
        frame_normalizado = frame_redimensionado / 255.0
        return np.reshape(frame_normalizado, (1, self.tamano_imagen, self.tamano_imagen, 1))

    def predecir_emocion(self, cara):
        """
        Realiza la predicción de la emoción en una cara utilizando el modelo.
        :param cara: Imagen de la cara para la predicción.
        :return: Etiqueta predicha y porcentajes de confianza.
        """
        cara_preprocesada = self.preprocesar_frame(cara)
        prediccion = self.modelo.predict(cara_preprocesada)
        return np.argmax(prediccion), prediccion[0] * 100

    def actualizar_contadores_emociones(self, etiqueta_predicha):
        """
        Actualiza los contadores de emociones detectadas.
        :param etiqueta_predicha: Etiqueta de la emoción predicha.
        """
        if etiqueta_predicha != self.emocion_previa:
            self.contadores_emociones[etiqueta_predicha] += 1
            self.emocion_previa = etiqueta_predicha

    def obtener_tiempo_formateado(self):
        tiempo_transcurrido = round(time.time() - self.tiempo_inicio)
        tiempo_delta = timedelta(seconds=tiempo_transcurrido)
        horas, segundos = divmod(tiempo_delta.seconds, 3600), tiempo_delta.seconds % 3600
        tiempo_formateado = "{:02}:{:02}:{:02}".format(horas[0], segundos // 60, segundos % 60)
        return tiempo_formateado

    def mostrar_info_en_frame(self, frame, porcentajes, bounding_box, etiqueta_emocion, confianza):
        """
        Muestra información de emociones en el frame, incluyendo porcentajes y etiquetas.
        :param frame: Frame de imagen en el que se mostrará la información.
        :param porcentajes: Porcentajes de confianza para cada emoción.
        :param bounding_box: Coordenadas de la caja delimitadora de la cara.
        :param etiqueta_emocion: Etiqueta de la emoción predicha.
        :param confianza: Porcentaje de confianza en la predicción.
        """
        # Crear una cadena de texto que incluye etiquetas de emociones y porcentajes formateados
        texto = ', '.join([f"{self.etiquetas_emociones[i]}: {porcentajes[i]:.2f}%" for i in range(len(self.etiquetas_emociones))])

        # Mostrar el texto en la ventana de la imagen
        cv2.putText(frame, texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Dibujar un rectángulo alrededor de la cara en la imagen
        cv2.rectangle(frame, bounding_box[0], bounding_box[1], (0, 255, 0), 2)

        # Crear una cadena de texto con la etiqueta de emoción y el nivel de confianza
        texto_etiqueta = f"Clase: {etiqueta_emocion}, Confianza: {confianza:.2f}%"

        # Mostrar el texto de la etiqueta encima del rectángulo de la cara
        cv2.putText(frame, texto_etiqueta, (bounding_box[0][0], bounding_box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Mostrar el tiempo formateado en la ventana
        cv2.putText(frame, f"Tiempo: {self.obtener_tiempo_formateado()}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    def procesar_frame(self, frame):
        """
        Procesa un frame de la cámara para detectar y analizar emociones.
        :param frame: Frame de la cámara.
        """
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en la imagen utilizando el clasificador Haar Cascade
        caras = self.cascade_clasificador.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterar sobre las coordenadas de las caras detectadas
        for (x, y, w, h) in caras:
            # Recortar la región de la cara de la imagen en escala de grises
            cara_recortada = frame_gris[y:y + h, x:x + w]

            # Predecir la emoción en la cara recortada
            etiqueta_predicha, porcentajes = self.predecir_emocion(cara_recortada)

            # Actualizar los contadores de emociones
            self.actualizar_contadores_emociones(etiqueta_predicha)

            # Mostrar información sobre la emoción
            self.mostrar_info_en_frame(frame, porcentajes, ((x, y), (x + w, y + h)),
                                       self.etiquetas_emociones.get(etiqueta_predicha, 'Desconocida'),
                                       max(porcentajes))


def crear_reporte_excel(contadores_emociones, etiquetas_emociones, archivo_excel, tiempo_transcurrido):
    """
    Crea un informe en formato Excel con estadísticas de emociones detectadas.
    :param contadores_emociones: Diccionario con los recuentos de emociones detectadas.
    :param etiquetas_emociones: Mapeo de etiquetas numéricas a nombres de clases emocionales.
    :param archivo_excel: Nombre del archivo Excel que se creará o modificará.
    """
    # Crear un nuevo libro de trabajo de Excel
    libro_trabajo = openpyxl.Workbook()
    hoja = libro_trabajo.active

    # Establecer los encabezados de las columnas en la hoja de cálculo
    hoja['A1'] = 'Emoción'
    hoja['B1'] = 'Cantidad'
    hoja['C1'] = 'Porcentaje'
    hoja['D1'] = 'Tiempo Transcurrido'

    # Calcular el total de emociones sumando los valores del diccionario 'contadores_emociones'
    total_emociones = sum(contadores_emociones.values())

    # Iterar sobre las emociones y cantidades en el diccionario 'contadores_emociones'
    for fila, (emocion, cantidad) in enumerate(contadores_emociones.items(), start=2):
        # Obtener el nombre de la emoción del diccionario 'etiquetas_emociones' o establecerlo como 'Desconocida' si no está definido
        nombre_emocion = etiquetas_emociones.get(emocion, 'Desconocida')

        # Calcular el porcentaje de la emoción en relación con el total de emociones
        porcentaje_emocion = (cantidad / total_emociones) * 100 if total_emociones > 0 else 0
        porcentaje_emocion = round(porcentaje_emocion, 2)

        # Agregar los datos de la emoción, cantidad y porcentaje en la hoja de cálculo
        hoja[f'A{fila}'] = nombre_emocion
        hoja[f'B{fila}'] = cantidad
        hoja[f'C{fila}'] = porcentaje_emocion

    # Agregar el tiempo transcurrido en la última fila
    hoja[f'D{fila + 1}'] = tiempo_transcurrido
    hoja[f'D{fila + 1}'].alignment = Alignment(horizontal='left')

    # Crear referencias para los datos y categorías del gráfico de barras
    valores = Reference(hoja, min_col=2, min_row=1, max_col=2, max_row=fila + 1)
    categorias = Reference(hoja, min_col=1, min_row=2, max_row=fila + 1)

    # Crear un gráfico de barras
    grafico_barras = BarChart()
    grafico_barras.add_data(valores, titles_from_data=True)
    grafico_barras.set_categories(categorias)
    grafico_barras.title = 'Estadísticas de emociones detectadas'
    grafico_barras.legend = None

    # Agregar el gráfico a la hoja de cálculo en la posición E2
    hoja.add_chart(grafico_barras, 'E2')

    # Guardar los cambios en el libro de trabajo de Excel con el nombre de archivo especificado
    libro_trabajo.save(filename=archivo_excel)


if __name__ == "__main__":

    analizador_emociones = AnalizadorEmociones(RUTA_MODELO, RUTA_CASCADE, TAMANO_IMAGEN, ETIQUETAS_EMOCIONES)

    captura = cv2.VideoCapture(0)

    while True:
        ret, frame = captura.read()
        analizador_emociones.procesar_frame(frame)
        cv2.imshow('Captura', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        tiempo_formateado = analizador_emociones.obtener_tiempo_formateado()
        crear_reporte_excel(analizador_emociones.contadores_emociones, ETIQUETAS_EMOCIONES, ARCHIVO_EXCEL, tiempo_formateado)

    captura.release()
    cv2.destroyAllWindows()
