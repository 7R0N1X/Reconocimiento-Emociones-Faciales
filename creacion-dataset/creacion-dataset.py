import cv2
import os


class CreacionDataset:

    def __init__(self, ruta_video, directorio_dataset):
        self.ruta_video = ruta_video
        self.directorio_dataset = directorio_dataset
        self.cascada_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.contador = 0
        self.contador_fotogramas = 0
        self.ventana_ancho = 1280
        self.ventana_alto = 720

    def crear_directorio_dataset(self):
        if not os.path.exists(self.directorio_dataset):
            os.makedirs(self.directorio_dataset)

    def procesar_video(self):
        # Inicializa el objeto para capturar video desde el archivo
        captura = cv2.VideoCapture(self.ruta_video)

        # Redimensiona la ventana del video
        cv2.namedWindow('Captura de Rostros', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Captura de Rostros', self.ventana_ancho, self.ventana_alto)

        while True:
            # Captura un fotograma desde el video
            ret, fotograma = captura.read()

            if not ret:
                break

            # Incrementa el contador de fotogramas
            self.contador_fotogramas += 1

            # Convierte el fotograma a escala de grises para la detección de rostros
            gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

            # Detecta rostros en el fotograma cada 10 cuadros
            if self.contador_fotogramas % 10 == 0:
                # Detecta rostros en el fotograma
                rostros = self.cascada_rostro.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

                # Procesa cada rostro detectado
                for (x, y, w, h) in rostros:
                    # Recorta la cara en escala de grises
                    cara_recortada_gris = cv2.resize(gris[y:y + h, x:x + w], (48, 48))
                    cv2.rectangle(fotograma, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Guarda la cara recortada en la carpeta correspondiente
                    directorio_emocion = os.path.join(self.directorio_dataset)
                    if not os.path.exists(directorio_emocion):
                        os.makedirs(directorio_emocion)

                    nombre_archivo = os.path.join(directorio_emocion, f"{self.contador}.jpg")
                    cv2.imwrite(nombre_archivo, cara_recortada_gris)
                    self.contador += 1

            # Obtén el tiempo actual del video en segundos
            tiempo_actual_segundos = captura.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Formatea el tiempo en un formato legible, por ejemplo, minutos:segundos
            tiempo_formateado = f"Tiempo: {int(tiempo_actual_segundos // 60)}:{int(tiempo_actual_segundos % 60)}"

            # Agrega el texto al fotograma actual
            cv2.putText(fotograma, tiempo_formateado, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Muestra el fotograma en la ventana redimensionada
            cv2.imshow('Captura de Rostros', fotograma)

            # Sale del bucle si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera el video y cierra la ventana
        captura.release()
        cv2.destroyAllWindows()


# Uso de la clase
if __name__ == "__main__":
    ruta_video = ''
    nombre_carpeta = "faces"

    recognizer = CreacionDataset(ruta_video, nombre_carpeta)
    recognizer.crear_directorio_dataset()
    recognizer.procesar_video()
