import cv2
import os


class ProcesadorRostrosDesdeVideo:

    def __init__(self, ruta_video, directorio_dataset):
        """
        Inicializa el ProcesadorRostrosDesdeVideo con la ruta del video y el directorio para almacenar las imágenes.

        Parámetros:
        - ruta_video (str): La ruta del archivo de video.
        - directorio_dataset (str): El directorio donde se guardarán las imágenes recortadas.
        - cascada_rostro: Objeto para la detección de rostros mediante cascada.
        - contador (int): Contador de imágenes procesadas.
        - ventana_ancho (int): Ancho de la ventana de visualización.
        - ventana_alto (int): Alto de la ventana de visualización.
        """
        self.ruta_video = ruta_video
        self.directorio_dataset = directorio_dataset
        self.cascada_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.contador = 0
        self.ventana_ancho = 1280
        self.ventana_alto = 720

    def crear_directorio_dataset(self):
        # Crea el directorio de destino si no existe.
        os.makedirs(self.directorio_dataset, exist_ok=True)

    def procesar_video(self):
        """
        Procesa el video, detecta rostros en cada fotograma y guarda las caras recortadas en el directorio especificado.
        """
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

            # Convierte el fotograma a escala de grises para la detección de rostros
            gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

            # Detecta rostros en el fotograma según el número de cuadros
            if self.contador % 1 == 0:
                # Detecta rostros en el fotograma
                rostros = self.cascada_rostro.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

                # Procesa cada rostro detectado
                for (x, y, w, h) in rostros:
                    # Recorta la cara en escala de grises
                    cara_recortada_gris = cv2.resize(gris[y:y + h, x:x + w], (48, 48))
                    cv2.rectangle(fotograma, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Guarda la cara recortada en la carpeta correspondiente
                    directorio_emocion = os.path.join(self.directorio_dataset)
                    os.makedirs(directorio_emocion, exist_ok=True)

                    nombre_archivo = os.path.join(directorio_emocion, f"{self.contador}.jpg")
                    cv2.imwrite(nombre_archivo, cara_recortada_gris)
                    self.contador += 1

            # Muestra el video a procesar en la ventana redimensionada
            cv2.imshow('Captura de Rostros', fotograma)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        captura.release()
        cv2.destroyAllWindows()