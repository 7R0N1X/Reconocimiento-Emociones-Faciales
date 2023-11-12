import cv2
import os


class RotadorImagenes:

    def __init__(self, directorio_destino):
        # Inicializa la clase con el directorio donde se guardarán las imágenes rotadas
        self.directorio_destino = directorio_destino
        self.contador = 0

    def crear_directorio_destino(self):
        # Crea el directorio de destino si no existe
        if not os.path.exists(self.directorio_destino):
            os.makedirs(self.directorio_destino)

    def guardar_imagen(self, imagen, nombre_archivo):
        # Guarda la imagen en el directorio de destino
        directorio_rotadas = os.path.join(self.directorio_destino)
        if not os.path.exists(directorio_rotadas):
            os.makedirs(directorio_rotadas)

        cv2.imwrite(os.path.join(directorio_rotadas, nombre_archivo), imagen)

    def rotar_imagen(self, cara_recortada, angulo_rotacion, tipo_rotacion):
        # Aplica la rotación a la cara recortada
        matriz_rotacion = cv2.getRotationMatrix2D((cara_recortada.shape[1] / 2, cara_recortada.shape[0] / 2),
                                                  angulo_rotacion, 1)
        cara_rotada = cv2.warpAffine(cara_recortada, matriz_rotacion, (cara_recortada.shape[1], cara_recortada.shape[0]),
                                     borderMode=cv2.BORDER_REFLECT)

        # Redimensiona la cara rotada a 48x48
        cara_rotada_48x48 = cv2.resize(cara_rotada, (48, 48))

        # Guarda la cara rotada con un nombre descriptivo
        self.guardar_imagen(cara_rotada_48x48, f"{self.contador}_rotada_{tipo_rotacion}_{angulo_rotacion}.jpg")
        self.contador += 1

    def aplicar_rotacion_carpeta(self, ruta_carpeta):
        # Obtiene la lista de archivos en la carpeta
        archivos = os.listdir(ruta_carpeta)

        for archivo in archivos:
            if archivo.endswith(".jpg") or archivo.endswith(".png"):
                # Lee la imagen original y la redimensiona
                ruta_imagen = os.path.join(ruta_carpeta, archivo)
                imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                cara_recortada_48x48 = cv2.resize(imagen_original, (48, 48))

                # Guarda la imagen sin rotar
                self.guardar_imagen(cara_recortada_48x48, f"{self.contador}_sin_rotar.jpg")

                # Ángulos de rotación específicos
                angulos_rotacion = [30, -30]

                for tipo_rotacion, angulo_rotacion in enumerate(angulos_rotacion):
                    # Aplica la rotación a la cara recortada y la guarda
                    self.rotar_imagen(cara_recortada_48x48, angulo_rotacion, tipo_rotacion)

        cv2.destroyAllWindows()
