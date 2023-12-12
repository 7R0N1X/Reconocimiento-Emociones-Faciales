import cv2
import os


class RotadorImagenes:
    def __init__(self, directorio_destino):
        """
        :param directorio_destino: Directorio donde se guardarán las imágenes rotadas.
        """
        self.directorio_destino = directorio_destino
        self.contador = 0

    def crear_directorio_destino(self):
        """
        Crea el directorio de destino si no existe.
        """
        if not os.path.exists(self.directorio_destino):
            os.makedirs(self.directorio_destino)

    def guardar_imagen(self, imagen, nombre_archivo):
        """
        :param imagen: Imagen a guardar.
        :param nombre_archivo: Nombre del archivo para la imagen.
        """
        directorio_rotadas = os.path.join(self.directorio_destino)
        cv2.imwrite(os.path.join(directorio_rotadas, nombre_archivo), imagen, [cv2.IMWRITE_JPEG_QUALITY, 100])

    @staticmethod
    def recortar_cara(imagen):
        """
        Recorta la región de la cara en la imagen en 48x48 píxeles.
        :param imagen: Imagen de entrada.
        """
        # Carga el clasificador Haar para detección de rostros
        cascade_clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convierte la imagen de color BGR a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        """
        Utiliza el clasificador Haar para detectar caras en la imagen en escala de grises
        detectMultiScale devuelve las coordenadas (x, y, ancho, alto) de las caras detectadas
        scaleFactor controla la compensación de escala, minNeighbors establece la detección mínima de vecinos y minSize
        define el tamaño mínimo de la cara
        """
        caras = cascade_clasificador.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in caras:
            cara_recortada = imagen_gris[y:y+h, x:x+w]  # Región de la cara en escala de grises
            cara_recortada_48x48 = cv2.resize(cara_recortada, (48, 48))  # Redimensionar la cara a 48x48
            return cara_recortada_48x48

        return None  # Si no se detecta ninguna cara en la imagen

    def rotar_imagen(self, cara_recortada, angulo_rotacion, tipo_rotacion):
        """
        Aplica rotación a la imagen de la cara recortada.
        :param cara_recortada: Imagen de la cara recortada.
        :param angulo_rotacion: Ángulo de rotación.
        :param tipo_rotacion: Tipo de rotación 'rotada, sin_rotar'.
        :return:
        """
        if cara_recortada is not None:
            # Obtiene la matriz de rotación para la cara recortada
            matriz_rotacion = cv2.getRotationMatrix2D((cara_recortada.shape[1] / 2, cara_recortada.shape[0] / 2),
                                                      angulo_rotacion, 1)
            # Aplica la transformación de rotación a la cara recortada
            cara_rotada = cv2.warpAffine(cara_recortada, matriz_rotacion,
                                         (cara_recortada.shape[1], cara_recortada.shape[0]),
                                         borderMode=cv2.BORDER_REFLECT)

            # Guarda la imagen de la cara rotada con un nombre de archivo único y aumenta el contador
            self.guardar_imagen(cara_rotada, f"{self.contador}_rotada_{tipo_rotacion}_{angulo_rotacion}.jpg")
            self.contador += 1

    def aplicar_rotacion_carpeta(self, ruta_carpeta):
        """
        Aplica rotación a las imágenes en una carpeta.
        :param ruta_carpeta: Ruta de la carpeta con las imágenes.
        """
        # Obtiene la lista de archivos
        archivos = os.listdir(ruta_carpeta)

        # Itera sobre cada archivo en la lista
        for archivo in archivos:

            # Verifica si el archivo es una imagen (formatos .jpg o .png)
            if archivo.endswith(".jpg") or archivo.endswith(".png"):

                # Obtiene la ruta completa de la imagen
                ruta_imagen = os.path.join(ruta_carpeta, archivo)

                # Lee la imagen
                imagen_original = cv2.imread(ruta_imagen)

                # Recorta la cara en la imagen original
                cara_recortada_48x48 = self.recortar_cara(imagen_original)

                # Verifica si se detectó correctamente una cara en la imagen
                if cara_recortada_48x48 is not None:
                    # Guarda la imagen de la cara recortada
                    self.guardar_imagen(cara_recortada_48x48, f"{self.contador}_cara_48x48.jpg")
                    self.contador += 1

                    # Lista de ángulos de rotación para la cara recortada
                    angulos_rotacion = [30, -30]

                    # Itera sobre los ángulos de rotación para aplicar la rotación
                    for tipo_rotacion, angulo_rotacion in enumerate(angulos_rotacion):
                        # Aplica la rotación a la cara recortada en cada ángulo
                        self.rotar_imagen(cara_recortada_48x48, angulo_rotacion, tipo_rotacion)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directorio donde se guardarán las imágenes rotadas
    directorio_destino = "E:\\Reconocimiento-Emociones-Faciales\\creacion-dataset\\caras-rotadas"

    # Inicializa el objeto RotadorImagenes con el directorio de destino
    rotador = RotadorImagenes(directorio_destino)

    # Crea el directorio de destino si no existe
    rotador.crear_directorio_destino()

    # Ruta de la carpeta que contiene las imágenes originales
    ruta_carpeta = "E:\\img"

    # Aplica la rotación a las imágenes en la carpeta especificada
    rotador.aplicar_rotacion_carpeta(ruta_carpeta)
