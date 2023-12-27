import cv2
import os


class RotadorImagenes:

    def __init__(self, directorio_destino):
        self.directorio_destino = directorio_destino  # Directorio donde se guardarán las imágenes rotadas
        self.contador = 0  # Contador para nombres de archivos únicos

    def crear_directorio_destino(self):
        if not os.path.exists(self.directorio_destino):  # Verifica si el directorio de destino existe
            os.makedirs(self.directorio_destino)  # Crea el directorio si no existe

    def guardar_imagen(self, imagen, nombre_archivo):
        directorio_rotadas = os.path.join(self.directorio_destino)  # Directorio para las imágenes rotadas
        # Guarda la imagen en el directorio con el nombre proporcionado y calidad JPEG al 100%
        cv2.imwrite(os.path.join(directorio_rotadas, nombre_archivo), imagen, [cv2.IMWRITE_JPEG_QUALITY, 100])

    @staticmethod
    def rotar_imagen(imagen, angulo_rotacion):
        # Obtiene la matriz de rotación para rotar la imagen según el ángulo dado
        matriz_rotacion = cv2.getRotationMatrix2D((imagen.shape[1] / 2, imagen.shape[0] / 2), angulo_rotacion, 1)
        # Aplica la transformación de rotación a la imagen
        imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (imagen.shape[1], imagen.shape[0]),
                                       borderMode=cv2.BORDER_REFLECT)
        return imagen_rotada

    def aplicar_rotacion_carpeta(self, ruta_carpeta):
        archivos = os.listdir(ruta_carpeta)  # Lista los archivos en la carpeta proporcionada

        for archivo in archivos:
            if archivo.endswith(".jpg") or archivo.endswith(".png"):  # Verifica si el archivo es una imagen
                ruta_imagen = os.path.join(ruta_carpeta, archivo)  # Ruta completa del archivo de imagen
                imagen_original = cv2.imread(ruta_imagen)  # Lee la imagen original
                self.guardar_imagen(imagen_original, f"{self.contador}_original.jpg")  # Guarda la imagen original
                self.contador += 1  # Incrementa el contador para generar nombres únicos
                for angulo_rotacion in [30, -30]:  # Itera sobre los ángulos de rotación
                    imagen_rotada = self.rotar_imagen(imagen_original, angulo_rotacion)  # Rota la imagen
                    # Guarda la imagen rotada con un nombre único
                    self.guardar_imagen(imagen_rotada, f"{self.contador}_rotada_{angulo_rotacion}.jpg")
                    self.contador += 1  # Incrementa el contador para generar nombres únicos

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directorio donde se guardarán las imágenes rotadas
    directorio_destino = "E:\\Reconocimiento-Emociones-Faciales\\creacion-dataset\\caras-rotadas"
    rotador = RotadorImagenes(directorio_destino)  # Inicializa un objeto RotadorImagenes
    rotador.crear_directorio_destino()  # Crea el directorio de destino si no existe
    ruta_carpeta = "E:\\img"  # Ruta de la carpeta que contiene las imágenes originales
    rotador.aplicar_rotacion_carpeta(ruta_carpeta)  # Aplica la rotación a las imágenes en la carpeta
