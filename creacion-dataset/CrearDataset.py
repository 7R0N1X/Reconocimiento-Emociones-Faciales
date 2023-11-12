import os
from ProcesadorRostrosDesdeVideo import ProcesadorRostrosDesdeVideo
from RotadorImagenes import RotadorImagenes


class CrearDataset:

    def __init__(self, ruta_video, ruta_destino, nombre_carpeta_rotada, nombre_carpeta_faces):
        """
        Inicializa la clase CrearDataset.

        Parámetros:
        - ruta_video (str): La ruta del archivo de video.
        - ruta_destino (str): La ruta del directorio donde se guardará el conjunto de datos.
        - nombre_carpeta_rotada (str): El nombre de la carpeta donde se guardarán las imágenes rotadas.
        - nombre_carpeta_faces (str): El nombre de la carpeta donde se guardarán las imágenes de rostros.
        """
        self.ruta_video = ruta_video
        self.ruta_destino = ruta_destino
        self.nombre_carpeta_rotada = nombre_carpeta_rotada
        self.nombre_carpeta_faces = nombre_carpeta_faces

    def crear_dataset(self):
        """
        Crea el conjunto de datos combinando imágenes rotadas y rostros detectados desde un video.
        """
        # Inicializa el procesador de rostros desde video
        procesador_rostros = ProcesadorRostrosDesdeVideo(self.ruta_video,
                                                         os.path.join(self.ruta_destino, self.nombre_carpeta_faces))
        procesador_rostros.crear_directorio_dataset()

        # Procesa el video y guarda las caras recortadas
        procesador_rostros.procesar_video()

        # Inicializa el rotador de imágenes
        rotador = RotadorImagenes(os.path.join(self.ruta_destino, self.nombre_carpeta_rotada))
        rotador.crear_directorio_destino()

        # Aplica rotación a las imágenes en la carpeta especificada
        rotador.aplicar_rotacion_carpeta('./caras_detectadas')


if __name__ == "__main__":
    # Define las rutas y nombres de carpetas
    ruta_video = 0  # Reemplaza con 0 para usar la cámara
    ruta_destino = '.'  # Puedes cambiar '.' a la ruta deseada
    nombre_carpeta_rotada = "caras_rotadas"
    nombre_carpeta_faces = "caras_detectadas"

    # Crea una instancia de la clase CrearDataset y ejecuta la creación del conjunto de datos
    creador_dataset = CrearDataset(ruta_video, ruta_destino, nombre_carpeta_rotada, nombre_carpeta_faces)
    creador_dataset.crear_dataset()
