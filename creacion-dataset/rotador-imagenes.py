import cv2
import os


def crear_directorio_salida(ruta_salida):
    """
    :param ruta_salida: La ruta al directorio de salida que se creará si no existe.
    """
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)


def guardar_imagen(imagen, nombre_archivo, ruta_salida):
    """
    :param imagen: La imagen a guardar.
    :param nombre_archivo: El nombre del archivo de imagen.
    :param ruta_salida: La ruta al directorio de salida donde se guardará la imagen.
    """
    cv2.imwrite(os.path.join(ruta_salida, nombre_archivo), imagen, [cv2.IMWRITE_JPEG_QUALITY, 100])


def rotar_imagen(imagen, angulo_rotacion):
    """
    :param imagen: La imagen a rotar.
    :param angulo_rotacion: El ángulo de rotación.
    :return: La imagen rotada.
    """
    altura, ancho = imagen.shape[:2]
    matriz_rotacion = cv2.getRotationMatrix2D((ancho / 2, altura / 2), angulo_rotacion, 1)
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, altura), borderMode=cv2.BORDER_REFLECT)

    # Aplicar volteo horizontal a la imagen rotada
    imagen_volteada = cv2.flip(imagen_rotada, 1)  # 1 indica volteo horizontal, 0 sería vertical

    return imagen_volteada


def aplicar_rotacion_carpeta(ruta_carpeta_entrada, ruta_carpeta_salida):
    """
    :param ruta_carpeta_entrada: La ruta a la carpeta que contiene las imágenes originales.
    :param ruta_carpeta_salida: La ruta a la carpeta donde se guardarán las imágenes rotadas.
    """
    contador = 0
    crear_directorio_salida(ruta_carpeta_salida)

    for archivo in os.listdir(ruta_carpeta_entrada):
        if archivo.lower().endswith((".jpg", ".png")):
            ruta_imagen = os.path.join(ruta_carpeta_entrada, archivo)
            imagen_original = cv2.imread(ruta_imagen)
            if imagen_original is not None:
                # guardar_imagen(imagen_original, f"{contador}_original.jpg", ruta_carpeta_salida)
                contador += 1

                for angulo in [30, -30]:
                    imagen_rotada = rotar_imagen(imagen_original, angulo)
                    guardar_imagen(imagen_rotada, f"{contador}_rotada_{angulo}.jpg", ruta_carpeta_salida)
                    contador += 1

    cv2.destroyAllWindows()


ruta_salida = 'caras-rotadas'
ruta_entrada = ''
aplicar_rotacion_carpeta(ruta_entrada, ruta_salida)
