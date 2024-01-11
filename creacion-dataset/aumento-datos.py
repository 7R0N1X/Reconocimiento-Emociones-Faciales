import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def aplicar_aumento_datos(carpeta_entrada, carpeta_salida):
    generador_datos = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.02,
        height_shift_range=0.01,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True
    )

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Itera sobre cada imagen en la carpeta de entrada
    for nombre_archivo in os.listdir(carpeta_entrada):
        # Lee la imagen
        img = load_img(os.path.join(carpeta_entrada, nombre_archivo))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Genera imágenes transformadas
        i = 0
        for lote in generador_datos.flow(x, batch_size=1, save_to_dir=carpeta_salida, save_prefix='img',
                                         save_format='jpg'):
            i += 1
            if i >= 3:
                break


if __name__ == "__main__":
    carpeta_entrada = ''
    carpeta_salida = ''

    aplicar_aumento_datos(carpeta_entrada, carpeta_salida)
