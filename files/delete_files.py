#Este codigo es para quedarme solamente con las imagenes que tienen anotaciones

import os
import shutil

# Directorio de la carpeta con imágenes y archivos de texto
carpeta = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\data_embryos\images'

# Obtener la lista de todos los archivos en la carpeta
archivos = os.listdir(carpeta)

# Crear una estructura de datos para almacenar los pares de archivos
pares = {}

# Separar los archivos en pares según el nombre (sin la extensión)
for archivo in archivos:
    nombre_base, extension = os.path.splitext(archivo)
    if extension.lower() == '.png':
        pares.setdefault(nombre_base, {})[extension] = archivo
    elif extension.lower() == '.txt':
        pares.setdefault(nombre_base, {})[extension] = archivo

# Crear una carpeta para los pares separados si no existe
carpeta_destino = os.path.join(carpeta, 'archivos_separados')
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Mover los pares de archivos a la carpeta de destino
for nombre_base, archivos in pares.items():
    if '.png' in archivos and '.txt' in archivos:
        ruta_imagen = os.path.join(carpeta, archivos['.png'])
        ruta_txt = os.path.join(carpeta, archivos['.txt'])
        ruta_destino_imagen = os.path.join(carpeta_destino, archivos['.png'])
        ruta_destino_txt = os.path.join(carpeta_destino, archivos['.txt'])
        
        shutil.move(ruta_imagen, ruta_destino_imagen)
        shutil.move(ruta_txt, ruta_destino_txt)

print("Archivos de imágenes y archivos de texto separados correctamente.")
