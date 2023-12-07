# este codigo no lo uso mas, es para anotar las imagenes en formato yolo para los embriones recortados
import os
import cv2

def normalize_coords(x, y, width, height, image_width, image_height):
    x_center = x / image_width
    y_center = y / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return x_center, y_center, norm_width, norm_height

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta_imagenes = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\Chapter03\02-exercices\data\images\val"
ruta_carpeta_labels = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\Chapter03\02-exercices\data\labels\val"
# Lista de las extensiones válidas de imágenes
extensiones_validas = [".jpg", ".png"]

# Recorre cada imagen en la carpeta
for filename in os.listdir(ruta_carpeta_imagenes):
    if filename.lower().endswith(tuple(extensiones_validas)):
        # Escribe los datos en el archivo .txt
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(ruta_carpeta_labels, txt_filename), "w") as f:
            f.write(f"0 0.5 0.5 0.95 0.95\n")
