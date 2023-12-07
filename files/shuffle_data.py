import os
from sklearn.model_selection import train_test_split
import random
random.seed(108)

image_path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\data_embryos\images - copia"
label_path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\data_embryos\images - copia"
# Read images and annotations
images = [os.path.join(image_path, x) for x in os.listdir(image_path) if x[-3:] == "png"]
annotations = [os.path.join(label_path, x) for x in os.listdir(label_path) if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Combina las listas de imágenes y anotaciones
archivos_combinados = list(zip(images, annotations))
# Mezcla la lista combinada
random.shuffle(archivos_combinados)

for i, (imagen, anotacion) in enumerate(archivos_combinados):
    nuevo_nombre_imagen = os.path.join(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\data_embryos\suffled_images\images", f"imagen3_{i}.png")
    nuevo_nombre_anotacion = os.path.join(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\data_embryos\suffled_images\labels", f"imagen3_{i}.txt")

    os.rename(imagen, nuevo_nombre_imagen)
    os.rename(anotacion, nuevo_nombre_anotacion)


# image_val_path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\Chapter03\02-exercices\data\images\val"
# label_val_path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\Chapter03\02-exercices\data\labels\val"
# # Read images and annotations
# images_val = [os.path.join(image_val_path, x) for x in os.listdir(image_val_path)]
# annotations_val = [os.path.join(label_val_path, x) for x in os.listdir(label_val_path) if x[-3:] == "txt"]

# images_val.sort()
# annotations_val.sort()

# # Combina las listas de imágenes y anotaciones
# archivos_combinados = list(zip(images_val, annotations_val))
# # Mezcla la lista combinada
# random.shuffle(archivos_combinados)

# for i, (imagen, anotacion) in enumerate(archivos_combinados):
#     nuevo_nombre_imagen = os.path.join(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\Chapter03\02-exercices\shuffled_data\images\val", f"imagen_{i}.png")
#     nuevo_nombre_anotacion = os.path.join(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\Chapter03\02-exercices\shuffled_data\labels\val", f"imagen_{i}.txt")

#     os.rename(imagen, nuevo_nombre_imagen)
#     os.rename(anotacion, nuevo_nombre_anotacion)