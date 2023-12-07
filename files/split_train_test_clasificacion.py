import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion'  # Ruta al directorio principal de datos
class_names = os.listdir(data_dir)  # Obtiene la lista de nombres de clases

train_dir = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion_train'  # Ruta al directorio donde guardarás las imágenes de entrenamiento
valid_dir = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion_val'    # Ruta al directorio donde guardarás las imágenes de validación

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    image_files = os.listdir(class_dir)
    train_files, valid_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    for file in train_files:
        src = os.path.join(class_dir, file)
        dst = os.path.join(train_dir, class_name, file)
        shutil.copy(src, dst)
        
    for file in valid_files:
        src = os.path.join(class_dir, file)
        dst = os.path.join(valid_dir, class_name, file)
        shutil.copy(src, dst)
