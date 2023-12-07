import os
import cv2

# Directorios de entrada y salida
input_images_folder = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\shuffled_data - copia\images'
input_annotations_folder = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\shuffled_data - copia\labels'

output_folder = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion'

# Crear las carpetas de salida para cada clase
for class_id in range(6):  # 0 a 5
    class_folder = os.path.join(output_folder, "clase" + str(class_id))
    os.makedirs(class_folder, exist_ok=True)

# Obtener la lista de nombres de archivos en la carpeta de imágenes
image_files = os.listdir(input_images_folder)

# Recorrer cada archivo de imagen
for image_file in image_files:
    image_name, _ = os.path.splitext(image_file)
    
    # Leer la imagen
    image_path = os.path.join(input_images_folder, image_file)
    image = cv2.imread(image_path)
    
    # Leer la anotación correspondiente
    annotation_file = image_name + '.txt'
    annotation_path = os.path.join(input_annotations_folder, annotation_file)
    
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    cont = 0
    # Procesar cada anotación en la imagen
    for annotation in annotations:
        class_id, x, y, w, h = map(float, annotation.strip().split())
        
        # Calcular coordenadas del rectángulo
        left = int((x - w / 2) * image.shape[1])
        top = int((y - h / 2) * image.shape[0])
        right = int((x + w / 2) * image.shape[1])
        bottom = int((y + h / 2) * image.shape[0])
        

        # Recortar la región de interés
        roi = image[top:bottom, left:right]
        
        # Crear el nombre del archivo de salida
        output_filename = f'{image_name}.png'
        output_path = os.path.join(output_folder, "clase"+str(int(class_id)), output_filename)
        print(output_path)
        # Mostrar la imagen recortada y pausar el programa hasta presionar una tecla
        # cv2.imshow('ROI', roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  #

        print(top, bottom, left, right)
        # Guardar la imagen recortada en la carpeta correspondiente a la clase
        try:
            cv2.imwrite(output_path, roi)
        except:
            cont+=1
            print("-----------------------------------------------------------", cont, "-------------------------------------------------")
