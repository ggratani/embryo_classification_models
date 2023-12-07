import torch
import cv2
import numpy as np
import os

model = torch.hub.load(r'C:\Users\gasto\Documents\Python_Scripts\yolov5', "custom", path = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\modelos\best_embryo5.pt', source='local')

video = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\videos\ID 30 (1.1).avi"
capture = cv2.VideoCapture(video)

frame_count = 0
frame_predictions = []

while True:
    ret, frame = capture.read()
    if not ret:
        break

    detect = model(frame)
    detections = detect.pred[0]    
    # Registrar las detecciones en este frame
    frame_detections = []
    for detection in detections:
        class_id = int(detection[5])
        # class_name = model.module.names[class_id]  # Obtener el nombre de la clase
        probability = detection[4].item()  # Obtener la probabilidad
        frame_detections.append((class_id, frame_count, probability))
    
    frame_predictions.append(frame_detections)
    frame_count += 1
    
    cv2.imshow("detector", np.squeeze(detect.render()))
    t = cv2.waitKey(5)
    if t == 27:
        break

capture.release()
# cv2.destroyAllWindows()  

path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\archivos"
filename = os.path.splitext(os.path.basename(video))[0]
os.makedirs(os.path.join(path, filename), exist_ok=True)

# Crear archivos de texto separados por objeto
for i, object_name in enumerate(['1cell', '1clivaje', '2clivaje', '3clivaje', 'blasto']):
    object_detections = [detections for detections in frame_predictions if any(det[0] == i for det in detections)]
    # print(object_detections)
    # Crear un diccionario para almacenar las probabilidades por frame
    frame_probabilities = {j: 0.0 for j in range(len(frame_predictions))}
    
    for detections in object_detections:
        for det in detections:
            if det[0] == i:
                frame_probabilities[det[1]] = det[2]

    object_file_path = os.path.join(path, filename)

    with open(object_file_path+f'/{object_name}_probabilidades.txt', 'w') as f:
        for frame_num, prob in frame_probabilities.items():
            f.write(f'{frame_num} {prob:.4f}\n')

# # Obtener las coordenadas de la detección (x, y, w, h) del primer objeto detectado (índice 0)
    # try:
    #     x, y, w, h, _, _ = np.squeeze(detect.pred[0]).tolist()
    #     # print(x,y,w,h)
    #     # Calcular las coordenadas del cuadro de recorte en píxeles (necesitas multiplicar por el tamaño original de la imagen)
    #     image_height, image_width, _ = frame.shape
    #     x_pixel = int(x * image_width)
    #     y_pixel = int(y * image_height)
    #     w_pixel = int(w * image_width)
    #     h_pixel = int(h * image_height)

    #     # Recortar la imagen original usando las coordenadas calculadas
    #     cropped_image = frame[int(y):int(y + h), int(x):int(x + w)]
    #     # print(cropped_image)
    #     cv2.imshow('Cropped', cropped_image)
    #     # print("llega aca")

    # except:
    #     pass