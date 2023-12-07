import torch
import cv2
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
import pytesseract

model = torch.hub.load(r'C:\Users\gasto\Documents\Python_Scripts\yolov5', "custom", path = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\modelos\best_embryo4.pt', source='local')
model_vgg = load_model(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\resnet50_keras.h5")

video = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\videos\ID 21288 (6).avi"
capture = cv2.VideoCapture(video)

frame_count = 0
frame_predictions = []

colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 255, 255),
          4: (255, 0, 255), 5: (255, 255, 0), 6: (255, 255, 255)}

clases = {0: "1 cell", 1: "1 clivaje", 2: "2 clivaje", 3: "3 clivaje",
          4: "blasto", 5: "bad embryo"}

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
time = []
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    text_roi = frame[650:, 1150:]
    text = pytesseract.image_to_string(text_roi)
    print(text[:-3])

    detect = model(frame)
    detections = detect.pred[0]   
    frame_detections = []
    if len(detections) > 0:  
        x, y, w, h = np.squeeze(detections[0][:4]).tolist()
        # print(x,y,w,h)
        # Calcular las coordenadas del cuadro de recorte en píxeles 
        image_height, image_width, _ = frame.shape
        x_pixel = int(x * image_width)
        y_pixel = int(y * image_height)
        w_pixel = int(w * image_width)
        h_pixel = int(h * image_height)

        # Recortar la imagen original usando las coordenadas calculadas
        roi = frame[int(y):int(h), int(x):int(w)]

        roi_resized = cv2.resize(roi, (224, 224))
        # roi_normalized = roi_resized / 255.0

        predicciones = model_vgg.predict(np.expand_dims(roi_resized, axis=0))
        clase_predicha = np.argmax(predicciones)
        print(predicciones)
        frame_detections.append((clase_predicha, frame_count, np.squeeze(predicciones)[clase_predicha]))
        
        # Obtener el nombre de la clase predicha
        print("clase_predicha", clase_predicha)
        nombre_clase = clases[clase_predicha]  # Asegúrate de tener una lista de nombres de clases
        
        # Dibujar el bounding box y el nombre de la clase en el frame original
        # cv2.putText(frame, nombre_clase, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('Cropped', roi)

    # Press q on keyboard to exit the program
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    frame_predictions.append(frame_detections)
    frame_count += 1
    time.append(float(text[:-3]))

    # cv2.imshow("detector", np.squeeze(detect.render()))
    cv2.rectangle(frame, (int(x),int(y)), (int(w),int(h)), colors[clase_predicha], 3)
    cv2.rectangle(frame, (int(x), int(h)), (int(x)+200,int(h)+35), colors[clase_predicha], -1)
    cv2.putText(frame, nombre_clase, (int(x)+1, int(h)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("detection", frame)
    t = cv2.waitKey(5)
    if t == 27:
        break

capture.release()
# cv2.destroyAllWindows()  

path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\archivos\clasificador"
filename = os.path.splitext(os.path.basename(video))[0]
os.makedirs(os.path.join(path, filename), exist_ok=True)

object_file_path = os.path.join(path, filename)

with open(object_file_path+f'/time.txt', 'w') as f:
    for t in time:
        f.write(f'{t}\n')

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

    # object_file_path = os.path.join(path, filename)

    with open(object_file_path+f'/{object_name}_probabilidades.txt', 'w') as f:
        for frame_num, prob in frame_probabilities.items():
            f.write(f'{frame_num} {prob:.4f}\n')

