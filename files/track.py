"""
Example to introduce how to read a camera connected to your computer and save frame
"""

# Import the required packages
import cv2
import os
import dlib
import get_points

def normalize_coords(x, y, width, height, image_width, image_height):
    x_center = x / image_width
    y_center = y / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return x_center, y_center, norm_width, norm_height

# La mayoria de los videos ID 30 son muy fragmentados
file = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\videos\ID 21288 (6).avi"
# file = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\videos\ID 21253 (10).avi"

clase = 4
embryo_number = 28

filename = os.path.splitext(os.path.basename(file))[0]
print(filename)

capture = cv2.VideoCapture(file)
print('Pulsa P ')
# We create a VideoCapture object to read from the camera (pass 0):

# Get some properties of VideoCapture (frame width, frame height and frames per second (fps)):
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# Print these values:
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(frame_height))
print("CAP_PROP_FPS : '{}'".format(fps))

# Check if camera opened successfully
if capture.isOpened() is False:
    print("Error opening the camera")

# Index to save current frame
frame_index = 0

# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame from the camera
    ret, frame = capture.read()

    # Si oprimimos P cerramos
    t = cv2.waitKey(1)
    if (t == ord('p')):
        # Las coordenadas de los objetos a rastrear
        # se almacenarán en una lista llamada `puntos`
        points = get_points.run(frame)
        if not points:
            print("ERROR: No objeto para seguimiento.")
            exit()
        if points:
            tracker = dlib.correlation_tracker()
            # Proporcionar al rastreador la posición inicial del objeto.
            tracker.start_track(frame, dlib.rectangle(*points[0]))
            while True:
                # Se lee la imagen desde el aparato o archivo
                ret, frame = capture.read()
                frame2 = frame.copy()
                if not ret:
                    exit()
                # Actualizacion del seguimiento
                tracker.update(frame)
                # Se obtiene la posición del objeto, se dibujar un
                # cuadro de límite alrededor de él y lo muestra.
                rect = tracker.get_position()
                pt1 = (int(rect.left()), int(rect.top()))
                pt2 = (int(rect.right()), int(rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
                print("Objecto tracked en [{}, {}] \r".format(pt1, pt2), )
                # loc = (int(rect.left()), int(rect.top() - 20))
                # txt = "Objecto tracked en [{}, {}]".format(pt1, pt2)
                # cv2.putText(frame, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

                cv2.imshow('Input frame from the camera', frame)
                ruta_carpeta = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\data_embryos\images"
                
                embryo_frame_name = r"\embryo_{}_frame_{}".format(embryo_number, frame_index)
                path = ruta_carpeta + embryo_frame_name + ".png"
                cv2.imwrite(path, frame2)

                x = (pt1[0] + pt2[0])/2
                y = (pt1[1] + pt2[1])/2

                width = pt2[0]-pt1[0]
                height = pt2[1]-pt1[1]

                x_center, y_center, norm_width, norm_height = normalize_coords(x, y, width, height, frame_width, frame_height)
                
                # Escribe los datos en el archivo .txt
                txt_filename = embryo_frame_name + ".txt"
                with open(ruta_carpeta + txt_filename, "w") as f:
                    f.write(f"{clase} {x_center} {y_center} {norm_width} {norm_height}\n".format(clase, x_center, y_center, norm_width, norm_height))

                frame_index += 1
                # Continua hasta que se pulsa Escape
                if cv2.waitKey(1) == 27:
                    break
        break

    # Mostramos los frame
    cv2.imshow("IMAGEN", frame)


# Release everything:
capture.release()
cv2.destroyAllWindows()
