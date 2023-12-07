import numpy as np
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from felzenszwalb_segmentation import segment
import urllib.request
import cv2
# Descargar la imagen de Lena desde Internet
# url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
# response = urllib.request.urlopen(url)
# imagen_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
# imagen = cv2.imdecode(imagen_array, -1)

# image_files = glob('./VOCdevkit/VOC2012/JPEGImages/*.jpg')
# image = np.array(Image.open(image_files[10]))

imagen = cv2.imread(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion\clase4\imagen_96 (2).png")
segmented_image = segment(imagen, 0.2, 400, 50)

fig = plt.figure(figsize=(12, 12))
a = fig.add_subplot(1, 2, 1)
plt.imshow(imagen)
a = fig.add_subplot(1, 2, 2)
plt.imshow(segmented_image.astype(np.uint8))
plt.show()