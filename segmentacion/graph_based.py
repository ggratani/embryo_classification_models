import cv2
import networkx as nx
import numpy as np
import urllib.request

# Descargar la imagen de Lena desde Internet
url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
response = urllib.request.urlopen(url)
imagen_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
imagen = cv2.imdecode(imagen_array, -1)
cv2.imshow('Imagen', imagen)
cv2.waitKey(0)
# Convierte la imagen a escala de grises si no lo está ya
if len(imagen.shape) == 3:
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

cv2.imshow('Imagen', imagen)
cv2.waitKey(0)

imagen = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
# Crear un grafo
G = nx.Graph()

# Obtener el alto y ancho de la imagen
alto, ancho = imagen.shape

# Agregar nodos para cada píxel en la imagen
for y in range(alto):
    for x in range(ancho):
        intensidad_pixel = imagen[y, x]
        G.add_node((x, y), pixel_value=intensidad_pixel)

# Agregar aristas basadas en la diferencia de intensidad entre píxeles vecinos
for y in range(alto):
    for x in range(ancho):
        if x < ancho - 1:
            # print("valores",imagen[y, x],imagen[y, x + 1])
            # print("resta", imagen[y, x] - imagen[y, x + 1])
            peso = abs(int(imagen[y, x]) - int(imagen[y, x + 1]))
            # print("peso",peso)
            G.add_edge((x, y), (x + 1, y), weight=peso)
        if y < alto - 1:
            # print("valores",imagen[y, x], imagen[y + 1, x])
            peso = abs(int(imagen[y, x]) - int(imagen[y + 1, x]))
            # print("peso",peso)
            G.add_edge((x, y), (x, y + 1), weight=peso)

# Definir los nodos fuente y sumidero
nodo_fuente = (0, 0)
nodo_sumidero = (ancho - 1, alto - 1)

# Encontrar el corte mínimo utilizando NetworkX
valor_corte, particion = nx.minimum_cut(G, _s=nodo_fuente, _t=nodo_sumidero)

# Crear una máscara de segmentación
mascara_segmentacion = np.zeros_like(imagen, dtype=np.uint8)
for pixel in particion[0]:
    x, y = pixel
    mascara_segmentacion[y, x] = 255

# Mostrar la imagen segmentada
cv2.imshow('Imagen Segmentada', mascara_segmentacion)
cv2.waitKey(0)
cv2.destroyAllWindows()
