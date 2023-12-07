import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo de texto y extraer los datos
# Lista de nombres de archivos
path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\archivos\clasificador\ID 21288 (6)"
# path = r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\archivos\ID 17861 (1)"

nombres_archivos = ['1cell_probabilidades.txt', '1clivaje_probabilidades.txt', '2clivaje_probabilidades.txt', '3clivaje_probabilidades.txt', 'blasto_probabilidades.txt']

# Colores para los gráficos de cada archivo
colores = ['b', 'g', 'r', 'c', 'm']

plt.figure(figsize=(10, 6))
time = np.loadtxt(path+ "/time.txt")
# Iterar sobre los archivos y sus colores correspondientes
for i, archivo in enumerate(nombres_archivos):
    datos = np.loadtxt(path+ "/"+ archivo)
    numeros_frame = datos[:, 0]
    probabilidades = datos[:, 1]
    plt.plot(time, probabilidades, linestyle='-', color=colores[i], label=archivo.split("_")[0])

    # indices = np.where(probabilidades > 0.6)
    # numeros_frame_filtrados = numeros_frame[indices]
    # probabilidades_filtradas = probabilidades[indices]
    
    # plt.plot(numeros_frame_filtrados, probabilidades_filtradas, linestyle='-', color=colores[i], label=archivo)

plt.title('Gráfico de Probabilidad vs. Tiempo')
plt.xlabel('Tiempo (h)')
plt.ylabel('Probabilidad')
plt.grid(True)
plt.legend()
plt.show()