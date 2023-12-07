
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import os
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

TARGET_SIZE = (224, 224)
BATCH_SIZE = 5
CLASSES = ['clase0', 'clase1', 'clase2', 'clase3', 'clase4', 'clase5']
RANDOM_SEED = 0

train_data_dir = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion_train'  # Ruta al directorio donde guardarás las imágenes de entrenamiento
validation_data_dir = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion_val'    # Ruta al directorio donde guardarás las imágenes de validación

train_generator = ImageDataGenerator().flow_from_directory(train_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=10
                                                           , classes=CLASSES
                                                           , seed=0
                                                           , shuffle=True)

valid_generator = ImageDataGenerator().flow_from_directory(validation_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=5
                                                           , classes=CLASSES
                                                           , seed=0
                                                           , shuffle=True)

# Type your code here
base = ResNet50(weights='imagenet')
# base.summary()

# Type your code here
for layer in base.layers:
    layer.trainable = False

# Type your code here
sec_last_base = base.layers[-2].output
connected_model = Dense(len(CLASSES),activation='softmax')(sec_last_base)
base_input=base.input
model = Model(inputs = base_input, outputs = connected_model)

# Type your code here
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Type your code here
N_EPOCHES = 5
STEPS = train_generator.n // train_generator.batch_size
model.fit_generator(generator=train_generator,validation_data=valid_generator, steps_per_epoch=STEPS, epochs =N_EPOCHES)


train_history = model.history.history

# Type your code here
x =train_history['loss']
y = train_history['val_loss']
plt.plot(x)
plt.plot(y)

# Type your code here
x =train_history['accuracy']
y=train_history['val_accuracy']

plt.plot(x)
plt.plot(y)

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

model.save("resnet50_keras.h5")

