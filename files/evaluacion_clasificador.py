from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import os
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image
import numpy as np 

CLASSES = ['clase0', 'clase1', 'clase2', 'clase3', 'clase4', 'clase5']

model_vgg = load_model(r"C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\resnet50_keras.h5")

print("=================================== VGG16 ============================================")
print("resnet50:\n", model_vgg.summary())

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize = 16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import random

test_data_dir = r'C:\Users\gasto\Documents\Python_Scripts\Mastering-OpenCV-4-with-Python\proyectos\files\imagenes_clasificacion_val'
test_valid_generator = ImageDataGenerator().flow_from_directory(test_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=5
                                                           , classes=CLASSES
                                                           , seed=0
                                                           , shuffle=False)

predictions = model_vgg.predict(test_valid_generator)
predictions.argmax(axis=1)
labels = {'0':'clase0', '1':'clase1', '2':'clase2', '3':'clase3', '4':'clase4', '5':'clase5'}
for i in range(0,len(predictions)):
    print(labels[str(predictions.argmax(axis=1)[i])])


test_labels = test_valid_generator.classes
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cmastype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i,j]> thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')


cm_plot_labels = ['clase0', 'clase1', 'clase2', 'clase3', 'clase4', 'clase5']
plot_confusion_matrix(cm, cm_plot_labels, title= "Confusion Matrix")