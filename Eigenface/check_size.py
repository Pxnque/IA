import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
import keras

# ----------------------------------------------------------------
# PASO 1: Cargar las imágenes de sportsimages
# ----------------------------------------------------------------

#dirname = os.path.join(os.getcwd(), 'sportimages/sportimages')
dirname = os.path.join(os.getcwd(), 'animals/PetImages')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("leyendo imagenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo... " + str(cant)
            print (b, end="\r")

    # --- Esta es la segunda parte de la imagen ---
    if prevRoot != root:
        print(root, cant)
        prevRoot = root
        directories.append(root)
        dircount.append(cant)
        cant = 0

dircount.append(cant)

dircount = dircount[1:]
#dircount[0] = dircount[0] + 1
print('Directorios leidos: ', len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:', sum(dircount))