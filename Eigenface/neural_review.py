import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from keras.models import load_model

# ----------------------------------------------------------------
# PASO 1: Cargar las imágenes de sportsimages
# ----------------------------------------------------------------

dirname = os.path.join(os.getcwd(), 'sportimages/sportimages')
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

    if prevRoot != root:
        print(root, cant)
        prevRoot = root
        directories.append(root)
        dircount.append(cant)
        cant = 0

dircount.append(cant)
dircount = dircount[1:]

print('Directorios leidos: ', len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:', sum(dircount))

# ----------------------------------------------------------------
# PASO 2: Crear etiquetas y arrays finales
# ----------------------------------------------------------------
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ", len(labels))

deportes=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    deportes.append(name[len(name)-1])
    indice=indice+1

y = np.array(labels)
X = np.array(images, dtype=np.uint8)

classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# ----------------------------------------------------------------
# PASO 3: Dividir y preprocesar los datos 
# ----------------------------------------------------------------
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
print('\nTraining data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Normalizar los datos de imagen
test_X = test_X.astype('float32')
test_X = test_X / 255.

# Convertir las etiquetas a formato one-hot encoding
test_Y_one_hot = to_categorical(test_Y)

# ----------------------------------------------------------------
# PASO 4: Cargar el modelo guardado y evaluar
# ----------------------------------------------------------------

print("\n" + "="*50)
print("CARGANDO MODELO GUARDADO")
print("="*50 + "\n")

# Cargar el modelo
sport_model = load_model("sports_mnist_2.h5")

print("Modelo cargado exitosamente!")
print("\nResumen del modelo:")
sport_model.summary()

# Evaluar el modelo con los datos de prueba
print("\n" + "="*50)
print("EVALUANDO MODELO")
print("="*50 + "\n")

test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print("\n" + "="*50)
print("RESULTADOS")
print("="*50)
print(f'Test loss: {test_eval[0]:.4f}')
print(f'Test accuracy: {test_eval[1]:.4f}')
print("="*50)

# ----------------------------------------------------------------
# PASO 5: Crear matriz de confusión
# ----------------------------------------------------------------

print("\n" + "="*50)
print("MATRIZ DE CONFUSIÓN")
print("="*50 + "\n")

# Realizar predicciones
predicted_classes_prob = sport_model.predict(test_X)
predicted_classes = np.argmax(predicted_classes_prob, axis=1)

# Crear matriz de confusión
cm = confusion_matrix(test_Y, predicted_classes)

# Visualizar matriz de confusión
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=deportes)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
plt.title('Matriz de Confusión - Clasificación de Deportes', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("\nMatriz de confusión (formato numérico):")
print(cm)