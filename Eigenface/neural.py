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

dirname = os.path.join(os.getcwd(), 'emotions2/train')
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
    
    # --- Continuación del bucle (de la segunda imagen) ---
    deportes.append(name[len(name)-1])
    indice=indice+1

# --- Resto del código de la segunda imagen ---
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
# Después de: X = np.array(images, dtype=np.uint8)
print("Forma original de X:", X.shape)

# Manejar diferentes formatos de imagen
if len(X.shape) == 3:  # (n, 48, 48) - escala de grises
    X = np.expand_dims(X, axis=-1)
    print("Imágenes en escala de grises detectadas, agregando dimensión de canal")
elif X.shape[-1] == 3:  # (n, 48, 48, 3) - RGB
    print("Imágenes RGB detectadas")
else:
    print(f"Formato inesperado: {X.shape}")

print("Forma ajustada de X:", X.shape)


# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# ----------------------------------------------------------------
# PASO 3: Dividir y preprocesar los datos 
# ----------------------------------------------------------------
# Mezclar todo y crear los grupos de entrenamiento y testing
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
print('\nTraining data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Normalizar los datos de imagen (píxeles de 0-255 a 0-1)
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Convertir las etiquetas a formato one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Mostrar el cambio
print('\nOriginal label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# Crear un set de validación a partir del set de entrenamiento
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print('\nFinal shapes:')
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# ----------------------------------------------------------------
# PASO 4: Crear y entrenar el modelo
# ----------------------------------------------------------------

INIT_LR = 1e-3
epochs = 10
batch_size = 64

channels = X.shape[-1]  # 1 para escala de grises, 3 para RGB
img_height, img_width = X.shape[1], X.shape[2]

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', 
                      padding='same', input_shape=(img_height, img_width, channels)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))

sport_model.add(Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Conv2D(128, kernel_size=(3, 3), activation='linear', padding='same'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(128, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))

print(f"\nModelo configurado para imágenes de {img_height}x{img_width}x{channels}")
sport_model.summary()

sport_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adagrad(learning_rate=INIT_LR),
    metrics=['accuracy']
)
# entrenamiento
sport_train = sport_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
#guardamos el entrenamiento
sport_model.save("emotions.h5")
# evaluar la red
test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)# entrenamiento


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
sport_train.history

accuracy = sport_train.history['accuracy']
val_accuracy = sport_train.history['val_accuracy']
loss = sport_train.history['loss']
val_loss = sport_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes2 = sport_model.predict(test_X)
predicted_classes=[]
for predicted_sport in predicted_classes2:
    predicted_classes.append(predicted_sport.tolist().index(max(predicted_sport)))
predicted_classes=np.array(predicted_classes)
predicted_classes.shape, test_Y.shape

target_names = ["Class {}".format(i) for i in range(nClasses)]
#print(classification_report(test_Y, predicted_classes, target_names=target_names))