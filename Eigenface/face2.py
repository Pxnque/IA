import numpy as np
import cv2 as cv
import os

# SOLUCIÓN 1: Usar la ruta completa de OpenCV
rostro = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# SOLUCIÓN 2 (alternativa): Verificar si el archivo existe localmente
# Si tienes el archivo en tu directorio, verifica la ruta:
# if not os.path.exists('haarcascade_frontalface_alt.xml'):
#     print("Error: No se encuentra el archivo XML")
#     exit()
# rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Verificar que el clasificador se cargó correctamente
if rostro.empty():
    print("Error: No se pudo cargar el clasificador Haar Cascade")
    exit()

cap = cv.VideoCapture(0)

# Verificar que la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

x = y = w = h = 0
img = None
count = 0

print("Presiona ESC para salir")

while True:
    ret, frame = cap.read()
    
    # Verificar que se leyó correctamente el frame
    if not ret:
        print("Error: No se puede recibir frame de la cámara")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in rostros:
        # Dibujar rectángulo alrededor del rostro
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Recortar la región del rostro
        img = frame[y:y+h, x:x+w]
        count = count + 1
    
    # Mostrar el frame con los rostros detectados
    cv.imshow('Rostros Detectados', frame)
    
    # Mostrar el rostro recortado solo si se detectó al menos uno
    if img is not None and img.size > 0:
        cv.imshow('Cara Recortada', img)
    
    # Esperar por la tecla ESC (código 27)
    k = cv.waitKey(1)
    if k == 27:
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()

print(f"Total de frames con rostros detectados: {count}")