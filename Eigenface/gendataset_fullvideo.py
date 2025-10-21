import cv2 as cv
import os

cap = cv.VideoCapture('/home/panque/Downloads/f11.mp4')
output_dir = '/home/panque/repos/IA/Eigenface/sportimages/sportimages/f1'
os.makedirs(output_dir, exist_ok=True)

i = 0

while True:
    ret, frame = cap.read()
    
    # Verificar si se leyó correctamente el fotograma
    if not ret:
        print(f"Fin del video o error al leer. Total de fotogramas procesados: {i}")
        break
    
    frame_resized = cv.resize(frame, (28, 21), interpolation=cv.INTER_AREA)
    filename = os.path.join(output_dir, f'f11_{i}.jpg')  
    cv.imwrite(filename, frame_resized)
    
    
    
    i += 1
    
    # Presionar ESC para salir
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

print(f"Se guardaron {i} imágenes en {output_dir}")