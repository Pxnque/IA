import os
import glob

# Ruta de la carpeta con las imágenes
carpeta = '/home/panque/repos/IA/Eigenface/animals/PetImagesx64/Mariquitax64/'

# Obtener todas las imágenes jpg en la carpeta
imagenes = sorted(glob.glob(os.path.join(carpeta, '*.jpg')))

print(f"Total de imágenes encontradas: {len(imagenes)}")

# Número de imágenes a eliminar
num_eliminar = 3351

if len(imagenes) < num_eliminar:
    print(f"Error: Solo hay {len(imagenes)} imágenes, no se pueden eliminar {num_eliminar}")
else:
    # Confirmar antes de eliminar
    respuesta = input(f"¿Estás seguro de eliminar {num_eliminar} imágenes? (si/no): ")
    
    if respuesta.lower() == 'si':
        # Eliminar las primeras 1731 imágenes (o puedes elegir las últimas)
        for i in range(num_eliminar):
            os.remove(imagenes[i])
            if (i + 1) % 100 == 0:  # Mostrar progreso cada 100 imágenes
                print(f"Eliminadas {i + 1} imágenes...")
        
        print(f"Se eliminaron {num_eliminar} imágenes exitosamente")
        print(f"Imágenes restantes: {len(imagenes) - num_eliminar}")
    else:
        print("Operación cancelada")