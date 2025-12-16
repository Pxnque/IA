"""
Entrenador de Clasificador en Cascada para Detección de Objetos
===============================================================
Este script prepara los datos y entrena un clasificador cascade (.xml)
para detectar objetos personalizados usando OpenCV.

Estructura de carpetas requerida:
- dataSet/
  - positivas/    <- Imágenes que CONTIENEN el objeto
  - negativas/    <- Imágenes que NO contienen el objeto (fondos variados)
"""

import cv2 as cv
import numpy as np
import os
import subprocess
import shutil

class CascadeTrainer:
    def __init__(self, dataset_path, output_name="objeto_detector"):
        self.dataset_path = dataset_path
        self.output_name = output_name
        self.pos_path = os.path.join(dataset_path, "positivas")
        self.neg_path = os.path.join(dataset_path, "negativas")
        self.work_dir = os.path.join(dataset_path, "cascade_training")
        
    def preparar_directorios(self):
        """Crea los directorios necesarios para el entrenamiento"""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "cascade"), exist_ok=True)
        print("✓ Directorios creados")
        
    def crear_archivo_negativos(self):
        """Crea el archivo bg.txt con las rutas de imágenes negativas"""
        neg_file = os.path.join(self.work_dir, "bg.txt")
        
        with open(neg_file, 'w') as f:
            for img_name in os.listdir(self.neg_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(self.neg_path, img_name)
                    f.write(f"{img_path}\n")
        
        num_neg = len(open(neg_file).readlines())
        print(f"✓ Archivo de negativos creado: {num_neg} imágenes")
        return neg_file, num_neg
    
    def crear_archivo_positivos(self, ancho=24, alto=24):
        """Crea el archivo positives.txt con anotaciones de las imágenes positivas"""
        pos_file = os.path.join(self.work_dir, "positives.txt")
        
        with open(pos_file, 'w') as f:
            for img_name in os.listdir(self.pos_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(self.pos_path, img_name)
                    img = cv.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        # Formato: ruta num_objetos x y ancho alto
                        f.write(f"{img_path} 1 0 0 {w} {h}\n")
        
        num_pos = len(open(pos_file).readlines())
        print(f"✓ Archivo de positivos creado: {num_pos} imágenes")
        return pos_file, num_pos
    
    def redimensionar_imagenes(self, ancho=100, alto=100):
        """Redimensiona todas las imágenes positivas a un tamaño uniforme"""
        print("Redimensionando imágenes positivas...")
        
        for img_name in os.listdir(self.pos_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(self.pos_path, img_name)
                img = cv.imread(img_path)
                if img is not None:
                    img_resized = cv.resize(img, (ancho, alto))
                    cv.imwrite(img_path, img_resized)
        
        print(f"✓ Imágenes redimensionadas a {ancho}x{alto}")
    
    def crear_samples_vec(self, num_pos, ancho=24, alto=24):
        """Crea el archivo .vec con las muestras positivas"""
        pos_file = os.path.join(self.work_dir, "positives.txt")
        vec_file = os.path.join(self.work_dir, "positives.vec")
        
        cmd = [
            "opencv_createsamples",
            "-info", pos_file,
            "-num", str(num_pos),
            "-w", str(ancho),
            "-h", str(alto),
            "-vec", vec_file
        ]
        
        print(f"Ejecutando: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Archivo .vec creado: {vec_file}")
            return vec_file
        except FileNotFoundError:
            print("⚠ opencv_createsamples no encontrado. Instalando opencv-contrib...")
            print("  Ejecuta: sudo apt-get install opencv-data")
            return None
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al crear samples: {e}")
            return None
    
    def entrenar_cascade(self, num_pos, num_neg, ancho=24, alto=24, num_stages=10):
        """Entrena el clasificador en cascada"""
        vec_file = os.path.join(self.work_dir, "positives.vec")
        neg_file = os.path.join(self.work_dir, "bg.txt")
        cascade_dir = os.path.join(self.work_dir, "cascade")
        
        # Usar menos muestras positivas para evitar errores
        num_pos_train = int(num_pos * 0.9)
        
        cmd = [
            "opencv_traincascade",
            "-data", cascade_dir,
            "-vec", vec_file,
            "-bg", neg_file,
            "-numPos", str(num_pos_train),
            "-numNeg", str(num_neg),
            "-numStages", str(num_stages),
            "-w", str(ancho),
            "-h", str(alto),
            "-featureType", "LBP",  # LBP es más rápido que HAAR
            "-minHitRate", "0.995",
            "-maxFalseAlarmRate", "0.5"
        ]
        
        print(f"\nIniciando entrenamiento con {num_stages} etapas...")
        print(f"Positivos: {num_pos_train}, Negativos: {num_neg}")
        print("Esto puede tomar varios minutos u horas dependiendo de los datos...\n")
        
        try:
            subprocess.run(cmd, check=True)
            
            # Copiar el archivo final
            src = os.path.join(cascade_dir, "cascade.xml")
            dst = f"{self.output_name}.xml"
            
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"\n✓ ¡Entrenamiento completado!")
                print(f"✓ Clasificador guardado en: {dst}")
                return dst
            else:
                print("✗ No se generó el archivo cascade.xml")
                return None
                
        except FileNotFoundError:
            print("⚠ opencv_traincascade no encontrado.")
            print("  Ejecuta: sudo apt-get install opencv-data")
            return None
        except subprocess.CalledProcessError as e:
            print(f"✗ Error en entrenamiento: {e}")
            return None
    
    def entrenar(self, ancho=24, alto=24, num_stages=10):
        """Proceso completo de entrenamiento"""
        print("="*60)
        print("ENTRENADOR DE CLASIFICADOR EN CASCADA")
        print("="*60)
        
        # Verificar que existan las carpetas
        if not os.path.exists(self.pos_path):
            print(f"✗ Error: No existe la carpeta {self.pos_path}")
            print("  Crea la carpeta 'positivas' con imágenes del objeto a detectar")
            return None
            
        if not os.path.exists(self.neg_path):
            print(f"✗ Error: No existe la carpeta {self.neg_path}")
            print("  Crea la carpeta 'negativas' con imágenes de fondo (sin el objeto)")
            return None
        
        self.preparar_directorios()
        neg_file, num_neg = self.crear_archivo_negativos()
        pos_file, num_pos = self.crear_archivo_positivos(ancho, alto)
        
        if num_pos < 10:
            print("⚠ Necesitas al menos 10 imágenes positivas (recomendado: 100+)")
            return None
            
        if num_neg < num_pos:
            print("⚠ Se recomienda tener más imágenes negativas que positivas")
        
        vec_file = self.crear_samples_vec(num_pos, ancho, alto)
        if vec_file is None:
            return None
            
        return self.entrenar_cascade(num_pos, num_neg, ancho, alto, num_stages)


def probar_detector(xml_path, imagen_path=None):
    """Prueba el detector entrenado"""
    detector = cv.CascadeClassifier(xml_path)
    
    if imagen_path:
        img = cv.imread(imagen_path)
    else:
        # Usar webcam
        cap = cv.VideoCapture(0)
        ret, img = cap.read()
        cap.release()
    
    if img is None:
        print("No se pudo cargar la imagen")
        return
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    objetos = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    print(f"Objetos detectados: {len(objetos)}")
    
    for (x, y, w, h) in objetos:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv.imshow("Detección", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Configuración
    DATASET_PATH = "/home/panque/repos/JJ"  # Tu ruta de dataset
    OUTPUT_NAME = "caja"      # Nombre del archivo .xml
    
    # Parámetros de entrenamiento
    ANCHO = 24          # Ancho de la ventana de detección
    ALTO = 24           # Alto de la ventana de detección
    NUM_STAGES = 10     # Número de etapas (más = mejor pero más lento)
    
    # Crear entrenador y ejecutar
    trainer = CascadeTrainer(DATASET_PATH, OUTPUT_NAME)
    resultado = trainer.entrenar(ANCHO, ALTO, NUM_STAGES)
    
    # Probar el detector si se creó exitosamente
    if resultado and os.path.exists(resultado):
        print("\n¿Deseas probar el detector? (s/n)")
        # probar_detector(resultado)