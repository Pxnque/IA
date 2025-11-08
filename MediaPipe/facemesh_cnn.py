import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# ----------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------------------------------------------

# Cargar el modelo entrenado
print("Cargando modelo...")
emotion_model = load_model("/home/panque/repos/IA/Eigenface/emotions.h5")
print("✓ Modelo cargado exitosamente")

# Nombres de las emociones (ajusta según tus clases)
# Orden según el entrenamiento del modelo
emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']
# Si tienes otros nombres, cámbialos aquí

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  # Cambiado a 1 para mejor rendimiento
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Captura de video
cap = cv2.VideoCapture(0)

# Landmarks para visualización
selected_points = [33, 133, 362, 263, 61, 291, 4, 36]
CEJA_IZQ = [70, 63, 105, 66, 107]
CEJA_DER = [336, 296, 334, 293, 300]

# Landmarks para extracción de región facial
# Contorno completo de la cara para recortar
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ----------------------------------------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------------------------------------

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def analizar_cejas(face_landmarks):
    """Analiza la posición de las cejas."""
    ojo_izq_y = face_landmarks.landmark[159].y
    ojo_der_y = face_landmarks.landmark[386].y
    
    ceja_izq_y = np.mean([face_landmarks.landmark[i].y for i in CEJA_IZQ])
    ceja_der_y = np.mean([face_landmarks.landmark[i].y for i in CEJA_DER])
    
    dist_izq = abs(ceja_izq_y - ojo_izq_y)
    dist_der = abs(ceja_der_y - ojo_der_y)
    dist_promedio = (dist_izq + dist_der) / 2
    
    return dist_promedio

def extract_face_roi(frame, face_landmarks, target_size=(48, 48)):
    """
    Extrae la región de la cara y la prepara para el modelo.
    """
    h, w = frame.shape[:2]
    
    # Obtener coordenadas del óvalo facial
    x_coords = []
    y_coords = []
    
    for idx in FACE_OVAL:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        x_coords.append(x)
        y_coords.append(y)
    
    # Calcular bounding box con margen
    x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
    y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
    
    # Recortar región
    face_roi = frame[y_min:y_max, x_min:x_max]
    
    if face_roi.size == 0:
        return None, None
    
    # Convertir a escala de grises
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar al tamaño del modelo
    resized_roi = cv2.resize(gray_roi, target_size)
    
    # Normalizar y agregar dimensiones para el modelo
    normalized_roi = resized_roi / 255.0
    model_input = np.expand_dims(normalized_roi, axis=-1)  # Agregar canal
    model_input = np.expand_dims(model_input, axis=0)      # Agregar batch
    
    return model_input, (x_min, y_min, x_max, y_max)

def predict_emotion(model, face_input):
    """
    Predice la emoción usando el modelo.
    """
    if face_input is None:
        return "Unknown", 0.0
    
    # Realizar predicción
    predictions = model.predict(face_input, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    confidence = predictions[0][emotion_idx]
    
    return emotion_labels[emotion_idx], confidence

# ----------------------------------------------------------------
# BUCLE PRINCIPAL
# ----------------------------------------------------------------

print("\n=== INICIANDO DETECCIÓN DE EMOCIONES ===")
print("Presiona 'q' para salir")
print("Presiona 's' para tomar una captura")

frame_count = 0
prediction_interval = 5  # Predecir cada N frames para mejor rendimiento

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # ========================================
            # PREDICCIÓN CON MODELO CNN
            # ========================================
            if frame_count % prediction_interval == 0:
                face_input, bbox = extract_face_roi(frame, face_landmarks)
                
                if face_input is not None:
                    emotion, confidence = predict_emotion(emotion_model, face_input)
                    
                    # Dibujar bounding box de la cara
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Mostrar emoción predicha
                    text = f"{emotion}: {confidence*100:.1f}%"
                    cv2.putText(frame, text, (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # ========================================
            # ANÁLISIS BASADO EN LANDMARKS (Original)
            # ========================================
            puntos = {}
            labio_izq = 61
            labio_der = 291
            
            # Dibujar landmarks seleccionados
            for idx in selected_points + CEJA_IZQ + CEJA_DER:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                
                # Color diferente para cejas
                if idx in CEJA_IZQ or idx in CEJA_DER:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                else:
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
            # Análisis de distancia de labios
            labio_izq_x = int(face_landmarks.landmark[labio_izq].x * frame.shape[1])
            labio_der_x = int(face_landmarks.landmark[labio_der].x * frame.shape[1])
            dist_boca = abs(labio_der_x - labio_izq_x)
            
            # Análisis de cejas
            dist_cejas = analizar_cejas(face_landmarks)
            
            # Clasificación basada en landmarks
            if dist_boca > 65:
                mood_landmarks = "Feliz"
            elif dist_boca < 59:
                if dist_cejas < 0.04:
                    mood_landmarks = "Enojado"
                else:
                    mood_landmarks = "Triste"
            else:
                if dist_cejas > 0.05:
                    mood_landmarks = "Sorprendido"
                else:
                    mood_landmarks = "Neutral"
            
            # Mostrar análisis de landmarks en la esquina inferior
            cv2.putText(frame, f'Landmarks: {mood_landmarks}', (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f'Boca: {int(dist_boca)} | Cejas: {dist_cejas:.3f}', 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    else:
        cv2.putText(frame, 'No face detected', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mostrar frame
    cv2.imshow('Emotion Recognition - CNN + Landmarks', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f'emotion_capture_{frame_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f"✓ Captura guardada: {filename}")
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("\n✓ Aplicación cerrada")