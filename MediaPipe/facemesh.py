import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

# Lista de índices de landmarks específicos (ojos y boca)
selected_points = [33, 133, 362, 263, 61, 291, 4, 36]  # Ojos y boca


def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            puntos = {}
            labio_izq=selected_points[4]#61 = labio izquierdo
            labio_der=selected_points[5]#291 = labio derecho
            
            for idx in selected_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dibuja el punto en verde
                
            dist=distancia(int(face_landmarks.landmark[labio_izq].x * frame.shape[1]),int(face_landmarks.landmark[labio_der].x * frame.shape[1]))

                
            if dist > 68:
                mood="Feliz"
            elif dist < 55:
                mood="Triste"
            else:
                mood="Neutral"
            
            #print(distancia(labio_izq,labio_der))
            cv2.putText(frame,f'{mood}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,125),2)
             
            # Calcular y mostrar distancia entre puntos (ejemplo: entre ojos)
            if 33 in puntos and 133 in puntos:
                d_ojos = distancia(puntos[33], puntos[133])
                #print(puntos[33])
                #cv2.line(frame, (puntos[33][0], puntos[33][1]), (puntos[133][0], puntos[133][1]), (23, 234,23), 2 )
                cv2.putText(frame, f" {int(puntos[61][0]-20), int(puntos[61][1]-20) }", (puntos[61][0], puntos[61][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(frame, f"D: {int(d_ojos)}", (puntos[33][0], puntos[33][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('PuntosFacialesMediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()