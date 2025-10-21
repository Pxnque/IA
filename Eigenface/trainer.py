import cv2 as cv 
import numpy as np 
import os
dataSet = '/home/panque/repos/IA/Eigenface/emotions2/test'
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataSet+'/'+face
    imageFiles = os.listdir(facePath)
    print(f'Carpeta "{face}": {len(imageFiles)} archivos')
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath+'/'+faceName,0))
    label = label + 1
print(np.count_nonzero(np.array(labels)==0)) 

#faceRecognizer = cv.face.EigenFaceRecognizer_create()
#faceRecognizer.train(facesData, np.array(labels))
#faceRecognizer.write('Emotions.xml')
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('Emotions.xml')
