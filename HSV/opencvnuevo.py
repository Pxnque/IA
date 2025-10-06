import cv2 as cv
img = cv.imread("/home/panque/uni/ia/figura.png")
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
ubb = (0,60,60)
uba = (10,255,255)
ubb1 = (170,60,60)
ubb2 = (180,255,255)
mask1 = cv.inRange(hsv,ubb,uba)
mask2 = cv.inRange(hsv,ubb1,ubb2)
mask = mask1 + mask2
# para el color azul
bluebb = (100, 60, 60)
bluebba = (130, 255, 255)
bluemask1 = cv.inRange(hsv,bluebb,bluebba)
azul = cv.bitwise_and(img,img,mask=bluemask1)
# para el verde
greenbb = (40, 100, 100)
greenba = (80, 255, 255)
greenmask = cv.inRange(hsv,greenbb,greenba)
green = cv.bitwise_and(img,img,mask=greenmask)
# para el amarillo
yellowbb = (20, 100, 100)
yellowba = (35, 255, 255)
2
resultado = cv.bitwise_and(img,img ,mask=mask)
cv.imshow('resutlado',resultado)
cv.imshow('azul',azul)
cv.imshow('green',green)
cv.imshow('yellow',yellow)

def encontrar_centrar(mask,color):
    gris = mask.detectMultiScale(color, 1.3, 5)


cv.imshow('rostros', img2)
cv.imshow('cara', img)
#cv.imshow('mask',mask)
#cv.imshow('img',img)
#cv.imshow('hsv',hsv)
cv.waitKey(0)
cv.destroyAllWindows()