import cv2
import numpy
face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier('haarcascade_eye.xml') #for detecting eyes
def dector(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    if faces is ():
        print("no face")
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+w),(127,0,255),5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

    return img
cap = cv2.VideoCapture(0)
while True:
    ret,frame =cap.read()
    cv2.imshow("face dect",dector(frame))
    if cv2.waitKey(1)==13:   #enter
        break
    
cap.release()
cv2.destroyAllWindows()  
