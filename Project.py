import numpy as np
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained Data.yml")
labels = {}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v :  k for k, v in labels.items()}
#cap = cv2.VideoCapture("d:\\alpha.mp4")
cap = cv2.VideoCapture(0) //default value
#img = cv2.imread("D:\\n3.jpg")
s = (500, 500)
#cv2.resize(img, s)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = img[y:y+h, x:x+w]
        _id, confidence = recognizer.predict(roi_gray)
        print(confidence)
        if confidence >= 45:#check
            print(labels[_id])
            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[_id]
            color = (0, 0, 255)
            stroke = 69
            cv2.putText(img, name, (x, y), font, 8, color, cv2.LINE_AA)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) and 0xff
    if k == 27:
        break
cv2.destroyallwindows()