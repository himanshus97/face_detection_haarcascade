# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:52:00 2017

@author: himanshu
"""
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret , img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color= img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (xx,yy,ww,hh) in eyes:
            cv2.rectangle(roi_color,(xx,yy),(xx+ww, yy+hh),(255,255,0), 2)
    
    cv2.imshow('frame',img)
    
    if cv2.waitKey(1) & 0xff== ord('q'):
        break
         
cap.release()
cv2.destroyAllWindows()


