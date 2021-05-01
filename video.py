#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:57:28 2021

@author: ashish
"""


import cv2

cap = cv2.VideoCapture('people_walk.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
result = cv2.VideoWriter('output.avi',  cv2.VideoWriter_fourcc(*'MJPG'),30, (frame_width,frame_height))
face_cascade = cv2.CascadeClassifier('code/haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(1280,700),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
        for (x, y, w, h) in faces:
                cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
                
        result.write(gray)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
result.release()
cv2.destroyAllWindows()
