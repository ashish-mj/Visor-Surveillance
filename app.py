#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:18:46 2021

@author: ashish
"""

from flask import Flask, render_template,Response
from flask_mail import Message,Mail
import cv2
import numpy as np
from keras.models import load_model
import time
import requests

app = Flask(__name__)
app.config['SECRET_KEY']='Mask_detect'
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT']=465
app.config['MAIL_USERNAME']='ana.customer1000@gmail.com'
app.config['MAIL_PASSWORD']='sooleinchara'
app.config['MAIL_USE_TLS']=False
app.config['MAIL_USE_SSL']=True


mail = Mail(app)



headers = {'authorization': "",'Content-Type': "application/x-www-form-urlencoded",'Cache-Control': "no-cache",}
url = "https://www.fast2sms.com/dev/bulk"

face_cascade = cv2.CascadeClassifier('code/haarcascade_frontalface_default.xml')
model = load_model("code/masknet2.h5")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("code/trainer2.yml")

results={0:'Mask',1:'No Mask'}
GR_dict={0:(0,255,0),1:(0,0,255)}

def send_notification(item):
    info = Message("Covid Norms Violation",sender = "ana.customer1000@gmail.com",recipients = [item["Email"]])
    info.body = item["Name"]+" please wear the mask.\nThank you."
    mail.send(info)
    
    


cap=""

def stream():
    global cap
    global face_cascade
    global recognizer
    global data
    global url
    global headers
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while True:
        _,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        new_img = cv2.cvtColor(gray,cv2.COLOR_RGB2BGR)
    
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
        for (x, y, w, h) in faces:
             
            crop = new_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            result=model.predict(crop)
            label=np.argmax(result,axis=1)[0]
            
            Id, confidence = recognizer.predict(gray_img[y:y+h,x:x+w])
            if (confidence<100 and label==1):
                print(data[Id]["Name"]+"\nConfidence"+str(round(100-confidence)))
                if data[Id]["Violation"]==None:
                    with app.app_context():
                        send_notification(data[Id])
                        payload = "sender_id=FSTSMS&message=Please Wear Mask&language=english&route=p&numbers="+str(data[Id]["Phone"])
                        requests.request("POST", url, data=payload, headers=headers)
                        data[Id]["Violation"]=time.process_time()
                        print(data[Id])
                elif data[Id]["Violation"]-time.process_time()>60:
                    with app.app_context():
                        send_notification(data[Id])
                        data[Id]["Violation"]=int(time.process_time())
                        print(data[Id])
          
            cv2.rectangle(frame,(x,y),(x+w,y+h),GR_dict[label],2)
        imgencode=cv2.imencode('.jpg',frame)[1]
        strinData = imgencode.tobytes()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+strinData+b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
	return render_template('home.html')

if __name__ == '__main__':
	app.run(debug=True,port=8080)