import time

import cv2
import os
import webbrowser
import gtts
from playsound import playsound
import pyttsx3


face_detect = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer.create()
rec.read("recognizer\\trainingData.yml")
ide = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# engine = pyttsx3.init()
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ide, conf = rec.predict(gray[y:y + h, x:x + w])
        if conf < 60:
            if ide == 2:
                ide = "Project"
                print('welcome')

                t1 = gtts.gTTS("DETECTED SUCCESSFULLY")
                # t1.save("A.mp3")
                playsound("A.mp3")

                cv2.waitKey(2)
                cam.release()
                cv2.destroyAllWindows()
                webbrowser.open_new_tab('http://127.0.0.1:8000/')
                os.system('python manage.py runserver')

            elif ide == 2:
                ide = "Manu"
            elif ide == 3:
                ide = "varun"
            elif ide == 4:
                ide = "laxmish"
        else:
            ide = "Unknown"
            t1 = gtts.gTTS("Not Detected")
            t1.save("b.mp3")
            #playsound("b.mp3")
            # engine.say("NOT DETECTED")
            # engine.setProperty('rate', 120)
            # engine.setProperty('volume', 0.9)
            # engine.runAndWait()
        cv2.putText(img=img, text=str(ide), org=(x, y+h), fontFace=font, fontScale=2.0, color=[255.0], thickness=2)
    cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()