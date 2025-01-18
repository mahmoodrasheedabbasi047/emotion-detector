import cv2
import os
import numpy as np
from tensorflow.keras.models import model_from_json

emotion_dict = {}
index = 0
dir_path = r'D://Python//Emotion Detector//data//train'
for path in os.scandir(dir_path):
    if "." not in path.name:
        emotion_dict[index] = path.name
        index+=1


json_file = open("emotion_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("emotion_model.weights.h5")
print("model loaded successfully")

cap = cv2.VideoCapture("D://Python//Emotion Detector//sample video.mp4")

while True:
    ret, frame = cap.read()
    frame  = cv2.resize(frame, (800, 700))

    if not ret:
        break
    
    face_detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=3, minNeighbors=5)


    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 0, 255), 4)
        roi_gray_frame = gray_frame[y:y+h, x:x+h]
        croped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(croped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()