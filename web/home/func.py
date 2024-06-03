import pandas as pd
import time
import os
from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen
import torch
from string import ascii_uppercase
from random import choice
import cv2
from deepface import DeepFace

# Pour l'alienware seulement
os.environ["TRANSFORMERS_CACHE"] = "D:/hughub"
os.environ["HF_HOME"] = "D:/hughub"

# pour 15sec d'audio
# small : 1min 02sec
# medium : 3min
# large : 

def initModel(init=False):
    if init:
        model = musicgen.MusicGen.get_pretrained("small", device="cuda" if torch.cuda.is_available() else "cpu")
        model.set_generation_params(duration = 15)
    else:
        model = None

    records = pd.read_csv("record.csv", sep=";")
    ids = records.id

    return ids, model

def addRecord(path, prompt, user="Inconnu"):
    user = "Inconnu" if user == "" else user
    records = pd.read_csv("record.csv", sep=";")
    add = pd.DataFrame([{"id":path, 
                         "path":f"{path}.wav", 
                         "prompt":prompt, 
                         "time":time.time(),
                         "user":user
                         }])
    r = pd.concat([records,add])
    r.to_csv("record.csv", sep=';', index=False)

def gen(model, prompt, path, user):
    # try:
        res = model.generate([prompt], progress = True)
        for idx, one_wav in enumerate(res):
            audio_write(f"media/{path}", one_wav.cpu(), model.sample_rate, strategy="loudness")
        addRecord(path, prompt, user)
        return True
    # except Exception as e:
    #     print(e)
    #     return False

def randomID(length=5):
    st = ""
    for i in range(length):
        st += choice(ascii_uppercase)
    return st

def getEmotion():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    emotions_list = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale to RGB format
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']
            
            # Add the detected emotion to the list
            emotions_list.append(emotion)

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        if len(emotions_list) > 100:
            break
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Print the list of detected emotions
    print("Detected Emotions:", emotions_list)

    return max(emotions_list, key=emotions_list.count)