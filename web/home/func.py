import shutil
import os
from django.conf import settings
os.environ["TRANSFORMERS_CACHE"] = settings.MODEL_DIR
os.environ["HF_HOME"] = settings.MODEL_DIR

import pandas as pd
import time
import torch
import cv2
import base64, secrets, io

from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen
from string import ascii_uppercase
from random import choice
from deepface import DeepFace
from PIL import Image

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

    records = pd.read_csv(settings.RECORD_PATH, sep=";")
    liste = []
    for i in records.iterrows():
        if time.time() - i[1]["time"] > 1209600:
            liste.append(i[0])
            try:
                os.remove(f"{settings.MEDIA_PATH}{i[1]['path']}")
            except:
                pass

    records = records.drop(liste)
    ids = records.id
    records.to_csv(settings.RECORD_PATH, sep=';', index=False)

    return ids, model

def addRecord(path, prompt, user="Inconnu"):
    user = "Inconnu" if user == "" else user
    records = pd.read_csv(settings.RECORD_PATH, sep=";")
    add = pd.DataFrame([{"id":path, 
                         "path":f"{path}.wav", 
                         "prompt":prompt, 
                         "time":time.time(),
                         "user":user
                         }])
    r = pd.concat([records,add])
    r.to_csv(settings.RECORD_PATH, sep=';', index=False)

def gen(model, prompt, path, user):
    try:
        res = model.generate([prompt], progress = True)
        for idx, one_wav in enumerate(res):
            audio_write(f"{settings.MEDIA_PATH}{path}", one_wav.cpu(), model.sample_rate, strategy="loudness")
        addRecord(path, prompt, user)
        return True
    except Exception as e:
        print(e)
        return False


def randomID(length=5):
    st = ""
    for i in range(length):
        st += choice(ascii_uppercase)
    return st

def get_image_from_data_url(data_url):

    # getting the file format and the necessary dataURl for the file
    _format, _dataurl       = data_url.split(';base64,')
    # file name and extension
    _filename, _extension   = secrets.token_hex(20), _format.split('/')[-1]

    image_bytes = base64.b64decode(_dataurl)
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    shutil.rmtree("{settings.MEDIA_PATH}camera")
    os.mkdir("{settings.MEDIA_PATH}camera")
    name = f"{settings.MEDIA_PATH}camera/{_filename}.{_extension}"
    image.save(name)

    return name


def getEmotionSingle(filename):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotion = None

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imwrite(filename, frame)

    return emotion