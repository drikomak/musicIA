import pandas as pd
import time
import os
from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch
from string import ascii_uppercase
from random import choice

def initModel(init=False):
    if init:
        model = musicgen.MusicGen.get_pretrained("small", device="cpu")
        model.set_generation_params(duration = 15)
    else:
        model = None

    records = pd.read_csv("record.csv", sep=";")
    ids = records.id

    return ids, model

def addRecord(path, prompt, maxi=5):
    records = pd.read_csv("record.csv", sep=";")
    add = pd.DataFrame([{"id":path, 
                         "path":f"{path}.wav", 
                         "prompt":prompt, 
                         "time":time.time(),
                         }])
    r = pd.concat([records,add])

    while len(r) > maxi:
        p = r.path[0]
        r = r.drop([0])
        os.remove(f"media/{p}")

    r.to_csv("record.csv", sep=';', index=False)

def gen(model, prompt, path):
    try:
        res = model.generate([prompt], progress = True)
        for idx, one_wav in enumerate(res):
            audio_write(f"media/{path}", one_wav.cpu(), model.sample_rate, strategy="loudness")
        addRecord(path, prompt)
        return True
    except Exception as e:
        print(e)
        return False

def randomID(length=5):
    st = ""
    for i in range(length):
        st += choice(ascii_uppercase)
    return st
