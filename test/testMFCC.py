import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers


df = pd.read_csv('musiccaps-public.csv',sep=";")
print(df)

sample_num = 3 #pick a file to display
filename = df.ytid[sample_num]+str('.wav') #get the filename
#define the beginning time of the signal
y, sr = librosa.load('music_data/'+str(filename))
librosa.display.waveshow(y,sr=sr, x_axis='time', color='purple',offset=0.0)
plt.show()
