import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import matplotlib.pyplot as plt



class MusicCapsDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.audio_extensions = ['.wav'] 
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        try:
            audio_path = self.find_audio_file(idx)
            if audio_path is None:
                raise FileNotFoundError
            audio_features = self.extract_audio_features(audio_path)
            caption = self.data_frame.iloc[idx]['caption']
            print("fichier audio trouvé et traité")
            return audio_features, caption
        except FileNotFoundError:
            print(f"Fichier audio non trouvé pour l'index {idx}.")
            return None  # Retourner None si le fichier n'est pas trouvé

    def find_audio_file(self, idx):
        base_filename = os.path.join(self.root_dir, self.data_frame.iloc[idx]['ytid'])
        for ext in self.audio_extensions:
            test_path = f"{base_filename}{ext}"
            if os.path.exists(test_path):
                return test_path
        return None  # Retourne None si aucun fichier n'est trouvé

    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return torch.tensor(mfcc.T, dtype=torch.float32)

        
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), []  # Retourner des tensors vides si tous les items sont None
    audios, captions = zip(*batch)
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0)
    return audios, captions


music_dataset = MusicCapsDataset(csv_file='musiccaps-public.csv', root_dir='music_data')
data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)


for audios, captions in data_loader:
    if audios.nelement() == 0:
        continue
    print("Forme des données audio:", audios.shape)
    print("Type des données audio:", audios.dtype)
    print("Légendes:", captions)

    plt.figure(figsize=(10, 4))
    plt.imshow(audios[0].numpy().T, aspect='auto', origin='lower')
    plt.title('MFCC')
    plt.colorbar()
    plt.show()
    break  # Afficher seulement le premier batch valide



