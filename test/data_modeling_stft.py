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
            # audio, caption et tags
            return self.extract_audio_features(audio_path), self.extractTags(idx)
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
        stft = librosa.stft(y)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)
        return torch.tensor(stft_magnitude_db.T, dtype=torch.float32)
    
    def extractTags(self, idx):
        tags = self.data_frame.iloc[idx]['aspect_list']
        tags = tags.replace("[","")
        tags = tags.replace("]","")
        tags = tags.replace("'","")
        return tags.split("; ")

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), []
    audios, tags = zip(*batch)
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=-100)  # Utiliser un padding value qui correspond aux valeurs dB faibles pour la visualisation
    return audios, tags

music_dataset = MusicCapsDataset(csv_file='musiccaps-public.csv', root_dir='music_data')
data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

for audios, captions in data_loader:
    print(audios)
    if audios.nelement() == 0:
        continue
    print("Forme des données audio:", audios.shape)
    print("Type des données audio:", audios.dtype)
    print("Légendes:", captions)

    plt.figure(figsize=(10, 4))
    print(len(audios[0].numpy().T), len(audios[0].numpy()))
    plt.imshow(audios[0].numpy().T, aspect='auto', origin='lower', cmap='coolwarm', interpolation='none')
    plt.title('Spectrogram (STFT)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.colorbar(format='%+2.0f dB')
    plt.show()





import torch
import torch.nn as nn

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
    

learning_rate = 5e-5
num_hidden_units = 16

model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")