import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class MusicCapsDataset(Dataset):
    def __init__(self, root_dir):
        self.dataset = load_dataset("google/musicCaps", split='train')
        self.root_dir = root_dir
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator((self.tokenizer(item['caption']) for item in self.dataset), specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_path = f"{self.root_dir}/_{item['ytid']}.wav"
        audio_features = self.extract_audio_features(audio_path)
        text_tokens = self.vocab(self.tokenizer(item['caption']))
        max_length = 50
        if len(text_tokens) > max_length:
            text_padded = text_tokens[:max_length]  # Tronquer si nécessaire
        else:
            text_padded = np.pad(text_tokens, (0, max_length - len(text_tokens)), mode='constant', constant_values=self.vocab['<pad>'])
        
        return torch.tensor(audio_features, dtype=torch.float32), torch.tensor(text_padded, dtype=torch.int64)

    def extract_audio_features(self, audio_path):
        if not os.path.exists(audio_path):
            print(f"Le fichier {audio_path} n'a pas été trouvé.")
            return np.zeros((1, 13))  # Retourner un array de zéros si le fichier est manquant
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.T

def custom_collate(batch):
    audios, texts = zip(*batch)
    
    # Trouver la longueur maximale d'audio dans le batch
    max_length = max(audio.shape[0] for audio in audios)
    
    # Appliquer le padding pour rendre tous les extraits audio de même longueur
    padded_audios = [torch.nn.functional.pad(audio, (0, 0, 0, max_length - audio.shape[0])) for audio in audios]
    audios = torch.stack(padded_audios, dim=0)
    
    texts = torch.stack(texts, dim=0)
    return audios, texts


class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.audio_lstm = nn.LSTM(13, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 13)  # Remarquez que le 'hidden_dim' reste sans multiplié par 2

    def forward(self, text, audio):
        text_embedded = self.embedding(text)
        text_output, _ = self.text_lstm(text_embedded)
        audio_output, _ = self.audio_lstm(audio)
        # Appliquer une transformation dense à chaque pas de temps dans la séquence
        output = self.fc(audio_output)  # Applique la couche dense à chaque pas de temps
        return output


def train(model, data_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for audios, texts in data_loader:
            optimizer.zero_grad()
            predictions = model(texts, audios)
            print("Predictions shape:", predictions.shape)  # Debug
            print("Audios shape:", audios.shape)  # Debug
            loss = criterion(predictions, audios)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}: Loss = {loss.item()}')


# Initialisation de l'ensemble de données et du DataLoader
music_dataset = MusicCapsDataset(root_dir="music_data")
data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

# Initialisation du modèle
model = MusicRNN(len(music_dataset.vocab), embedding_dim=128, hidden_dim=128)

# Entraînement du modèle
train(model, data_loader)
