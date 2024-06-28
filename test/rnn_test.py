from datasets import load_dataset
import librosa
import numpy as np
import torch
from torch import nn
import torchtext
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
torchtext.disable_torchtext_deprecation_warning()


def custom_collate(batch):
    audios, texts = zip(*batch)
    # Empile les features audio et les séquences textuelles
    audios = torch.stack(audios)
    texts = torch.stack(texts)
    return audios, texts


class MusicCapsDataset(Dataset):
    def __init__(self, data, root_dir):
        self.data = data
        self.root_dir = root_dir
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator((self.tokenizer(desc['caption']) for desc in data), specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = f"{self.root_dir}/{item['ytid']}.wav"
        audio_features = self.extract_audio_features(audio_path)
        text_tokens = self.vocab(self.tokenizer(item['caption']))
        max_length = 50  # Longueur maximale fixée pour tous les textes
        # S'assurer que le padding est correctement appliqué
        text_padded = np.pad(text_tokens, (0, max_length - len(text_tokens)), mode='constant', constant_values=self.vocab['<pad>'])
        # Tronque si jamais il y a un dépassement accidentel
        text_padded = text_padded[:max_length]
        return torch.tensor(audio_features, dtype=torch.float32), torch.tensor(text_padded, dtype=torch.int64)



    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.T


class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.audio_lstm = nn.LSTM(13, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 13)  # Chaque pas de temps prédit un vecteur MFCC

    def forward(self, text, audio):
        text_embedded = self.embedding(text)
        text_output, _ = self.text_lstm(text_embedded)
        audio_output, _ = self.audio_lstm(audio)
        # Nous utilisons la dernière sortie textuelle pour chaque pas de temps audio
        text_output_repeated = text_output[:, -1, :].unsqueeze(1).repeat(1, audio_output.size(1), 1)
        combined_output = torch.cat((text_output_repeated, audio_output), dim=2)
        output = self.fc(combined_output)
        return output


def train(model, data_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for audio, text in data_loader:
            optimizer.zero_grad()
            predictions = model(text, audio)
            loss = criterion(predictions, audio)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


dataset = load_dataset("google/MusicCaps", split='train')

music_dataset = MusicCapsDataset(dataset, "music_data")
data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True)

model = MusicRNN(len(music_dataset.vocab), embedding_dim=128, hidden_dim=128)




# Utiliser la fonction de collation personnalisée dans DataLoader
data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

train(model, data_loader, epochs=10)
