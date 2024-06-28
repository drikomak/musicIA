import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np

# Définition du Dataset
class MusicCapsDataset(torch.utils.data.Dataset):
    def __init__(self, captions, features):
        self.captions = captions
        self.features = features
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        return self.captions[idx], self.features[idx]

def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    captions = df['caption'].apply(word_tokenize).tolist()
    word_model = Word2Vec(captions, vector_size=100, window=5, min_count=1, workers=4)
    vocab = word_model.wv.key_to_index

    caption_vectors = [torch.tensor([word_model.wv[word] for word in cap if word in vocab], dtype=torch.float32) for cap in captions]
    caption_tensors = pad_sequence(caption_vectors, batch_first=True, padding_value=0.0)

    max_length = max([len(feat) for feat in caption_vectors])
    features = [torch.randn(max_length, 128) for _ in range(len(caption_vectors))]

    return caption_tensors, torch.stack(features), word_model


# Modèle RNN pour la génération
class MusicGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MusicGenerator, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def collate_fn(batch):
    # Unpack the batch
    captions, features = zip(*batch)
    
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0.0)
    
    # Assuming 'features' are lists of tensors and need to be converted to tensors
    features_padded = pad_sequence([torch.tensor(f) for f in features], batch_first=True, padding_value=0.0)
    
    return captions_padded, features_padded

captions, features, word_model = prepare_data('musiccaps-public.csv')
train_captions, test_captions, train_features, test_features = train_test_split(captions, features, test_size=0.2)
train_dataset = MusicCapsDataset(train_captions, train_features)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

# Entraînement du modèle
model = MusicGenerator(input_dim=100, hidden_dim=256, output_dim=128)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    for captions, features in train_loader:
        outputs = model(captions)
        loss = criterion(outputs, features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


def generate_music_from_prompt(prompt, word_model, model):
    tokenized_prompt = word_tokenize(prompt)
    prompt_vector = [word_model.wv[word] for word in tokenized_prompt if word in word_model.wv.key_to_index]
    prompt_tensor = torch.tensor([prompt_vector], dtype=torch.float32)
    with torch.no_grad():
        generated_features = model(prompt_tensor)
    return generated_features.numpy()


prompt = "A soft piano playing in a large hall"
generated_audio_features = generate_music_from_prompt(prompt, word_model, model)
