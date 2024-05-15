import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import matplotlib.pyplot as plt
import soundfile as sf
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm  

class MusicCapsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, sequence_length=100, device="cpu"):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.audio_extensions = ['.wav']
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5EncoderModel.from_pretrained('t5-small').to(device)
        self.sequence_length = sequence_length
        self.device = device

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        try:
            audio_path = self.find_audio_file(idx)
            if (audio_path is None):
                raise FileNotFoundError
            audio_features = self.extract_audio_features(audio_path)
            caption = self.data_frame.iloc[idx]['caption']
            caption_features = self.extract_caption_features(caption)
            return audio_features, caption_features
        except FileNotFoundError:
            print(f"Audio file not found for index {idx}.")
            return None

    def find_audio_file(self, idx):
        base_filename = os.path.join(self.root_dir, self.data_frame.iloc[idx]['ytid'])
        for ext in self.audio_extensions:
            test_path = f"{base_filename}{ext}"
            if os.path.exists(test_path):
                return test_path
        return None

    def extract_audio_features(self, audio_path, device="cpu"):
        y, sr = librosa.load(audio_path, sr=None)
        n_fft = 1024
        hop_length = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)
        stft_magnitude_db = (stft_magnitude_db - np.min(stft_magnitude_db)) / (np.max(stft_magnitude_db) - np.min(stft_magnitude_db))
        
        if stft_magnitude_db.shape[1] > self.sequence_length:
            stft_magnitude_db = stft_magnitude_db[:, :self.sequence_length]
        else:
            pad_width = self.sequence_length - stft_magnitude_db.shape[1]
            stft_magnitude_db = np.pad(stft_magnitude_db, ((0, 0), (0, pad_width)), mode='constant')

        # Ensure the output has 512 dimensions
        return torch.tensor(stft_magnitude_db.T, dtype=torch.float32)[:,:512].to(self.device)

    def extract_caption_features(self, caption):
        input_ids = self.tokenizer(caption, return_tensors='pt').input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids)
        if outputs.last_hidden_state.size(1) > self.sequence_length:
            outputs = outputs.last_hidden_state[:, :self.sequence_length, :]
        else:
            pad_width = self.sequence_length - outputs.last_hidden_state.size(1)
            outputs = F.pad(outputs.last_hidden_state, (0, 0, 0, pad_width), mode='constant', value=0)
        return outputs.squeeze(0)

class MusicGenerationModel(nn.Module):
    def __init__(self, text_feature_dim, audio_feature_dim, hidden_dim):
        super(MusicGenerationModel, self).__init__()
        self.text_encoder = nn.LSTM(input_size=text_feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.audio_decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.decoder_output_layer = nn.Linear(hidden_dim, audio_feature_dim)

    def forward(self, text_features):
        encoder_output, (hidden, cell) = self.text_encoder(text_features)
        decoder_output, _ = self.audio_decoder(encoder_output)
        output = self.decoder_output_layer(decoder_output)
        return output

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    audios, captions = zip(*batch)
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=-100)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)
    return audios, captions

def train_model(data_loader, model, criterion, optimizer, num_epochs=10, save_path='model_checkpoint.pth', device="cpu"):

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            if batch is None:
                continue
            text_features, audio_features = batch

            # Move data to the device
            text_features = text_features.to(device)
            audio_features = audio_features.to(device)

            optimizer.zero_grad()
            output = model(text_features)
            loss = criterion(output, audio_features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix({
                "Batch": i+1,
                "Loss": loss.item()
            })

        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}')

        # Save model checkpoint
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

def load_model(model, load_path='model_checkpoint.pth'):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    print(f"Model loaded from {load_path}")
    return model

def generate_music(prompt, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        t5_model = T5EncoderModel.from_pretrained('t5-small').to(device)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        text_features = t5_model(input_ids=input_ids).last_hidden_state.to(device)

        generated_audio_features = model(text_features)

        generated_audio_features = generated_audio_features.squeeze(0).cpu().numpy().T
        generated_audio_features = (generated_audio_features - np.min(generated_audio_features)) / (np.max(generated_audio_features) - np.min(generated_audio_features))

        librosa.display.specshow(generated_audio_features, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogramme')
        plt.tight_layout()
        plt.show()

        n_fft = 1023
        hop_length = 512
        e = np.exp(generated_audio_features)
        y_hat = librosa.griffinlim(e, n_iter=32, hop_length=hop_length, win_length=n_fft, n_fft=n_fft)

        audio_file_path = 'temp_audio236.wav'
        sf.write(audio_file_path, y_hat, 22050, format='wav')

        return audio_file_path

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGenerationModel(text_feature_dim=512, audio_feature_dim=512, hidden_dim=256).to(device)

    path = "model_checkpoint.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        music_dataset = MusicCapsDataset(csv_file='musiccaps-public.csv', root_dir='music_data', device=device)
        data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(data_loader, model, criterion, optimizer, num_epochs=10, device=device)

    prompt = "A calm and relaxing piano melody with soft background strings."
    audio_file_path = generate_music(prompt, model, device=device)
    print(f"Generated audio saved to {audio_file_path}")
