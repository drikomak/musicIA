import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import gradio as gr
import matplotlib.pyplot as plt
import soundfile as sf
from transformers import T5Tokenizer, T5EncoderModel
from PIL import Image
import io

class MusicCapsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.audio_extensions = ['.wav']
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5EncoderModel.from_pretrained('t5-small')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        try:
            audio_path = self.find_audio_file(idx)
            if audio_path is None:
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

    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        n_fft = 1024  # Set n_fft to match win_length
        hop_length = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)
        
        # Do not trim the frequency bins
        return torch.tensor(stft_magnitude_db.T, dtype=torch.float32)


    def extract_caption_features(self, caption):
        input_ids = self.tokenizer(caption, return_tensors='pt').input_ids
        outputs = self.model(input_ids=input_ids)
        return outputs.last_hidden_state.squeeze(0)

class MusicGenerationModel(nn.Module):
    def __init__(self, text_feature_dim, audio_feature_dim, hidden_dim):
        super(MusicGenerationModel, self).__init__()
        self.text_encoder = nn.LSTM(input_size=text_feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.audio_decoder = nn.LSTM(input_size=hidden_dim, hidden_size=audio_feature_dim, num_layers=2, batch_first=True)
        self.decoder_output_layer = nn.Linear(audio_feature_dim, 513)  # Change to 513

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

def train_model(data_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for text_features, audio_features in data_loader:
            if text_features is None:
                continue
            optimizer.zero_grad()
            output = model(text_features)
            loss = criterion(output, audio_features)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def generate_music(prompt):
    global model
    model.eval()
    with torch.no_grad():
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        t5_model = T5EncoderModel.from_pretrained('t5-small')
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        text_features = t5_model(input_ids=input_ids).last_hidden_state

        # Debugging: Check the shape of text features
        print(f"Text features shape: {text_features.shape}")

        generated_audio_features = model(text_features)

        # Debugging: Check the shape of generated audio features
        print(f"Generated audio features shape: {generated_audio_features.shape}")

        generated_audio_features = generated_audio_features.squeeze(0).numpy().T

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(generated_audio_features, aspect='auto', origin='lower', cmap='coolwarm', interpolation='none')
        ax.set_title('Generated Spectrogram (STFT)')
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Frequency Bins')
        plt.colorbar(im, ax=ax, format='%+2.0f dB')

        buf_image = io.BytesIO()
        plt.savefig(buf_image, format='png')
        buf_image.seek(0)
        image = Image.open(buf_image)
        plt.close(fig)

        n_fft = 1024  # Ensure n_fft matches win_length
        hop_length = 512
        # Using np.exp as the inverse of the log operation
        y_hat = librosa.griffinlim(np.exp(generated_audio_features), n_iter=32, hop_length=hop_length, win_length=n_fft, n_fft=n_fft)

        # Debugging: Check the length of the generated audio
        print(f"Generated audio length: {len(y_hat)} samples")

        audio_file_path = 'temp_audio.wav'
        sf.write(audio_file_path, y_hat, 22050, format='wav')

        return image, audio_file_path


# Initialize the model globally
model = MusicGenerationModel(text_feature_dim=512, audio_feature_dim=512, hidden_dim=256)

# Create the Gradio interface
iface = gr.Interface(fn=generate_music, inputs="text", outputs=["image", "audio"],
                     title="Prompt to Music", description="Enter a text prompt to generate a spectrogram and corresponding audio.")

# Launch the Gradio interface
iface.launch()

# Load dataset and create DataLoader
music_dataset = MusicCapsDataset(csv_file='musiccaps-public.csv', root_dir='music_data')
data_loader = DataLoader(music_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

# Define criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(data_loader, model, criterion, optimizer)
