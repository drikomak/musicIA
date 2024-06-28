import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import soundfile as sf

audio_path = 'music_data/-6HBGg1cAI0.wav'
y, sr = librosa.load(audio_path, sr=None)  


D = librosa.stft(y)  # STFT du signal
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convertir l'amplitude en dB

# Affichage 
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme')
plt.tight_layout()
plt.show()

# Reconstruire le signal audio à partir de la STFT
y_reconstructed = librosa.istft(D)


# Générer les axes de temps et de fréquence
t = np.linspace(0, len(y) / sr, num=D.shape[1])
f = np.linspace(0, sr / 2, num=D.shape[0])

T, F = np.meshgrid(t, f)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracer la surface
surf = ax.plot_surface(T, F, S_db, cmap='viridis', edgecolor='none')

# Ajouter les labels des axes et le titre
ax.set_xlabel('Temps (s)')
ax.set_ylabel('Fréquence (Hz)')
ax.set_zlabel('Amplitude (dB)')
ax.set_title('Spectrogramme 3D')

# Ajouter une barre de couleur
fig.colorbar(surf, ax=ax, pad=0.2)

plt.show()

# Chemin du dossier de sortie
output_folder = 'new_audio'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Chemin des fichiers audio dans le nouveau dossier
original_audio_output = os.path.join(output_folder, 'original_audio.wav')
reconstructed_audio_output = os.path.join(output_folder, 'reconstructed_audio.wav')

# Sauvegarder les fichiers audio
sf.write(original_audio_output, y, sr)  
sf.write(reconstructed_audio_output, y_reconstructed, sr)  
sf.write('reconstructed_audio.wav', y_reconstructed, sr)