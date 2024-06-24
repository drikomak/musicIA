import os

os.environ["TRANSFORMERS_CACHE"] = "model/"
os.environ["HF_HOME"] = "model/"

from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen
import torch

model = musicgen.MusicGen.get_pretrained("small", device="cuda" if torch.cuda.is_available() else "cpu")