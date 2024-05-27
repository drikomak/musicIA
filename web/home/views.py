import time
from django.shortcuts import render
from home.func import getEmotion, initModel, gen, randomID
import os
from django.http import HttpResponse
from django.conf import settings
import csv

ids = []
model = None
init = True

# Create your views here.


def index(request):
    audio_data = []

    try:
        with open('record.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                audio_data.append({
                    'name': row['path'],  # Assuming 'path' contains the file name
                    'prompt': row['prompt'],
                    'url': os.path.join(settings.MEDIA_URL, row['path'])
                })
    except FileNotFoundError:
        return HttpResponse("Le fichier CSV est introuvable.")
    except Exception as e:
        return HttpResponse(f"Une erreur s'est produite: {str(e)}")
    
    media_path = settings.MEDIA_ROOT
    available_audio_files = []
    if os.path.exists(media_path):
        for file_name in os.listdir(media_path):
            if file_name.endswith('.wav'):
                for audio in audio_data:
                    if audio['name'] == file_name:
                        available_audio_files.append(audio)
                        break
    else:
        return HttpResponse("Le chemin des médias n'existe pas.")

    return render(request, 'home/index.html', {'audio_files': available_audio_files})

def api(request):
    
    prompt = request.GET.get('prompt', '')                  # Récupération du prompt
    user = request.GET.get('name', 'Inconnu') 

    if not init or len(prompt) == 0:                        # Vérification que le modèle est lancé et que le prompt n'est pas vide
        return render(request, "home/error.html", {})
    if len(user) == 0 or len(user) > 20:
        user = "Inconnu"

    path = randomID()                                       # Création d'un ID unique
    while path in ids:
        path = randomID()
    ids.add(path)

    if not gen(model, prompt, path, user):                        # Si la génération s'est bien passée
        return render(request, "home/error.html", {}) 
    
    return render(request, "home/audio.html", {"path":path, "prompt":prompt})

def camera(request):
    return render(request, "home/face.html", {"emotion":getEmotion()})



def get_audio_files(request):
    media_path = settings.MEDIA_ROOT
    print(f"Checking media path: {media_path}")  # Debug
    if not os.path.exists(media_path):
        print("Media path does not exist")  # debug
        return render(request, 'error.html', {'error': 'Media path does not exist'}, status=400)

    audio_files = []
    for file_name in os.listdir(media_path):
        if file_name.endswith('.wav'):
            print(f"Found audio file: {file_name}")  # debug
            audio_files.append({
                'name': file_name,
                'url': os.path.join(settings.MEDIA_URL, file_name)
            })
    
    print(f"Returning audio files: {audio_files}")  # debug
    return render(request, 'audio_files.html', {'audio_files': audio_files})

ids, model = initModel(init)