import csv
import os
import webbrowser
from home.func import get_image_from_data_url, getEmotionSingle, initModel, gen, randomID

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

ids = []
model = None
init = True

# Create your views here.

def index(request):
    audio_data = []

    try:
        with open(settings.RECORD_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                audio_data.append({
                    'id':row["id"],
                    'name': row['path'],  # Assuming 'path' contains the file name
                    'prompt': row['prompt'],
                    'user': row['user'], 
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
    
    available_audio_files.reverse()

    return render(request, 'home/index.html', {'audio_files': available_audio_files})

def api(request):
    
    prompt = request.GET.get('prompt', '')                  # Récupération du prompt
    user = request.GET.get('name', '') 

    # return render(request, "home/audio.html", {"path":"ABUQH", "prompt":"ttt", "user":"uuu"})

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
    
    return render(request, "home/audio.html", {"path":path, "prompt":prompt, "user":user})

@csrf_exempt
def camera(request):
    filename = get_image_from_data_url(request.POST.get('imageURL', ''))
    emotion = getEmotionSingle(filename)
    return render(request, "home/face.html", {"filename":filename, "emotion":emotion})

@csrf_exempt
def getCamera(request):
    return render(request, "home/setupcam.html", {})

def categ(request):
    
    genre = request.GET.get('genre', '')
    emotion = request.GET.get('emotion', '')
    user = request.GET.get('name', '')
    instruments = request.GET.get('instruments', '')

    prompt = f"{emotion} {genre}"
    
    if instruments != '':
        prompt += f" with {instruments}"

    if not init or len(prompt) == 0:
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
webbrowser.open('http://127.0.0.1:8000', new=0)