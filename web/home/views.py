import time
from django.shortcuts import render
from home.func import getEmotion, initModel, gen, randomID

ids = []
model = None
init = True

# Create your views here.

def index(request):
    return render(request, "home/index.html", {})

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


ids, model = initModel(init)