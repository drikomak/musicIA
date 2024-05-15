import nltk
from nltk.tokenize import word_tokenize

def analyze_prompt(prompt):
    nltk.download('punkt')
    tokens = word_tokenize(prompt.lower())
    genre = [word for word in tokens if word in ['jazz', 'rock', 'classique', 'rap']]
    mood = [word for word in tokens if word in ['joyeuse', 'triste', 'énergique', 'dansant']]
    instruments = [word for word in tokens if word in ['piano', 'guitare', 'violon', 'tamtam']]
    return {
        'genre': genre,
        'mood': mood,
        'instruments': instruments
    }

prompt = "Je veux une musique de rap énergique avec un air de tamtam"
result = analyze_prompt(prompt)
print(result)
