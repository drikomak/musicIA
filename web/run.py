import subprocess
import os
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def find_manager(path):
    result = []
    for root, dir, files in os.walk(path):
        if "manage.py" in files:
            result.append(os.path.join(root, "manage.py"))
    return result

def fake_wait(text, sleep, color):
    for i in range(5):
        print(f"{color}\r{text}{'.'*(i+1)}{bcolors.ENDC}", flush=True, end='')
        time.sleep(sleep)

try:
    subprocess.run("cd", shell=True)
    print(f"{bcolors.HEADER}/~/ MusicIA 1.1 /~/{bcolors.ENDC}")

    fake_wait("Recherche du fichier 'manage.py' pour démarrer le modèle", 0.5, bcolors.HEADER)
    r = find_manager("/".join(__file__.split("\\")[:-1]))

    if len(r) == 0:
        print(f"{bcolors.FAIL}\nImpossible de lancer MusicIA. Le fichier 'manage.py' est nécessaire au lancement, et n'a pas été trouvé.{bcolors.ENDC}")

        while True:
            time.sleep(100)
    else:
        print(f"{bcolors.OKGREEN}\nFichier trouvé ! Lancement du script.{bcolors.ENDC}\n\n\n\n")
        proc = subprocess.run(f"python {r[0]} runserver", shell=True)

except KeyboardInterrupt:
    print("PROCESS KILLED")


# while True:
#     time.sleep(100)