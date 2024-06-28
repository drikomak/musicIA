# MusicIA - Documentation

## Introduction
MusicIA est une application Python permettant de lancer une interface générant des extraits de musique par IA générative, suivant un prompt donné. 

Notre application a été réalisée entre mai 2024 et juin 2024 par Idriss Makroum et Lucas Triozon, dans le cadre du stage de fin d'études de la licence MIASHS de l'Université Paul Valéry Montpellier 3, au sein de Dell Technologies. 

Le modèle derrière MusicIA est [MusicGen](https://musicgen.com/) proposé par Meta.

## Installation et lancement automatique
Pour une installation rapide et efficace de MusicIA, lancez le script `setup.exe` sur votre ordinateur. Tous les logiciels nécessaires au bon fonctionnement de l'application seront installés.

Cela inclut : Chocolatey, Python 3.10, Git et FFMPEG.

Pour lancer MusicIA, ouvrez `run.exe`, et le serveur s'activera tout seul, en ouvrant une page internet quand tout sera prêt.

## Installation manuelle
En cas de problème ou bien pour reprendre le projet, voici la démarche à suivre pour installer et lancer MusicIA manuellement.

MusicIA possède de multiples dépendances pour son fonctionnement. La section qui suit vous indique comment installer tous les composants nécessaires.

#### Python
MusicIA est codé en Python, sur la version majeure `3.10`. Toute autre version ne vous permettra pas forcément de lancer l'application. Ce choix est fait car nous utilisons aussi PyTorch `2.1.0`, qui n'est pas disponible pour les distributions supérieures de Python.

Vous pouvez retrouver toutes les installations de Python sur son [site officiel](https://www.python.org/downloads/).
Si vous ne voulez pas chercher quoi installer, vous pouvez directement aller sur [cette version](https://www.python.org/downloads/release/python-31011/) (`3.10.11`)

**Indiquez bien que Python doit être ajouté au `PATH` lors de l'installation, sinon rien ne fonctionnera !**

L'application MusicIA consiste en un serveur web géré par le framework [Django](https://www.djangoproject.com/), je vous invite à consulter sa documentation en cas de modification dans le code.

#### Télécharger le dépot
Téléchargez tout le repertoire GitHub et déposez-le où vous le souhaitez, en fonction de ce que vous voulez :
- **Lancement depuis `run.exe`** : si vous voulez profiter du lanceur `run.exe`, proposé dans le dépôt, le répertoire doit être ***obligatoirement*** placé dans le dossier `ProgramFiles` de votre machine. Il se trouve en général à l'adresse `C:\Program Files`, mais en cas de doute, vous pouvez appuyer sur `Windows`+`R`, et chercher `%ProgramFiles%` pour trouver l'emplacement automatiquement.
- **Lancement manuel** : le répertoire peut être n'importe où. Dans ce cas présent, je vous conseille de prendre connaissance des [environnements virtuels](https://docs.python.org/3/library/venv.html) Python.

#### Bibliothèques
Une fois Python installé, vous pouvez télécharger toutes les bibliothèques nécessaires à MusicIA. Assurez vous de bien posséder les fichiers `requirementsCUDA.txt` et/ou `requirementsCPU.txt` dans le répertoire de l'application. Ouvrez un terminal **en mode administateur** et entrez les commandes suivantes :

- Accéder au répertoire de MusicIA
  - Récupèrez le chemin entier de là où se trouve MusicIA, par exemple ici `C:\Users\Dell\Documents\musicIA` et positionnez votre terminal dans ce dossier
  - ```bash
    cd C:\Users\Dell\Documents\musicIA
    ```

- Installer les bibliothèques depuis PIP, si vous avez une carte graphique NVIDIA
  - ```bash
    pip install -r requirementsCUDA.txt
    ```

- Installer les bibliothèques depuis PIP, autre GPU (carte graphique d'une autre marque ou aucune)
  - ```bash
    pip install -r requirementsCPU.txt
    ```
  - ***REMARQUE*** : MusicIA sera beaucoup moins performant sans une carte graphique NVIDIA compatible avec la technologie CUDA. 

Le processus d'installation peut prendre beaucoup (beaucoup) de temps. Dans le cas où de nouvelles bibliothèques devraient être ajoutées, merci de tenir ces deux fichiers à jour.

#### FFmpeg / Chocolatey
L'outil FFmpeg est obligatoire pour le bon fonctionnement de MusicIA. Malheureusement, c'est une douleur de l'installer sur Windows et nécessite une connaissance technique poussée.

Je vous propose une alternative à l'installation manuelle : passer par Chocolatey. Mais comme c'est une douleur encore plus grande, nous allons passer par NodeJS.

Installez [NodeJS](https://nodejs.org/en/download/prebuilt-installer). Lors de l'installation, acceptez d'installer les outils complémentaires, dont Chocolatey.

Une fois fait, redémarrez votre ordinateur.

Ensuite, ouvrez à nouveau votre terminal en **mode administrateur** et entrez cette commande : 
```bash
choco install ffmpeg
```

Si l'installation pose problème, il est probable que ce soit un problème de variables d'environnement, et plus précisemment de `PATH`. Vous pouvez trouver de l'aide [ici](https://stackoverflow.com/questions/28235388/where-is-the-chocolatey-installation-path). Si vous êtes totalement inconnu avec le concept de variables systèmes, réferrez vous à ce [tutoriel](https://grafikart.fr/tutoriels/path-windows-1309), proposé par Grafikart.

## Conclusion




