# Driver Monitoring System (DMS)

**Real-Time Driver Monitoring System Using Computer Vision**

## 1. Présentation du projet

Driver Monitoring System (DMS) est un projet de vision par ordinateur visant à développer un système de surveillance du conducteur en temps réel.

L'objectif est d'analyser les caractéristiques faciales à partir d'une webcam afin d'identifier certains indicateurs visuels associés à la fatigue, notamment la fermeture prolongée des yeux et les bâillements.

Le système repose sur une approche hybride combinant un modèle pré-entraîné de détection des points faciaux, des algorithmes géométriques et une analyse temporelle.

**Statut :** Phase initiale de développement — détection faciale.

## 2. Objectifs du projet

Le projet vise à développer progressivement les fonctionnalités suivantes :

- **Face Detection :** détection du visage et extraction des points faciaux.
- **Eye Monitoring :** analyse de l'ouverture des yeux à l'aide de l'Eye Aspect Ratio (EAR).
- **Drowsiness Detection :** identification des périodes prolongées de fermeture des yeux.
- **Yawn Detection :** analyse de l'ouverture de la bouche à l'aide du Mouth Aspect Ratio (MAR).
- **Alert System :** génération d'alertes visuelles et sonores selon les événements détectés.
- **Performance Monitoring :** mesure des FPS, de la latence et de l'utilisation des ressources système.

## 3. Approche technique

L'architecture repose sur trois composants principaux :

1. **MediaPipe Face Landmarker :** extraction des 478 points faciaux à partir des images vidéo.
2. **Analyse géométrique :** calcul de l'EAR et du MAR pour caractériser les mouvements des yeux et de la bouche.
3. **Analyse temporelle :** prise en compte de la durée et de la fréquence des événements afin de distinguer les comportements ponctuels des événements prolongés.

Aucun entraînement de modèle personnalisé n'est prévu dans la première version. Les datasets seront intégrés ultérieurement pour l'évaluation et l'amélioration du système.

## 4. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.10.11 | Langage principal |
| MediaPipe 0.10.21 | Détection des points faciaux |
| OpenCV 4.11.0 | Acquisition et traitement vidéo |
| NumPy 1.26.4 | Calculs numériques |
| psutil | Surveillance des ressources système |
| Docker | Conteneurisation prévue dans une phase ultérieure |

## 5. Architecture du projet

```text
DriverMonitoringSystem/
│
├── dms_env/
│
├── src/
│   ├── main.py
│   ├── face_detection.py
│   ├── eye_detection.py
│   └── alerts.py
│
├── models/
│   └── face_landmarker.task
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## 6. Installation et configuration

### Prérequis

- Windows 11
- Python 3.10.11
- Webcam
- Connexion Internet pour l'installation des dépendances et le téléchargement du modèle

### Création de l'environnement virtuel

```powershell
python -m venv dms_env
```

### Activation de l'environnement

```powershell
.\dms_env\Scripts\Activate.ps1
```

### Installation des dépendances

```powershell
python -m pip install numpy==1.26.4 opencv-contrib-python==4.11.0.86 mediapipe==0.10.21 psutil
```

## 7. Modèle pré-entraîné

Le projet utilise MediaPipe Face Landmarker.

Le fichier du modèle doit être placé dans :

```text
models/face_landmarker.task
```

Modèle utilisé :

[MediaPipe Face Landmarker — Documentation officielle](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

## 8. État d'avancement

- [x] Création de la structure du projet
- [x] Configuration et validation de l'environnement Python
- [x] Intégration de MediaPipe Face Landmarker
- [x] Détection faciale en temps réel
- [ ] Calcul de l'Eye Aspect Ratio (EAR)
- [ ] Calcul du Mouth Aspect Ratio (MAR)
- [ ] Détection temporelle de la somnolence
- [ ] Détection des bâillements
- [ ] Système d'alertes
- [ ] Surveillance des performances
- [ ] Évaluation sur des datasets publics
- [ ] Conteneurisation avec Docker

## 9. Évaluation prévue

L'évaluation portera notamment sur :

- La précision de la détection des événements.
- Les faux positifs et faux négatifs.
- La robustesse aux variations d'éclairage et aux mouvements de la tête.
- Le nombre d'images traitées par seconde (FPS).
- La latence de traitement.
- L'utilisation du CPU et de la mémoire.

Les datasets seront sélectionnés et intégrés après la réalisation du premier prototype fonctionnel.

## 10. Déploiement

Le développement et les premiers tests seront réalisés localement sous Windows.

La conteneurisation avec Docker sera effectuée après validation du prototype, afin de garantir la reproductibilité de l'environnement et de faciliter le déploiement.

## 11. Limites

Ce projet constitue un prototype expérimental de vision par ordinateur. Les indicateurs EAR et MAR ne permettent pas, à eux seuls, de diagnostiquer la fatigue ou de garantir la sécurité d'un conducteur.

Le système devra faire l'objet d'une évaluation approfondie avant toute utilisation dans un contexte réel.

---

**Project:** Driver Monitoring System  
**Category:** Computer Vision · Artificial Intelligence · Real-Time Video Processing  
**Development status:** Initial prototype