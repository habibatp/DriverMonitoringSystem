# Driver Monitoring System (DMS)

**Real-Time Driver Monitoring System Using Computer Vision**

## 1. Présentation du projet

Driver Monitoring System (DMS) est un projet de vision par ordinateur visant à développer un système de surveillance du conducteur en temps réel.

Le système analyse les caractéristiques faciales à partir d'une webcam afin d'identifier des indicateurs visuels associés à la fatigue, notamment :

- La fermeture prolongée des yeux.
- Les bâillements.
- Les mouvements inhabituels ou prolongés de la tête.
- Les signes de fatigue résultant de la combinaison de plusieurs indicateurs.

Le projet repose sur une approche hybride combinant un modèle pré-entraîné de détection des points faciaux, des algorithmes géométriques et une analyse temporelle.

L'application utilise une architecture client-serveur : la webcam est gérée côté Windows, tandis que le traitement des images est effectué par une API FastAPI exécutée dans un conteneur Docker.

**Statut :** Prototype fonctionnel – intégration de la vision par ordinateur, de l'API et de Docker.

## 2. Fonctionnalités principales

- **Face Detection :** détection du visage et extraction des points faciaux avec MediaPipe Face Landmarker.
- **Eye Monitoring :** analyse de l'ouverture des yeux à l'aide de l'Eye Aspect Ratio (EAR).
- **Drowsiness Detection :** détection des fermetures prolongées des yeux à partir d'une analyse temporelle.
- **Yawn Detection :** détection des bâillements à l'aide du Mouth Aspect Ratio (MAR).
- **Head Pose Estimation :** estimation des angles de rotation de la tête (Yaw, Pitch et Roll) avec OpenCV et solvePnP.
- **Fatigue Detection :** combinaison de plusieurs indicateurs pour déterminer un niveau de fatigue.
- **Alert System :** affichage d'alertes visuelles et émission de notifications sonores côté client.
- **Head Pose Calibration :** calibration et réinitialisation de la position de référence de la tête.
- **REST API :** traitement des images et transmission des résultats au client via FastAPI.
- **Docker Deployment :** exécution de l'API et du traitement dans un environnement conteneurisé.

## 3. Approche technique

Le système repose sur plusieurs étapes de traitement :

### 3.1. Extraction des points faciaux

MediaPipe Face Landmarker est utilisé pour détecter les points caractéristiques du visage à partir des images de la webcam.

### 3.2. Analyse géométrique

Deux indicateurs principaux sont calculés :

- **EAR (Eye Aspect Ratio) :** mesure le degré d'ouverture des yeux.
- **MAR (Mouth Aspect Ratio) :** mesure le degré d'ouverture de la bouche.

Ces indicateurs permettent d'identifier les fermetures des yeux et les ouvertures prolongées de la bouche.

### 3.3. Estimation de la pose de la tête

OpenCV et l'algorithme `solvePnP` sont utilisés pour estimer l'orientation de la tête selon trois angles :

- Yaw : rotation horizontale.
- Pitch : rotation verticale.
- Roll : inclinaison latérale.

Une calibration permet de définir une position de référence.

### 3.4. Analyse temporelle et détection de la fatigue

Les événements détectés sont analysés dans le temps afin de distinguer les mouvements ponctuels des comportements prolongés ou répétés.

Le système combine les indicateurs oculaires, les bâillements et la pose de la tête pour déterminer un niveau de fatigue :

- NORMAL
- ATTENTION
- WARNING
- HIGH_RISK

Ces niveaux correspondent à des règles expérimentales définies dans le prototype.

## 4. Architecture du système

L'application adopte une architecture client-serveur.

1. **Client Windows :** capture les images depuis la webcam avec OpenCV.
2. **Communication HTTP :** transmet les images à l'API FastAPI.
3. **Serveur Docker :** reçoit les images et exécute les modules de détection.
4. **Traitement :** MediaPipe et les algorithmes géométriques analysent les caractéristiques faciales.
5. **Réponse API :** retourne les métriques et l'image annotée au format JSON.
6. **Affichage côté client :** affiche les résultats et déclenche les alertes sonores localement.

### Architecture simplifiée

```text
              WINDOWS
        +-------------------+
        |   Webcam Client   |
        |      OpenCV       |
        +---------+---------+
                  |
                  | HTTP POST
                  | /process-frame
                  v
        +-------------------+
        |    Docker         |
        |                   |
        |   FastAPI         |
        |       |           |
        |       v           |
        |  DMS Processor    |
        |       |           |
        |  +----+---------+  |
        |  |    |         |  |
        | MediaPipe       |  |
        | EAR / MAR       |  |
        | Head Pose       |  |
        | Fatigue         |  |
        +---------+---------+
                  |
                  | JSON Response
                  | Annotated Image
                  | Detection Metrics
                  v
        +-------------------+
        |   Windows Client  |
        |                   |
        | Display & Alerts  |
        +-------------------+
```

## 5. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.10 | Langage principal |
| MediaPipe 0.10.21 | Détection des points faciaux |
| OpenCV 4.11.0 | Traitement d'images, vidéo et estimation de pose |
| NumPy 1.26.4 | Calculs numériques |
| FastAPI | Développement de l'API REST |
| Uvicorn | Serveur ASGI |
| Docker | Conteneurisation |
| psutil | Surveillance des ressources système |

## 6. Structure du projet

```text
DriverMonitoringSystem/
│
├── models/
│   └── face_landmarker.task
│
├── src/
│   ├── api.py
│   ├── main.py
│   ├── dms_processor.py
│   ├── face_detection.py
│   ├── eye_detection.py
│   ├── yawn_detection.py
│   ├── head_pose.py
│   ├── fatigue_detection.py
│   ├── alerts.py
│   ├── evaluate_closed_eyes.py
│   ├── test_eye_detection.py
│   ├── test_fatigue_detection.py
│   └── test_head_pose_webcam.py
│
├── webcam_client.py
├── test_api_client.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

Les datasets et les résultats locaux d'évaluation ne sont pas inclus dans le dépôt GitHub.

## 7. Installation et configuration

### Prérequis

- Windows 10 ou Windows 11.
- Python 3.10.
- Docker Desktop.
- Une webcam.
- Git.

### 7.1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/DriverMonitoringSystem.git
cd DriverMonitoringSystem
```

Remplacer `VOTRE_USERNAME` par votre nom d'utilisateur GitHub.

### 7.2. Créer un environnement virtuel

```powershell
python -m venv dms_env
```

### 7.3. Activer l'environnement

```powershell
.\dms_env\Scripts\Activate.ps1
```

### 7.4. Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

### 7.5. Vérifier le modèle MediaPipe

Le modèle Face Landmarker doit être présent dans :

```text
models/face_landmarker.task
```

Le fichier du modèle est inclus dans le dépôt.

Documentation officielle :

[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

## 8. Exécution locale

Le système peut être exécuté sans Docker en lançant l'API et le client dans deux terminaux distincts.

### Terminal 1 : démarrer l'API

Depuis le dossier racine du projet :

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Vérifier le fonctionnement de l'API :

```text
http://127.0.0.1:8000/health
```

Documentation interactive :

```text
http://127.0.0.1:8000/docs
```

### Terminal 2 : démarrer le client webcam

Activer l'environnement virtuel, puis exécuter :

```powershell
python .\webcam_client.py
```

Commandes disponibles dans le client :

| Touche | Action |
|---|---|
| C | Calibrer la pose de la tête |
| R | Réinitialiser la calibration |
| Q | Quitter l'application |

## 9. API REST

L'application expose plusieurs endpoints :

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Vérifier l'état de l'API |
| POST | `/process-frame` | Traiter une image et retourner les résultats |
| POST | `/calibrate` | Calibrer la pose de la tête |
| POST | `/reset-calibration` | Réinitialiser la calibration |

L'endpoint `/process-frame` reçoit une image et retourne notamment :

- L'image annotée encodée en Base64.
- Les indicateurs EAR et MAR.
- Les angles de la tête.
- L'état de détection du visage.
- Le niveau de fatigue détecté.

## 10. Déploiement avec Docker

### 10.1. Construire l'image Docker

Depuis le répertoire racine :

```powershell
docker build -t driver-monitoring-system:v1 .
```

### 10.2. Démarrer le conteneur

```powershell
docker run -d --name dms-api -p 8000:8000 driver-monitoring-system:v1 uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 10.3. Vérifier le conteneur

```powershell
docker ps
```

Vérifier l'API :

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 10.4. Lancer le client webcam

Dans un autre terminal Windows :

```powershell
python .\webcam_client.py
```

Le client communique avec l'API exécutée dans Docker via le port 8000.

### 10.5. Arrêter le conteneur

```powershell
docker stop dms-api
```

Pour le supprimer :

```powershell
docker rm dms-api
```

## 11. Tests et évaluation

Plusieurs tests ont été réalisés pour vérifier le fonctionnement des différents composants.

- Tests unitaires de la détection de fermeture des yeux.
- Tests de la détection de fatigue.
- Tests de l'estimation de la pose de la tête.
- Tests de communication entre le client et l'API.
- Vérification du traitement des images et du retour JSON.
- Tests d'intégration avec la webcam et l'API exécutée dans Docker.

Un script d'évaluation sur des images de visages aux yeux fermés est également disponible dans `src/evaluate_closed_eyes.py`.

Cette évaluation permet notamment d'étudier la détection du visage et le comportement du seuil EAR sur des exemples d'yeux fermés. Elle ne constitue pas, à elle seule, une mesure complète de classification binaire.

## 12. Limites et perspectives

### Limites actuelles

- Les seuils EAR et MAR sont basés sur des règles géométriques et peuvent varier selon les individus et les conditions de capture.
- Les performances peuvent être influencées par l'éclairage, l'orientation du visage et la qualité de la webcam.
- L'évaluation doit être approfondie sur des datasets diversifiés contenant des exemples positifs et négatifs.
- Le système constitue un prototype expérimental et ne représente pas un dispositif certifié de sécurité automobile.

### Perspectives d'amélioration

- Réaliser une évaluation quantitative sur des datasets publics.
- Mesurer les FPS, la latence et l'utilisation des ressources.
- Améliorer la robustesse aux variations d'éclairage et aux mouvements de la tête.
- Optimiser les performances du traitement en temps réel.
- Enrichir les mécanismes de détection et de fusion des indicateurs de fatigue.

## 13. Auteur et informations

**Project:** Driver Monitoring System (DMS)  
**Category:** Computer Vision · Artificial Intelligence · Real-Time Video Processing  
**Architecture:** Client-Server · FastAPI · Docker  
**Development status:** Functional experimental prototype