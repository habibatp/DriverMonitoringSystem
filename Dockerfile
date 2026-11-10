
# 1. Utiliser Python 3.10 comme image de base
FROM python:3.10-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Installer les bibliothèques système nécessaires à OpenCV et MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier le fichier des dépendances
COPY requirements.txt .

# 5. Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier le code source
COPY src/ ./src/

# 7. Copier le modèle MediaPipe
COPY models/ ./models/

# 8. Définir la commande de démarrage
CMD ["python", "src/main.py"]
