import cv2
import numpy as np
import requests
import base64

# 1. Créer une image de test
frame = np.zeros((480, 640, 3), dtype=np.uint8)

cv2.putText(
    frame, "DMS API TEST", (150, 240),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
)

# 2. Encoder en JPEG
success, encoded = cv2.imencode(".jpg", frame)

if not success:
    raise RuntimeError("Erreur d'encodage JPEG")

# 3. Envoyer à l'API
response = requests.post(
    "http://127.0.0.1:8000/process-frame",
    data=encoded.tobytes(),
    headers={"Content-Type": "image/jpeg"},
    timeout=30
)

print("Code HTTP :", response.status_code)
print("Type de réponse :", response.headers.get("Content-Type"))

response.raise_for_status()

# 4. Récupérer le JSON
result = response.json()

# 5. Afficher les métriques
print("\n--- DMS METRICS ---")

for key, value in result["metrics"].items():
    print(f"{key}: {value}")

# 6. Décoder l'image Base64
image_bytes = base64.b64decode(result["image"])

image_array = np.frombuffer(image_bytes, dtype=np.uint8)

received_frame = cv2.imdecode(
    image_array,
    cv2.IMREAD_COLOR
)

if received_frame is None:
    raise RuntimeError("Image retournée invalide")

# 7. Enregistrer le résultat
cv2.imwrite("result.jpg", received_frame)

print("\nDimensions :", received_frame.shape)
print("Image enregistrée : result.jpg")
print("Traitement API réussi !")
