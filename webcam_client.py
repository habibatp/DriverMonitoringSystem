
import cv2
import requests
import numpy as np
import base64
import time

try:
    import winsound
except ImportError:
    winsound = None

API_URL = "http://127.0.0.1:8000"
PROCESS_URL = f"{API_URL}/process-frame"

# Ouvrir la webcam Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

last_alert_time = 0
alert_cooldown = 3

print("Webcam connectée à l'API DMS")
print("C : calibrer le Head Pose")
print("R : réinitialiser la calibration")
print("Q : quitter")

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("Erreur de capture webcam")
            break

        # Réduire la résolution pour limiter le transfert
        frame = cv2.resize(frame, (640, 480))

        # Encoder l'image
        encoded_success, encoded_image = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not encoded_success:
            continue

        try:
            # Envoyer l'image à l'API
            response = requests.post(
                PROCESS_URL,
                data=encoded_image.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=5
            )

            response.raise_for_status()
            result = response.json()

            # Récupérer les métriques
            metrics = result["metrics"]

            # Décoder l'image annotée
            image_bytes = base64.b64decode(result["image"])
            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            processed_frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if processed_frame is None:
                print("Image reçue invalide")
                continue

            # Alertes sonores locales
            fatigue_level = metrics["fatigue_level"]
            current_time = time.monotonic()

            if (
                fatigue_level in ["ATTENTION", "WARNING", "HIGH_RISK"]
                and winsound is not None
                and current_time - last_alert_time >= alert_cooldown
            ):

                frequencies = {
                    "ATTENTION": (1000, 500),
                    "WARNING": (1500, 700),
                    "HIGH_RISK": (2000, 1000)
                }

                frequency, duration = frequencies[fatigue_level]

                winsound.Beep(frequency, duration)
                last_alert_time = time.monotonic()

            # Afficher le niveau de fatigue
            cv2.imshow("Driver Monitoring System - API", processed_frame)

        except requests.RequestException as error:
            print("Erreur API :", error)
            break

        except (KeyError, ValueError) as error:
            print("Erreur de traitement de la réponse :", error)
            break

        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            try:
                calibration = requests.post(
                    f"{API_URL}/calibrate",
                    timeout=5
                )
                print("Calibration :", calibration.status_code,
                      calibration.text)
            except requests.RequestException as error:
                print("Erreur calibration :", error)

        elif key == ord("r"):
            try:
                reset = requests.post(
                    f"{API_URL}/reset-calibration",
                    timeout=5
                )
                print("Reset :", reset.status_code, reset.text)
            except requests.RequestException as error:
                print("Erreur reset :", error)

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Client webcam arrêté")
