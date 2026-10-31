
import cv2
import mediapipe as mp
import time
from pathlib import Path

from face_detection import (
    LEFT_EYE,
    RIGHT_EYE,
    get_eye_points,
    calculate_ear
)

from eye_detection import EyeClosureDetector


# --------------------------------------------------
# 1. Configuration du modèle MediaPipe
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_landmarker.task"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


# --------------------------------------------------
# 2. Initialisation du détecteur de fermeture
# --------------------------------------------------

eye_detector = EyeClosureDetector(
    close_threshold=0.20,
    open_threshold=0.25,
    min_closed_duration=1.5,
    smoothing_window=5
)


# --------------------------------------------------
# 3. Initialisation de la webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")


# --------------------------------------------------
# 4. Boucle principale
# --------------------------------------------------

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Erreur lors de la lecture de la webcam.")
                break

            # Inverser horizontalement l'image
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convertir BGR vers RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp requis par MediaPipe en mode VIDEO
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Détection du visage
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]

                # Extraire les points des yeux
                left_eye = get_eye_points(
                    landmarks,
                    LEFT_EYE,
                    width,
                    height
                )

                right_eye = get_eye_points(
                    landmarks,
                    RIGHT_EYE,
                    width,
                    height
                )

                # Calculer les EAR
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)

                average_ear = (
                    left_ear + right_ear
                ) / 2.0

                # Mettre à jour le détecteur
                eye_status = eye_detector.update(average_ear)

                # Afficher les points des yeux
                for point in left_eye + right_eye:
                    cv2.circle(
                        frame,
                        point,
                        2,
                        (0, 255, 255),
                        -1
                    )

                # Récupérer les résultats
                smoothed_ear = eye_status["smoothed_ear"]
                closure_duration = eye_status["closure_duration"]
                eyes_closed = eye_status["eyes_closed"]
                prolonged = eye_status["prolonged_closure"]

                # Affichage des valeurs EAR
                cv2.putText(
                    frame,
                    f"Left EAR: {left_ear:.3f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right EAR: {right_ear:.3f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Average EAR: {average_ear:.3f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Smoothed EAR: {smoothed_ear:.3f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # État des yeux
                eye_color = (
                    (0, 0, 255)
                    if eyes_closed
                    else (0, 255, 0)
                )

                eye_label = (
                    "EYES CLOSED"
                    if eyes_closed
                    else "EYES OPEN"
                )

                cv2.putText(
                    frame,
                    eye_label,
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    eye_color,
                    2
                )

                # Durée de fermeture
                cv2.putText(
                    frame,
                    f"Closure: {closure_duration:.2f}s",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Alerte de fermeture prolongée
                if prolonged:

                    cv2.putText(
                        frame,
                        "PROLONGED EYE CLOSURE!",
                        (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            else:

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Afficher la fenêtre
            cv2.imshow(
                "DMS - Eye Closure Webcam Test",
                frame
            )

            # Quitter avec la touche Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Webcam test stopped.")
