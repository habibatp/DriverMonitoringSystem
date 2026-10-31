
import cv2
import mediapipe as mp
import time

from pathlib import Path

from yawn_detection import YawnDetector
from eye_detection import EyeClosureDetector

from face_detection import (
    LEFT_EYE,
    RIGHT_EYE,
    get_eye_points,
    get_mouth_points,
    draw_eye_landmarks,
    draw_mouth_landmarks,
    calculate_mar,
    calculate_ear
)


# ==========================================
# Chemin du modèle
# ==========================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_landmarker.task"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )


# ==========================================
# Configuration MediaPipe
# ==========================================

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


# ==========================================
# Initialisation webcam
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ==========================================
# Initialisation FPS
# ==========================================

previous_time = time.perf_counter()
fps = 0.0


# ==========================================
# Initialisation détection bâillements
# ==========================================

yawn_detector = YawnDetector(
    mar_threshold=0.12,
    min_duration=1.5
)


# ==========================================
# Initialisation détection fermeture des yeux
# ==========================================

eye_detector = EyeClosureDetector(
    close_threshold=0.20,
    open_threshold=0.25,
    min_closed_duration=1.5,
    smoothing_window=5
)


# ==========================================
# Boucle principale
# ==========================================

try:

    with FaceLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Impossible de lire l'image.")
                break

            # Inversion horizontale de l'image
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Conversion BGR vers RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp vidéo
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Détection des landmarks
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ==================================
            # Détection du visage
            # ==================================

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # Extraction des points des yeux
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

                # ==================================
                # Calcul EAR
                # ==================================

                left_ear = calculate_ear(left_eye)

                right_ear = calculate_ear(right_eye)

                average_ear = (
                    left_ear + right_ear
                ) / 2.0

                # Détection de fermeture prolongée
                eye_status = eye_detector.update(average_ear)

                eyes_closed = eye_status["eyes_closed"]
                smoothed_ear = eye_status["smoothed_ear"]
                closure_duration = eye_status["closure_duration"]
                prolonged_closure = eye_status["prolonged_closure"]

                # ==================================
                # Extraction des points de la bouche
                # ==================================

                mouth = get_mouth_points(
                    landmarks,
                    width,
                    height
                )

                # Calcul du MAR
                mar = calculate_mar(mouth)

                # Détection des bâillements
                yawn_detected, mouth_duration = (
                    yawn_detector.update(mar)
                )

                # ==================================
                # Visualisation des landmarks
                # ==================================

                draw_eye_landmarks(frame, left_eye)
                draw_eye_landmarks(frame, right_eye)
                draw_mouth_landmarks(frame, mouth)

                # ==================================
                # Affichage détection visage
                # ==================================

                cv2.putText(
                    frame,
                    "Facial features detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # ==================================
                # Affichage FPS
                # ==================================

                # Affichage MAR
                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Mouth duration: {mouth_duration:.1f}s",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # ==================================
                # Affichage EAR
                # ==================================

                cv2.putText(
                    frame,
                    f"Left EAR: {left_ear:.3f}",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right EAR: {right_ear:.3f}",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Average EAR: {average_ear:.3f}",
                    (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # ==================================
                # Affichage détection fermeture des yeux
                # ==================================

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
                    (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    eye_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Eye closure: {closure_duration:.2f}s",
                    (20, 345),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Smoothed EAR: {smoothed_ear:.3f}",
                    (20, 375),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Alerte fermeture prolongée
                if prolonged_closure:

                    cv2.putText(
                        frame,
                        "PROLONGED EYE CLOSURE!",
                        (20, 410),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # ==================================
                # Affichage alerte bâillement
                # ==================================

                if yawn_detected:

                    cv2.putText(
                        frame,
                        "POTENTIAL YAWN DETECTED",
                        (20, 445),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            else:

                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # ==================================
            # Calcul FPS
            # ==================================

            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            if elapsed > 0:
                fps = 1.0 / elapsed

            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # ==================================
            # Affichage webcam
            # ==================================

            cv2.imshow(
                "DMS - Facial Features Extraction",
                frame
            )

            # Quitter avec la touche Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

