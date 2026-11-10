
import os
import cv2
import mediapipe as mp
import time

try:
    import winsound
except ImportError:
    winsound = None

from pathlib import Path

from yawn_detection import YawnDetector
from eye_detection import EyeClosureDetector
from head_pose import estimate_head_pose, calculate_relative_pose
from fatigue_detection import FatigueDetector

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
# 1. Alertes sonores
# ==========================================

def play_alert(frequency, duration):
    if winsound is not None:
        winsound.Beep(frequency, duration)
    else:
        print(
            f"ALERTE : fréquence={frequency} Hz, "
            f"durée={duration} ms"
        )


# ==========================================
# 2. Fonction d'affichage
# ==========================================

def display_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.55
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


# ==========================================
# 3. Fonction principale
# ==========================================

def main():

    # --------------------------------------
    # Chemin du modèle
    # --------------------------------------

    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "face_landmarker.task"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}"
        )

    # --------------------------------------
    # Configuration MediaPipe
    # --------------------------------------

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1
    )

    # --------------------------------------
    # Initialisation des détecteurs
    # --------------------------------------

    yawn_detector = YawnDetector(
        mar_threshold=0.12,
        min_duration=1.5
    )

    eye_detector = EyeClosureDetector(
        close_threshold=0.20,
        open_threshold=0.25,
        min_closed_duration=1.5,
        smoothing_window=5
    )

    fatigue_detector = FatigueDetector()

    # --------------------------------------
    # Configuration des alertes
    # --------------------------------------

    last_alert_time = 0
    alert_cooldown = 3.0

    # --------------------------------------
    # Calibration Head Pose
    # --------------------------------------

    reference_rotation = None
    current_rotation = None

    # --------------------------------------
    # Initialisation webcam
    # --------------------------------------

    camera_backend = (
        cv2.CAP_DSHOW
        if os.name == "nt"
        else cv2.CAP_ANY
    )

    cap = cv2.VideoCapture(0, camera_backend)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Impossible d'ouvrir la webcam."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --------------------------------------
    # FPS
    # --------------------------------------

    previous_time = time.perf_counter()
    fps = 0.0

    print("Driver Monitoring System started.")
    print("C : Calibrer Head Pose")
    print("R : Reinitialiser calibration")
    print("Q : Quitter")

    try:

        with FaceLandmarker.create_from_options(options) as landmarker:

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    print("Impossible de lire l'image.")
                    break

                # Image miroir
                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                current_rotation = None

                # Conversion BGR vers RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = (
                    time.monotonic_ns() // 1_000_000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # ==================================
                # Détection du visage
                # ==================================

                if result.face_landmarks:

                    landmarks = result.face_landmarks[0]

                    # ==================================
                    # A. EYE DETECTION
                    # ==================================

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

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)

                    average_ear = (
                        left_ear + right_ear
                    ) / 2.0

                    eye_status = eye_detector.update(
                        average_ear
                    )

                    eyes_closed = eye_status["eyes_closed"]
                    prolonged_closure = eye_status["prolonged_closure"]
                    closure_duration = eye_status["closure_duration"]

                    # ==================================
                    # B. YAWN DETECTION
                    # ==================================

                    mouth = get_mouth_points(
                        landmarks,
                        width,
                        height
                    )

                    mar = calculate_mar(mouth)

                    yawn_detected, mouth_duration = (
                        yawn_detector.update(mar)
                    )

                    # ==================================
                    # C. HEAD POSE
                    # ==================================

                    pose = estimate_head_pose(
                        landmarks,
                        width,
                        height
                    )

                    yaw = 0.0
                    pitch = 0.0
                    roll = 0.0

                    pose_calibrated = False

                    if pose is not None:

                        current_rotation = pose["rotation_matrix"]

                        if reference_rotation is not None:

                            relative_pose = calculate_relative_pose(
                                current_rotation,
                                reference_rotation
                            )

                            if relative_pose is not None:

                                yaw = relative_pose["yaw"]
                                pitch = relative_pose["pitch"]
                                roll = relative_pose["roll"]

                                pose_calibrated = True

                    # ==================================
                    # D. FATIGUE DETECTION
                    # ==================================

                    fatigue_result = fatigue_detector.update(
                        prolonged_eye_closure=prolonged_closure,
                        yawn_detected=yawn_detected,
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll
                    )

                    fatigue_level = fatigue_result["fatigue_level"]
                    active_indicators = fatigue_result["active_indicators"]
                    yawn_count = fatigue_result["yawn_count"]
                    head_pose_duration = fatigue_result["head_pose_duration"]

                    # ==================================
                    # E. ALERTES SONORES
                    # ==================================

                    current_alert_time = time.monotonic()

                    if fatigue_level == "ATTENTION":

                        if current_alert_time - last_alert_time >= alert_cooldown:
                            play_alert(1000, 500)
                            last_alert_time = current_alert_time

                    elif fatigue_level == "WARNING":

                        if current_alert_time - last_alert_time >= alert_cooldown:
                            play_alert(1500, 700)
                            last_alert_time = current_alert_time

                    elif fatigue_level == "HIGH_RISK":

                        if current_alert_time - last_alert_time >= alert_cooldown:
                            play_alert(2000, 1000)
                            last_alert_time = current_alert_time

                    # ==================================
                    # F. LANDMARKS
                    # ==================================

                    draw_eye_landmarks(frame, left_eye)
                    draw_eye_landmarks(frame, right_eye)
                    draw_mouth_landmarks(frame, mouth)

                    # ==================================
                    # G. AFFICHAGE EYES
                    # ==================================

                    eye_color = (
                        (0, 0, 255) if eyes_closed
                        else (0, 255, 0)
                    )

                    display_text(
                        frame,
                        f"Eyes: {'CLOSED' if eyes_closed else 'OPEN'}",
                        (20, 35),
                        eye_color
                    )

                    display_text(
                        frame,
                        f"EAR: {average_ear:.3f}",
                        (20, 65)
                    )

                    display_text(
                        frame,
                        f"Closure: {closure_duration:.1f}s",
                        (20, 95)
                    )

                    if prolonged_closure:

                        display_text(
                            frame,
                            "PROLONGED EYE CLOSURE",
                            (20, 125),
                            (0, 0, 255)
                        )

                    # ==================================
                    # H. AFFICHAGE YAWN
                    # ==================================

                    display_text(
                        frame,
                        f"MAR: {mar:.3f}",
                        (20, 165)
                    )

                    display_text(
                        frame,
                        f"Yawns: {yawn_count}",
                        (20, 195)
                    )

                    if yawn_detected:

                        display_text(
                            frame,
                            "POTENTIAL YAWN",
                            (20, 225),
                            (0, 165, 255)
                        )

                    # ==================================
                    # I. AFFICHAGE HEAD POSE
                    # ==================================

                    if pose_calibrated:

                        display_text(
                            frame,
                            f"Yaw: {yaw:.1f}",
                            (350, 35)
                        )

                        display_text(
                            frame,
                            f"Pitch: {pitch:.1f}",
                            (350, 65)
                        )

                        display_text(
                            frame,
                            f"Roll: {roll:.1f}",
                            (350, 95)
                        )

                        display_text(
                            frame,
                            f"Head pose: {head_pose_duration:.1f}s",
                            (350, 125)
                        )

                    else:

                        display_text(
                            frame,
                            "Press C to calibrate",
                            (350, 35),
                            (0, 255, 255)
                        )

                    # ==================================
                    # J. AFFICHAGE FATIGUE
                    # ==================================

                    if fatigue_level == "NORMAL":
                        fatigue_color = (0, 255, 0)

                    elif fatigue_level == "ATTENTION":
                        fatigue_color = (0, 255, 255)

                    elif fatigue_level == "WARNING":
                        fatigue_color = (0, 165, 255)

                    else:
                        fatigue_color = (0, 0, 255)

                    display_text(
                        frame,
                        f"FATIGUE: {fatigue_level}",
                        (20, 285),
                        fatigue_color,
                        0.8
                    )

                    display_text(
                        frame,
                        f"Active indicators: {active_indicators}/3",
                        (20, 320)
                    )

                else:

                    # Aucun visage détecté
                    display_text(
                        frame,
                        "NO FACE DETECTED",
                        (20, 40),
                        (0, 0, 255),
                        0.7
                    )

                    # Réinitialisation Head Pose
                    fatigue_result = fatigue_detector.update(
                        prolonged_eye_closure=False,
                        yawn_detected=False,
                        yaw=0,
                        pitch=0,
                        roll=0
                    )

                    display_text(
                        frame,
                        f"FATIGUE: {fatigue_result['fatigue_level']}",
                        (20, 80),
                        (255, 255, 255)
                    )

                # ==================================
                # K. FPS
                # ==================================

                current_time = time.perf_counter()
                elapsed = current_time - previous_time

                if elapsed > 0:
                    fps = 1.0 / elapsed

                previous_time = current_time

                display_text(
                    frame,
                    f"FPS: {fps:.1f}",
                    (width - 130, height - 20),
                    (255, 255, 0)
                )

                # ==================================
                # L. AFFICHAGE WEBCAM
                # ==================================

                cv2.imshow(
                    "Driver Monitoring System",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                # Calibration position neutre
                if key == ord("c"):

                    if current_rotation is not None:

                        reference_rotation = current_rotation.copy()
                        print("Head pose calibrated.")

                    else:

                        print(
                            "No face detected. Calibration impossible."
                        )

                # Réinitialisation calibration
                elif key == ord("r"):

                    reference_rotation = None
                    print("Head pose calibration reset.")

                # Quitter
                elif key == ord("q"):
                    break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("Driver Monitoring System stopped.")


# ==========================================
# 4. Point d'entrée du programme
# ==========================================

if __name__ == "__main__":
    main()
