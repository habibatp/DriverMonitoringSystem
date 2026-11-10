
import cv2
import mediapipe as mp
import time
from pathlib import Path

from head_pose import (
    estimate_head_pose,
    calculate_relative_pose
)


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
    num_faces=1,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    min_tracking_confidence=0.4
)


# --------------------------------------------------
# 2. Initialisation de la webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")


# --------------------------------------------------
# 3. Calibration
# --------------------------------------------------

reference_rotation = None


# --------------------------------------------------
# 4. Boucle principale
# --------------------------------------------------

try:

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Erreur lors de la lecture webcam.")
                break

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000

            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            current_rotation = None

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]

                pose = estimate_head_pose(
                    landmarks,
                    width,
                    height
                )

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

                            cv2.putText(
                                frame,
                                f"Yaw: {yaw:.2f} deg",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )

                            cv2.putText(
                                frame,
                                f"Pitch: {pitch:.2f} deg",
                                (20, 75),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )

                            cv2.putText(
                                frame,
                                f"Roll: {roll:.2f} deg",
                                (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )

                    else:

                        cv2.putText(
                            frame,
                            "Press C to calibrate",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2
                        )

                else:

                    cv2.putText(
                        frame,
                        "HEAD POSE ESTIMATION FAILED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
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

            # Calibration status
            if reference_rotation is not None:

                cv2.putText(
                    frame,
                    "CALIBRATED",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NOT CALIBRATED",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

            # Instructions
            cv2.putText(
                frame,
                "C: Calibrate | R: Reset | Q: Quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "DMS - Head Pose Estimation Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and current_rotation is not None:

                reference_rotation = current_rotation.copy()
                print("Head pose calibrated.")

            elif key == ord("r"):

                reference_rotation = None
                print("Calibration reset.")

            elif key == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Head pose test stopped.")
