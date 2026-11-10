
import time
from pathlib import Path

import cv2
import mediapipe as mp

from src.yawn_detection import YawnDetector
from src.eye_detection import EyeClosureDetector
from src.head_pose import estimate_head_pose, calculate_relative_pose
from src.fatigue_detection import FatigueDetector

from src.face_detection import (
    LEFT_EYE,
    RIGHT_EYE,
    get_eye_points,
    get_mouth_points,
    draw_eye_landmarks,
    draw_mouth_landmarks,
    calculate_mar,
    calculate_ear
)


class DMSProcessor:

    def __init__(self):

        # Chemin du modèle
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "face_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele introuvable : {model_path}"
            )

        # Configuration MediaPipe
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

        # Détecteurs persistants
        self.yawn_detector = YawnDetector(
            mar_threshold=0.12,
            min_duration=1.5
        )

        self.eye_detector = EyeClosureDetector(
            close_threshold=0.20,
            open_threshold=0.25,
            min_closed_duration=1.5,
            smoothing_window=5
        )

        self.fatigue_detector = FatigueDetector()

        # Calibration
        self.reference_rotation = None
        self.current_rotation = None

        # Timestamp MediaPipe
        self.last_timestamp_ms = -1

    def calibrate(self):

        if self.current_rotation is None:
            return False

        self.reference_rotation = self.current_rotation.copy()
        return True

    def reset_calibration(self):

        self.reference_rotation = None

    def display_text(
        self,
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

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        # Conversion BGR vers RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp strictement croissant
        timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        results = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        metrics = {
            "face_detected": False,
            "eyes_closed": False,
            "prolonged_eye_closure": False,
            "ear": None,
            "mar": None,
            "yawn_detected": False,
            "yawn_count": 0,
            "yaw": None,
            "pitch": None,
            "roll": None,
            "pose_calibrated": False,
            "fatigue_level": "NORMAL",
            "active_indicators": 0
        }

        if results.face_landmarks:

            landmarks = results.face_landmarks[0]
            metrics["face_detected"] = True

            # A. Détection des yeux
            left_eye = get_eye_points(
                landmarks, LEFT_EYE, width, height
            )

            right_eye = get_eye_points(
                landmarks, RIGHT_EYE, width, height
            )

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            average_ear = (left_ear + right_ear) / 2.0

            eye_status = self.eye_detector.update(average_ear)

            eyes_closed = eye_status["eyes_closed"]
            prolonged_closure = eye_status["prolonged_closure"]
            closure_duration = eye_status["closure_duration"]

            metrics["ear"] = float(average_ear)
            metrics["eyes_closed"] = eyes_closed
            metrics["prolonged_eye_closure"] = prolonged_closure
            metrics["closure_duration"] = float(closure_duration)

            # B. Détection des bâillements
            mouth = get_mouth_points(landmarks, width, height)
            mar = calculate_mar(mouth)

            yawn_detected, mouth_duration = (
                self.yawn_detector.update(mar)
            )

            metrics["mar"] = float(mar)
            metrics["yawn_detected"] = yawn_detected
            metrics["mouth_duration"] = float(mouth_duration)

            # C. Head Pose
            pose = estimate_head_pose(
                landmarks, width, height
            )

            yaw = 0.0
            pitch = 0.0
            roll = 0.0
            pose_calibrated = False

            if pose is not None:

                self.current_rotation = pose["rotation_matrix"]

                if self.reference_rotation is not None:

                    relative_pose = calculate_relative_pose(
                        self.current_rotation,
                        self.reference_rotation
                    )

                    if relative_pose is not None:

                        yaw = relative_pose["yaw"]
                        pitch = relative_pose["pitch"]
                        roll = relative_pose["roll"]
                        pose_calibrated = True

            metrics["pose_calibrated"] = pose_calibrated

            if pose_calibrated:
                metrics["yaw"] = float(yaw)
                metrics["pitch"] = float(pitch)
                metrics["roll"] = float(roll)

            # D. Détection de fatigue
            fatigue_result = self.fatigue_detector.update(
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

            metrics["fatigue_level"] = fatigue_level
            metrics["active_indicators"] = active_indicators
            metrics["yawn_count"] = yawn_count
            metrics["head_pose_duration"] = float(head_pose_duration)

            # E. Affichage des landmarks
            draw_eye_landmarks(frame, left_eye)
            draw_eye_landmarks(frame, right_eye)
            draw_mouth_landmarks(frame, mouth)

            # F. Affichage des yeux
            eye_color = (
                (0, 0, 255) if eyes_closed
                else (0, 255, 0)
            )

            self.display_text(
                frame,
                f"Eyes: {'CLOSED' if eyes_closed else 'OPEN'}",
                (20, 35),
                eye_color
            )

            self.display_text(
                frame,
                f"EAR: {average_ear:.3f}",
                (20, 65)
            )

            self.display_text(
                frame,
                f"Closure: {closure_duration:.1f}s",
                (20, 95)
            )

            if prolonged_closure:
                self.display_text(
                    frame,
                    "PROLONGED EYE CLOSURE",
                    (20, 125),
                    (0, 0, 255)
                )

            # G. Affichage des bâillements
            self.display_text(
                frame,
                f"MAR: {mar:.3f}",
                (20, 165)
            )

            self.display_text(
                frame,
                f"Yawns: {yawn_count}",
                (20, 195)
            )

            if yawn_detected:
                self.display_text(
                    frame,
                    "POTENTIAL YAWN",
                    (20, 225),
                    (0, 165, 255)
                )

            # H. Affichage Head Pose
            if pose_calibrated:

                self.display_text(frame, f"Yaw: {yaw:.1f}", (350, 35))
                self.display_text(frame, f"Pitch: {pitch:.1f}", (350, 65))
                self.display_text(frame, f"Roll: {roll:.1f}", (350, 95))

                self.display_text(
                    frame,
                    f"Head pose: {head_pose_duration:.1f}s",
                    (350, 125)
                )

            else:

                self.display_text(
                    frame,
                    "Press C to calibrate",
                    (350, 35),
                    (0, 255, 255)
                )

            # I. Affichage fatigue
            fatigue_colors = {
                "NORMAL": (0, 255, 0),
                "ATTENTION": (0, 255, 255),
                "WARNING": (0, 165, 255),
                "HIGH_RISK": (0, 0, 255)
            }

            fatigue_color = fatigue_colors.get(
                fatigue_level, (255, 255, 255)
            )

            self.display_text(
                frame,
                f"FATIGUE: {fatigue_level}",
                (20, 285),
                fatigue_color,
                0.8
            )

            self.display_text(
                frame,
                f"Active indicators: {active_indicators}/3",
                (20, 320)
            )

        else:

            self.display_text(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                (0, 0, 255),
                0.7
            )

            # Réinitialisation de l'état de fatigue
            fatigue_result = self.fatigue_detector.update(
                prolonged_eye_closure=False,
                yawn_detected=False,
                yaw=0,
                pitch=0,
                roll=0
            )

            metrics["fatigue_level"] = fatigue_result["fatigue_level"]
            metrics["active_indicators"] = (
                fatigue_result["active_indicators"]
            )

        return frame, metrics

    def close(self):
        self.landmarker.close()
