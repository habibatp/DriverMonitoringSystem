
import time
from collections import deque


class FatigueDetector:

    def __init__(
        self,
        yaw_threshold=35.0,
        pitch_threshold=20.0,
        roll_threshold=25.0,
        head_pose_duration=2.0,
        yawn_window=60.0,
        yawn_count_threshold=3
    ):

        # Seuils Head Pose (degrés)
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold

        # Durée minimale d'une posture inhabituelle
        self.head_pose_duration = head_pose_duration

        # Paramètres de détection des bâillements répétés
        self.yawn_window = yawn_window
        self.yawn_count_threshold = yawn_count_threshold

        # Historique des bâillements
        self.yawn_history = deque()

        # Gestion temporelle Head Pose
        self.head_pose_start_time = None

        # Pour détecter un nouvel événement de bâillement
        self.previous_yawn_state = False

    def update(
        self,
        prolonged_eye_closure,
        yawn_detected,
        yaw,
        pitch,
        roll
    ):

        current_time = time.monotonic()

        # ==========================================
        # 1. Détection des événements Head Pose
        # ==========================================

        abnormal_head_pose = (
            abs(yaw) > self.yaw_threshold
            or abs(pitch) > self.pitch_threshold
            or abs(roll) > self.roll_threshold
        )

        if abnormal_head_pose:

            if self.head_pose_start_time is None:
                self.head_pose_start_time = current_time

            head_pose_duration = (
                current_time - self.head_pose_start_time
            )

        else:

            self.head_pose_start_time = None
            head_pose_duration = 0.0

        prolonged_head_pose = (
            abnormal_head_pose
            and head_pose_duration >= self.head_pose_duration
        )

        # ==========================================
        # 2. Comptage des bâillements
        # ==========================================

        # Compter uniquement le début d'un nouvel événement
        if yawn_detected and not self.previous_yawn_state:
            self.yawn_history.append(current_time)

        self.previous_yawn_state = yawn_detected

        # Supprimer les événements trop anciens
        while (
            self.yawn_history
            and current_time - self.yawn_history[0] > self.yawn_window
        ):
            self.yawn_history.popleft()

        yawn_count = len(self.yawn_history)

        repeated_yawns = (
            yawn_count >= self.yawn_count_threshold
        )

        # ==========================================
        # 3. Fusion des indicateurs
        # ==========================================

        indicators = {
            "prolonged_eye_closure": prolonged_eye_closure,
            "repeated_yawns": repeated_yawns,
            "prolonged_head_pose": prolonged_head_pose
        }

        active_indicators = sum(indicators.values())

        # ==========================================
        # 4. Estimation globale
        # ==========================================

        if prolonged_eye_closure and active_indicators >= 2:

            fatigue_level = "HIGH_RISK"

        elif active_indicators >= 2:

            fatigue_level = "WARNING"

        elif active_indicators == 1:

            fatigue_level = "ATTENTION"

        else:

            fatigue_level = "NORMAL"

        # ==========================================
        # 5. Résultats
        # ==========================================

        return {
            "fatigue_level": fatigue_level,
            "active_indicators": active_indicators,
            "indicators": indicators,
            "yawn_count": yawn_count,
            "head_pose_duration": head_pose_duration
        }
