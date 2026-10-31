
import time
from collections import deque


class EyeClosureDetector:

    def __init__(
        self,
        close_threshold=0.20,
        open_threshold=0.25,
        min_closed_duration=1.5,
        smoothing_window=5
    ):

        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_closed_duration = min_closed_duration

        # Historique des valeurs EAR
        self.ear_history = deque(maxlen=smoothing_window)

        # État des yeux
        self.eyes_closed = False
        self.closure_start_time = None

        # Durée de la dernière fermeture
        self.last_closure_duration = 0.0

    def update(self, ear):

        current_time = time.monotonic()

        # 1. Ajouter la nouvelle valeur EAR
        self.ear_history.append(ear)

        # 2. Calculer l'EAR moyen
        smoothed_ear = (
            sum(self.ear_history) / len(self.ear_history)
        )

        # 3. Détection avec hystérésis
        if not self.eyes_closed:

            # Détecter une fermeture potentielle
            if smoothed_ear <= self.close_threshold:

                self.eyes_closed = True
                self.closure_start_time = current_time

        else:

            # Détecter la réouverture
            if smoothed_ear >= self.open_threshold:

                duration = (
                    current_time - self.closure_start_time
                )

                self.last_closure_duration = duration

                self.eyes_closed = False
                self.closure_start_time = None

        # 4. Calculer la durée de fermeture actuelle
        if self.eyes_closed:

            closure_duration = (
                current_time - self.closure_start_time
            )

        else:

            closure_duration = 0.0

        # 5. Distinguer clignement et fermeture prolongée
        prolonged_closure = (
            self.eyes_closed
            and closure_duration >= self.min_closed_duration
        )

        # 6. Retourner les informations
        return {
            "eyes_closed": self.eyes_closed,
            "smoothed_ear": smoothed_ear,
            "closure_duration": closure_duration,
            "last_closure_duration": self.last_closure_duration,
            "prolonged_closure": prolonged_closure
        }
