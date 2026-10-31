
import time
from collections import deque


class YawnDetector:

    def __init__(
        self,
        mar_threshold=0.12,
        close_threshold=0.08,
        min_duration=1.5,
        alert_duration=2.0,
        smoothing_window=5
    ):
        self.mar_threshold = mar_threshold
        self.close_threshold = close_threshold
        self.min_duration = min_duration
        self.alert_duration = alert_duration

        # Historique des valeurs MAR
        self.mar_history = deque(maxlen=smoothing_window)

        # Etat de la bouche
        self.mouth_open = False
        self.open_start_time = None

        # Gestion de l'alerte
        self.alert_until = 0.0

    def update(self, mar):

        current_time = time.monotonic()

        # 1. Ajouter la nouvelle valeur MAR
        self.mar_history.append(mar)

        # 2. Calculer le MAR moyen
        smoothed_mar = (
            sum(self.mar_history) / len(self.mar_history)
        )

        # 3. Détection avec hystérésis
        if not self.mouth_open:

            # Déclencher l'ouverture
            if smoothed_mar >= self.mar_threshold:

                self.mouth_open = True
                self.open_start_time = current_time

        else:

            # Détecter la fermeture uniquement
            # lorsque le MAR descend sous le seuil inférieur
            if smoothed_mar <= self.close_threshold:

                duration = current_time - self.open_start_time

                if duration >= self.min_duration:
                    self.alert_until = (
                        current_time + self.alert_duration
                    )

                self.mouth_open = False
                self.open_start_time = None

        # 4. Calculer la durée d'ouverture
        if self.mouth_open:
            duration = current_time - self.open_start_time
        else:
            duration = 0.0

        # 5. Maintenir l'alerte pendant 2 secondes
        yawn_detected = current_time < self.alert_until

        return yawn_detected, duration
