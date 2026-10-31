
import time

from eye_detection import EyeClosureDetector


# ==========================================
# Initialisation du détecteur
# ==========================================

detector = EyeClosureDetector(
    close_threshold=0.20,
    open_threshold=0.25,
    min_closed_duration=1.5,
    smoothing_window=5
)


# ==========================================
# Fonction de simulation
# ==========================================

def simulate_ear(ear_value, duration, label):

    print(f"\n--- {label} ---")

    start_time = time.monotonic()

    while time.monotonic() - start_time < duration:

        result = detector.update(ear_value)

        print(
            f"EAR: {result['smoothed_ear']:.3f} | "
            f"Closed: {result['eyes_closed']} | "
            f"Duration: {result['closure_duration']:.2f}s | "
            f"Prolonged: {result['prolonged_closure']}"
        )

        time.sleep(0.1)


# ==========================================
# TEST 1 : Yeux ouverts
# ==========================================

simulate_ear(
    ear_value=0.40,
    duration=2.0,
    label="TEST 1 - EYES OPEN"
)


# ==========================================
# TEST 2 : Clignement bref
# ==========================================

simulate_ear(
    ear_value=0.10,
    duration=0.3,
    label="TEST 2 - SHORT BLINK"
)

simulate_ear(
    ear_value=0.40,
    duration=1.0,
    label="TEST 2 - EYES REOPENED"
)


# ==========================================
# TEST 3 : Fermeture prolongée
# ==========================================

simulate_ear(
    ear_value=0.10,
    duration=2.0,
    label="TEST 3 - PROLONGED CLOSURE"
)


# ==========================================
# TEST 4 : Réouverture
# ==========================================

simulate_ear(
    ear_value=0.40,
    duration=1.0,
    label="TEST 4 - EYES REOPENED"
)


print("\nAll tests completed.")
