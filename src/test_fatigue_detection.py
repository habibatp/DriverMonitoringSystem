
import time
from fatigue_detection import FatigueDetector


def test_scenario(
    name,
    eye=False,
    yawn=False,
    yaw=0,
    pitch=0,
    roll=0,
    duration=3,
    expected=None
):
    # Un nouveau détecteur pour chaque scénario
    detector = FatigueDetector()

    print("\n" + "=" * 60)
    print(f"TEST : {name}")
    print("=" * 60)

    start_time = time.monotonic()
    result = None

    while time.monotonic() - start_time < duration:

        result = detector.update(
            prolonged_eye_closure=eye,
            yawn_detected=yawn,
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )

        print(
            f"Fatigue: {result['fatigue_level']} | "
            f"Indicateurs: {result['active_indicators']} | "
            f"Baillements: {result['yawn_count']} | "
            f"Head pose: {result['head_pose_duration']:.1f}s"
        )

        time.sleep(0.5)

    actual = result["fatigue_level"]

    print(f"\nRésultat attendu : {expected}")
    print(f"Résultat obtenu  : {actual}")

    if expected is None or actual == expected:
        print("TEST REUSSI")
    else:
        print("TEST ECHOUE")


def test_repeated_yawns():
    detector = FatigueDetector()

    print("\n" + "=" * 60)
    print("TEST : Trois bâillements répétés")
    print("=" * 60)

    for i in range(3):

        result = detector.update(
            prolonged_eye_closure=False,
            yawn_detected=True,
            yaw=0,
            pitch=0,
            roll=0
        )

        print(
            f"Bâillement {i + 1} | "
            f"Total: {result['yawn_count']} | "
            f"Fatigue: {result['fatigue_level']}"
        )

        # Fin du bâillement
        detector.update(
            prolonged_eye_closure=False,
            yawn_detected=False,
            yaw=0,
            pitch=0,
            roll=0
        )

        time.sleep(0.2)

    print("\nRésultat attendu : ATTENTION")
    print(f"Résultat obtenu : {result['fatigue_level']}")

    if result["fatigue_level"] == "ATTENTION":
        print("TEST REUSSI")
    else:
        print("TEST ECHOUE")


def test_head_pose_and_repeated_yawns():
    detector = FatigueDetector()

    print("\n" + "=" * 60)
    print("TEST : Head Pose + 3 bâillements répétés")
    print("=" * 60)

    # Simuler 3 bâillements distincts avec posture inhabituelle (pitch=25)
    for i in range(3):
        detector.update(
            prolonged_eye_closure=False,
            yawn_detected=True,
            yaw=0,
            pitch=25,
            roll=0
        )
        detector.update(
            prolonged_eye_closure=False,
            yawn_detected=False,
            yaw=0,
            pitch=25,
            roll=0
        )
        time.sleep(0.2)

    # Maintenir la posture pour que head_pose_duration >= 2.0s
    start_time = time.monotonic()
    result = None

    while time.monotonic() - start_time < 2.5:
        result = detector.update(
            prolonged_eye_closure=False,
            yawn_detected=False,
            yaw=0,
            pitch=25,
            roll=0
        )

        print(
            f"Fatigue: {result['fatigue_level']} | "
            f"Indicateurs: {result['active_indicators']} | "
            f"Baillements: {result['yawn_count']} | "
            f"Head pose: {result['head_pose_duration']:.1f}s"
        )

        time.sleep(0.5)

    actual = result["fatigue_level"]
    expected = "WARNING"

    print(f"\nRésultat attendu : {expected}")
    print(f"Résultat obtenu  : {actual}")

    if actual == expected:
        print("TEST REUSSI")
    else:
        print("TEST ECHOUE")


# ==================================================
# TEST 1 : Situation normale
# Attendu : NORMAL
# ==================================================

test_scenario(
    name="Situation normale",
    expected="NORMAL"
)


# ==================================================
# TEST 2 : Fermeture prolongée des yeux
# Attendu : ATTENTION
# ==================================================

test_scenario(
    name="Fermeture prolongee des yeux",
    eye=True,
    expected="ATTENTION"
)


# ==================================================
# TEST 3 : Head Pose prolongée
# Attendu : ATTENTION
# ==================================================

test_scenario(
    name="Posture inhabituelle prolongee",
    pitch=25,
    expected="ATTENTION"
)


# ==================================================
# TEST 4 : Trois bâillements
# Attendu : ATTENTION
# ==================================================

test_repeated_yawns()


# ==================================================
# TEST 5 : Deux indicateurs sans fermeture des yeux
# Attendu : WARNING
# ==================================================

test_head_pose_and_repeated_yawns()


# ==================================================
# TEST 6 : Fermeture des yeux + Head Pose
# Attendu : HIGH_RISK
# ==================================================

test_scenario(
    name="Fermeture des yeux + Head Pose",
    eye=True,
    yaw=40,
    expected="HIGH_RISK"
)


print("\n" + "=" * 60)
print("TOUS LES SCENARIOS SONT TERMINES")
print("=" * 60)
