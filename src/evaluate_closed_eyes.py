
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from pathlib import Path

from face_detection import (
    LEFT_EYE,
    RIGHT_EYE,
    get_eye_points,
    calculate_ear
)

# ==========================================
# CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"

MODEL_PATH = ROOT_DIR / "models" / "face_landmarker.task"

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "closed_eyes_evaluation.csv"

MAX_IMAGES = 100

# À ajuster selon le seuil réellement utilisé
# par ton EyeClosureDetector.
EAR_THRESHOLD = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==========================================
# MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)


# ==========================================
# CHARGEMENT DES IMAGES
# ==========================================

images = sorted([
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
])

images = images[:MAX_IMAGES]

if not images:
    raise RuntimeError(
        f"Aucune image trouvée dans {DATASET_DIR}"
    )

print(f"Nombre d'images sélectionnées : {len(images)}")


# ==========================================
# EVALUATION
# ==========================================

results = []

detected_faces = 0
detected_closed = 0
failed_images = 0

ear_values = []

with FaceLandmarker.create_from_options(options) as landmarker:

    for index, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            failed_images += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        detection = landmarker.detect(mp_image)

        if not detection.face_landmarks:
            results.append({
                "image": image_path.name,
                "face_detected": False,
                "left_ear": "",
                "right_ear": "",
                "mean_ear": "",
                "prediction": "FACE_NOT_DETECTED"
            })

            print(f"[{index}] Visage non détecté")
            continue

        detected_faces += 1

        landmarks = detection.face_landmarks[0]

        height, width = image.shape[:2]

        left_points = get_eye_points(
            landmarks, LEFT_EYE, width, height
        )

        right_points = get_eye_points(
            landmarks, RIGHT_EYE, width, height
        )

        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)

        mean_ear = (left_ear + right_ear) / 2

        ear_values.append(mean_ear)

        prediction = (
            "CLOSED"
            if mean_ear < EAR_THRESHOLD
            else "OPEN"
        )

        if prediction == "CLOSED":
            detected_closed += 1

        results.append({
            "image": image_path.name,
            "face_detected": True,
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "mean_ear": round(mean_ear, 4),
            "prediction": prediction
        })

        print(
            f"[{index}/{len(images)}] "
            f"EAR={mean_ear:.4f} | {prediction}"
        )


# ==========================================
# SAUVEGARDE CSV
# ==========================================

fieldnames = [
    "image",
    "face_detected",
    "left_ear",
    "right_ear",
    "mean_ear",
    "prediction"
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# STATISTIQUES
# ==========================================

total = len(images)
evaluated = len(ear_values)

print("\n========== EVALUATION RESULTS ==========")

print(f"Images sélectionnées : {total}")
print(f"Images illisibles : {failed_images}")
print(f"Visages détectés : {detected_faces}")
print(f"Images avec EAR calculé : {evaluated}")
print(f"Images classées CLOSED : {detected_closed}")

if evaluated:
    print(
        f"Images classées OPEN : "
        f"{evaluated - detected_closed}"
    )

    print(
        f"EAR moyen : {np.mean(ear_values):.4f}"
    )

    print(
        f"EAR minimum : {np.min(ear_values):.4f}"
    )

    print(
        f"EAR maximum : {np.max(ear_values):.4f}"
    )

    print(
        f"Taux de détection des visages : "
        f"{detected_faces / total * 100:.2f}%"
    )

    print(
        f"Taux de classification CLOSED parmi "
        f"les EAR calculés : "
        f"{detected_closed / evaluated * 100:.2f}%"
    )

print(f"\nRésultats CSV : {CSV_PATH}")
