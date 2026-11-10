
import cv2
import numpy as np


# --------------------------------------------------
# 1. MediaPipe facial landmarks
# --------------------------------------------------

FACE_POINTS = [1, 152, 33, 263, 61, 291]


# --------------------------------------------------
# 2. Approximate 3D face model
# --------------------------------------------------

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nez
    (0.0, -330.0, -65.0),     # Menton
    (-225.0, 170.0, -135.0),  # Coin externe oeil
    (225.0, 170.0, -135.0),   # Coin externe oeil
    (-150.0, -150.0, -125.0), # Coin gauche bouche
    (150.0, -150.0, -125.0)   # Coin droit bouche
], dtype=np.float64)


# --------------------------------------------------
# 3. Head pose estimation
# --------------------------------------------------

def estimate_head_pose(landmarks, width, height):

    if landmarks is None or len(landmarks) <= max(FACE_POINTS):
        return None

    image_points = []

    for index in FACE_POINTS:
        landmark = landmarks[index]

        x = landmark.x * width
        y = landmark.y * height

        image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float64)

    # Camera matrix
    focal_length = width
    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion_coefficients = np.zeros((4, 1))

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    if not np.all(np.isfinite(rotation_matrix)):
        return None

    return {
        "rotation_matrix": rotation_matrix
    }


# --------------------------------------------------
# 4. Relative head pose calculation
# --------------------------------------------------

def calculate_relative_pose(current_rotation, reference_rotation):

    relative_rotation = current_rotation @ reference_rotation.T

    angles = cv2.RQDecomp3x3(relative_rotation)[0]

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    if not np.all(np.isfinite([yaw, pitch, roll])):
        return None

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll
    }
