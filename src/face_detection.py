
import cv2
import math


# Indices des landmarks des yeux
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Indices des landmarks de la bouche

MOUTH = {
    "upper_lip_inner": 13,
    "lower_lip_inner": 14,

    "upper_lip_outer": 12,
    "lower_lip_outer": 15,

    "left_corner": 78,
    "right_corner": 308
}

def get_eye_points(landmarks, eye_indices, width, height):
    """
    Récupère les coordonnées des points d'un œil.
    """
    points = []

    for index in eye_indices:
        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return points


def get_mouth_points(landmarks, width, height):
    """
    Récupère les points principaux de la bouche.
    """
    points = {}

    for name, index in MOUTH.items():
        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points[name] = (x, y)

    return points


def draw_eye_landmarks(frame, points, color=(0, 255, 255)):
    """
    Affiche les points d'un œil.
    """
    for point in points:
        cv2.circle(frame, point, 3, color, -1)


def draw_mouth_landmarks(frame, points):

    # Points intérieurs : jaune
    for name in ["upper_lip_inner", "lower_lip_inner"]:
        cv2.circle(frame, points[name], 4, (0, 255, 255), -1)

    # Points extérieurs : rouge
    for name in ["upper_lip_outer", "lower_lip_outer"]:
        cv2.circle(frame, points[name], 4, (0, 0, 255), -1)

    # Coins de la bouche : magenta
    for name in ["left_corner", "right_corner"]:
        cv2.circle(frame, points[name], 4, (255, 0, 255), -1)



def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio (MAR).

    mouth_points contains:
    - upper_lip_inner
    - lower_lip_inner
    - left_corner
    - right_corner
    """

    upper = mouth_points["upper_lip_inner"]
    lower = mouth_points["lower_lip_inner"]
    left = mouth_points["left_corner"]
    right = mouth_points["right_corner"]

    vertical_distance = math.dist(upper, lower)
    horizontal_distance = math.dist(left, right)

    if horizontal_distance == 0:
        return 0.0

    return vertical_distance / horizontal_distance


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).

    eye_points must contain 6 points in this order:
    [P1, P2, P3, P4, P5, P6]
    """

    if len(eye_points) != 6:
        raise ValueError(
            "EAR requires exactly 6 eye landmarks."
        )

    p1, p2, p3, p4, p5, p6 = eye_points

    # Vertical distances
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    # Horizontal distance
    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear
